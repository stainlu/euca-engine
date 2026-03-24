//! Dynamic entity visibility — configurable per-observer filtering.
//!
//! Components: [`ViewFilter`], [`Tags`], [`VisibleTo`].
//! System: [`visibility_system`].
//!
//! Attach a `ViewFilter` to any observer entity (player, camera) to control
//! what that observer can see. Each tick, `visibility_system` evaluates every
//! `ViewFilter` against all world entities and writes the resulting
//! `VisibleTo` component on each observed entity.

use std::collections::HashSet;

use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_scene::{LocalTransform, SpatialIndex};

use crate::teams::Team;

// ── Components ──────────────────────────────────────────────────────────

/// Filter rules attached to an observer entity. All rules must pass (AND
/// semantics) for a target entity to be visible to this observer.
#[derive(Clone, Debug)]
pub struct ViewFilter {
    pub rules: Vec<VisibilityRule>,
}

impl ViewFilter {
    /// Create a view filter with the given rules.
    pub fn new(rules: Vec<VisibilityRule>) -> Self {
        Self { rules }
    }

    /// A filter that sees everything (no rules = always visible).
    pub fn see_all() -> Self {
        Self {
            rules: vec![VisibilityRule::Always],
        }
    }
}

/// A single visibility rule. Rules compose with AND semantics inside a
/// `ViewFilter`.
#[derive(Clone, Debug)]
pub enum VisibilityRule {
    /// Only see entities within `radius` world units of the observer.
    WithinRadius { radius: f32 },
    /// Only see entities on the same team as the observer (uses [`Team`]).
    SameTeam,
    /// Only see entities inside a specific zone entity's area.
    /// The zone entity must have a `LocalTransform` and a [`ZoneRadius`] component.
    InZone { zone_entity: Entity },
    /// Only see entities that carry a [`Tags`] component containing this tag.
    HasTag { tag: String },
    /// Invert a rule: the target is visible only when the inner rule *fails*.
    Not(Box<VisibilityRule>),
    /// Always visible — no filtering. Useful as an explicit default.
    Always,
}

/// A radius marker for zone-based visibility checks (used with `InZone` rule).
#[derive(Clone, Copy, Debug)]
pub struct ZoneRadius(pub f32);

/// Arbitrary string tags on an entity (e.g. `"stealth"`, `"revealed"`, `"structure"`).
#[derive(Clone, Debug, Default)]
pub struct Tags(pub HashSet<String>);

impl Tags {
    pub fn new() -> Self {
        Self(HashSet::new())
    }

    pub fn with(mut self, tag: impl Into<String>) -> Self {
        self.0.insert(tag.into());
        self
    }

    pub fn insert(&mut self, tag: impl Into<String>) {
        self.0.insert(tag.into());
    }

    pub fn remove(&mut self, tag: &str) -> bool {
        self.0.remove(tag)
    }

    pub fn contains(&self, tag: &str) -> bool {
        self.0.contains(tag)
    }
}

/// Computed each tick by [`visibility_system`]: the set of observer entities
/// that can currently see this entity.
#[derive(Clone, Debug, Default)]
pub struct VisibleTo(pub HashSet<Entity>);

// ── System ──────────────────────────────────────────────────────────────

/// Evaluate all [`ViewFilter`]s against all world entities and write
/// [`VisibleTo`] on each entity.
///
/// Uses [`SpatialIndex`] for efficient radius queries when available.
pub fn visibility_system(world: &mut World) {
    // 1. Collect observers (entities with ViewFilter).
    let observers: Vec<(Entity, ViewFilter)> = {
        let q = Query::<(Entity, &ViewFilter)>::new(world);
        q.iter().map(|(e, vf)| (e, vf.clone())).collect()
    };

    if observers.is_empty() {
        return;
    }

    // 2. Collect all entities with positions (candidates for visibility).
    let candidates: Vec<(Entity, Vec3)> = {
        let q = Query::<(Entity, &LocalTransform)>::new(world);
        q.iter().map(|(e, lt)| (e, lt.0.translation)).collect()
    };

    // 3. Pre-fetch observer positions and teams.
    let observer_data: Vec<(Entity, Vec3, Option<u8>, ViewFilter)> = observers
        .into_iter()
        .map(|(e, vf)| {
            let pos = world
                .get::<LocalTransform>(e)
                .map(|lt| lt.0.translation)
                .unwrap_or(Vec3::ZERO);
            let team = world.get::<Team>(e).map(|t| t.0);
            (e, pos, team, vf)
        })
        .collect();

    // 4. Pre-fetch zone data for InZone rules.
    let zone_data: Vec<(Entity, Vec3, f32)> = {
        let q = Query::<(Entity, &ZoneRadius, &LocalTransform)>::new(world);
        q.iter()
            .map(|(e, zr, lt)| (e, lt.0.translation, zr.0))
            .collect()
    };

    // 5. Build visibility map: target_entity -> set of observers who can see it.
    let mut visibility_map: std::collections::HashMap<Entity, HashSet<Entity>> =
        std::collections::HashMap::new();

    // Index candidates by Entity for O(1) lookup when narrowing via SpatialIndex.
    let candidate_positions: std::collections::HashMap<Entity, Vec3> =
        candidates.iter().copied().collect();

    // Pre-compute spatial index queries for each observer that has a radius rule.
    // We do this while we have an immutable borrow on the world, then use the
    // results in the main loop where we need mutable access.
    let spatial_candidates: Vec<Option<Vec<Entity>>> =
        if let Some(si) = world.resource::<SpatialIndex>() {
            observer_data
                .iter()
                .map(|(_, obs_pos, _, filter)| {
                    extract_radius(&filter.rules).map(|radius| si.query_radius(*obs_pos, radius))
                })
                .collect()
        } else {
            observer_data.iter().map(|_| None).collect()
        };

    for (i, (observer, obs_pos, obs_team, filter)) in observer_data.iter().enumerate() {
        // If we pre-computed a spatial query for this observer, use it to
        // narrow the candidate set. Otherwise, check all candidates.
        let candidates_for_observer: Vec<(Entity, Vec3)> =
            if let Some(ref spatial_hits) = spatial_candidates[i] {
                spatial_hits
                    .iter()
                    .filter_map(|e| candidate_positions.get(e).map(|pos| (*e, *pos)))
                    .collect()
            } else {
                candidates.clone()
            };

        for (target, target_pos) in &candidates_for_observer {
            // Don't evaluate visibility for the observer itself.
            if *target == *observer {
                visibility_map.entry(*target).or_default().insert(*observer);
                continue;
            }

            let visible = evaluate_rules(
                &filter.rules,
                *obs_pos,
                *obs_team,
                *target,
                *target_pos,
                world,
                &zone_data,
            );

            if visible {
                visibility_map.entry(*target).or_default().insert(*observer);
            }
        }
    }

    // 6. Write VisibleTo components.
    // First, clear VisibleTo on all entities that had it previously.
    let existing_visible_to: Vec<Entity> = {
        let q = Query::<Entity, euca_ecs::With<VisibleTo>>::new(world);
        q.iter().collect()
    };
    for e in existing_visible_to {
        if let Some(vt) = world.get_mut::<VisibleTo>(e) {
            vt.0.clear();
        }
    }

    // Write new visibility data.
    for (entity, observers) in visibility_map {
        if let Some(vt) = world.get_mut::<VisibleTo>(entity) {
            vt.0 = observers;
        } else {
            world.insert(entity, VisibleTo(observers));
        }
    }
}

/// Returns the `WithinRadius` value from a rule set, if present.
/// Used to pre-filter candidates via `SpatialIndex` before evaluating
/// the full rule set.
fn extract_radius(rules: &[VisibilityRule]) -> Option<f32> {
    let mut radius = None;
    for rule in rules {
        if let VisibilityRule::WithinRadius { radius: r } = rule {
            radius = Some(*r);
        }
    }
    radius
}

/// Evaluate all rules (AND semantics). Returns `true` if the target is
/// visible to the observer.
fn evaluate_rules(
    rules: &[VisibilityRule],
    obs_pos: Vec3,
    obs_team: Option<u8>,
    target: Entity,
    target_pos: Vec3,
    world: &World,
    zone_data: &[(Entity, Vec3, f32)],
) -> bool {
    for rule in rules {
        if !evaluate_single_rule(
            rule, obs_pos, obs_team, target, target_pos, world, zone_data,
        ) {
            return false;
        }
    }
    true
}

fn evaluate_single_rule(
    rule: &VisibilityRule,
    obs_pos: Vec3,
    obs_team: Option<u8>,
    target: Entity,
    target_pos: Vec3,
    world: &World,
    zone_data: &[(Entity, Vec3, f32)],
) -> bool {
    match rule {
        VisibilityRule::Always => true,

        VisibilityRule::WithinRadius { radius } => {
            let diff = target_pos - obs_pos;
            diff.length_squared() <= radius * radius
        }

        VisibilityRule::SameTeam => {
            let target_team = world.get::<Team>(target).map(|t| t.0);
            match (obs_team, target_team) {
                (Some(a), Some(b)) => a == b,
                // If either lacks a team, rule fails.
                _ => false,
            }
        }

        VisibilityRule::InZone { zone_entity } => {
            // Find the zone in our pre-fetched data.
            zone_data
                .iter()
                .find(|(e, _, _)| *e == *zone_entity)
                .map(|(_, zone_pos, zone_radius)| {
                    let diff = target_pos - *zone_pos;
                    diff.length_squared() <= zone_radius * zone_radius
                })
                .unwrap_or(false)
        }

        VisibilityRule::HasTag { tag } => world
            .get::<Tags>(target)
            .map(|tags| tags.contains(tag))
            .unwrap_or(false),

        VisibilityRule::Not(inner) => !evaluate_single_rule(
            inner, obs_pos, obs_team, target, target_pos, world, zone_data,
        ),
    }
}

// ── Rule parsing (for CLI / HTTP) ───────────────────────────────────────

/// Parse a rule string like `"within:8"`, `"same-team"`, `"tag:stealth"`,
/// `"not:same-team"`, `"always"`, `"zone:5"`.
pub fn parse_rule(s: &str) -> Option<VisibilityRule> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("always") {
        return Some(VisibilityRule::Always);
    }
    if s.eq_ignore_ascii_case("same-team") {
        return Some(VisibilityRule::SameTeam);
    }
    if let Some(rest) = s.strip_prefix("within:") {
        let radius = rest.parse::<f32>().ok()?;
        return Some(VisibilityRule::WithinRadius { radius });
    }
    if let Some(rest) = s.strip_prefix("tag:") {
        return Some(VisibilityRule::HasTag {
            tag: rest.to_string(),
        });
    }
    if let Some(rest) = s.strip_prefix("zone:") {
        let id = rest.parse::<u32>().ok()?;
        return Some(VisibilityRule::InZone {
            zone_entity: Entity::from_raw(id, 0),
        });
    }
    if let Some(rest) = s.strip_prefix("not:") {
        let inner = parse_rule(rest)?;
        return Some(VisibilityRule::Not(Box::new(inner)));
    }
    None
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    fn spawn_at(world: &mut World, pos: Vec3) -> Entity {
        world.spawn(LocalTransform(Transform::from_translation(pos)))
    }

    #[test]
    fn always_rule_sees_everything() {
        let mut world = World::new();

        let observer = spawn_at(&mut world, Vec3::ZERO);
        world.insert(observer, ViewFilter::see_all());

        let target = spawn_at(&mut world, Vec3::new(1000.0, 0.0, 0.0));

        visibility_system(&mut world);

        let vt = world.get::<VisibleTo>(target).unwrap();
        assert!(vt.0.contains(&observer));
    }

    #[test]
    fn within_radius_filters_by_distance() {
        let mut world = World::new();

        let observer = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            observer,
            ViewFilter::new(vec![VisibilityRule::WithinRadius { radius: 5.0 }]),
        );

        let close = spawn_at(&mut world, Vec3::new(3.0, 0.0, 0.0));
        let far = spawn_at(&mut world, Vec3::new(10.0, 0.0, 0.0));

        visibility_system(&mut world);

        let close_vt = world.get::<VisibleTo>(close).unwrap();
        assert!(close_vt.0.contains(&observer));

        // Far entity should either not have VisibleTo or not contain observer.
        let far_visible = world
            .get::<VisibleTo>(far)
            .map(|vt| vt.0.contains(&observer))
            .unwrap_or(false);
        assert!(!far_visible);
    }

    #[test]
    fn same_team_filters_correctly() {
        let mut world = World::new();

        let observer = spawn_at(&mut world, Vec3::ZERO);
        world.insert(observer, Team(1));
        world.insert(observer, ViewFilter::new(vec![VisibilityRule::SameTeam]));

        let ally = spawn_at(&mut world, Vec3::new(5.0, 0.0, 0.0));
        world.insert(ally, Team(1));

        let enemy = spawn_at(&mut world, Vec3::new(5.0, 0.0, 0.0));
        world.insert(enemy, Team(2));

        visibility_system(&mut world);

        let ally_vt = world.get::<VisibleTo>(ally).unwrap();
        assert!(ally_vt.0.contains(&observer));

        let enemy_visible = world
            .get::<VisibleTo>(enemy)
            .map(|vt| vt.0.contains(&observer))
            .unwrap_or(false);
        assert!(!enemy_visible);
    }

    #[test]
    fn tag_filter_works() {
        let mut world = World::new();

        let observer = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            observer,
            ViewFilter::new(vec![VisibilityRule::HasTag {
                tag: "revealed".to_string(),
            }]),
        );

        let tagged = spawn_at(&mut world, Vec3::new(5.0, 0.0, 0.0));
        world.insert(tagged, Tags::new().with("revealed"));

        let untagged = spawn_at(&mut world, Vec3::new(5.0, 0.0, 0.0));

        visibility_system(&mut world);

        let tagged_vt = world.get::<VisibleTo>(tagged).unwrap();
        assert!(tagged_vt.0.contains(&observer));

        let untagged_visible = world
            .get::<VisibleTo>(untagged)
            .map(|vt| vt.0.contains(&observer))
            .unwrap_or(false);
        assert!(!untagged_visible);
    }

    #[test]
    fn not_rule_inverts() {
        let mut world = World::new();

        // Observer can see entities that are NOT on the same team (enemies only).
        let observer = spawn_at(&mut world, Vec3::ZERO);
        world.insert(observer, Team(1));
        world.insert(
            observer,
            ViewFilter::new(vec![VisibilityRule::Not(Box::new(
                VisibilityRule::SameTeam,
            ))]),
        );

        let ally = spawn_at(&mut world, Vec3::new(5.0, 0.0, 0.0));
        world.insert(ally, Team(1));

        let enemy = spawn_at(&mut world, Vec3::new(5.0, 0.0, 0.0));
        world.insert(enemy, Team(2));

        visibility_system(&mut world);

        // Ally should NOT be visible (SameTeam passes, Not inverts it).
        let ally_visible = world
            .get::<VisibleTo>(ally)
            .map(|vt| vt.0.contains(&observer))
            .unwrap_or(false);
        assert!(!ally_visible);

        // Enemy SHOULD be visible (SameTeam fails, Not inverts it).
        let enemy_vt = world.get::<VisibleTo>(enemy).unwrap();
        assert!(enemy_vt.0.contains(&observer));
    }

    #[test]
    fn composed_rules_use_and_semantics() {
        let mut world = World::new();

        // Observer can see same-team entities within radius 10.
        let observer = spawn_at(&mut world, Vec3::ZERO);
        world.insert(observer, Team(1));
        world.insert(
            observer,
            ViewFilter::new(vec![
                VisibilityRule::SameTeam,
                VisibilityRule::WithinRadius { radius: 10.0 },
            ]),
        );

        // Same team + close => visible.
        let ally_close = spawn_at(&mut world, Vec3::new(5.0, 0.0, 0.0));
        world.insert(ally_close, Team(1));

        // Same team + far => not visible.
        let ally_far = spawn_at(&mut world, Vec3::new(50.0, 0.0, 0.0));
        world.insert(ally_far, Team(1));

        // Different team + close => not visible.
        let enemy_close = spawn_at(&mut world, Vec3::new(5.0, 0.0, 0.0));
        world.insert(enemy_close, Team(2));

        visibility_system(&mut world);

        assert!(
            world
                .get::<VisibleTo>(ally_close)
                .unwrap()
                .0
                .contains(&observer)
        );

        assert!(
            !world
                .get::<VisibleTo>(ally_far)
                .map(|vt| vt.0.contains(&observer))
                .unwrap_or(false)
        );

        assert!(
            !world
                .get::<VisibleTo>(enemy_close)
                .map(|vt| vt.0.contains(&observer))
                .unwrap_or(false)
        );
    }

    #[test]
    fn backward_compat_no_filters_means_everything_visible() {
        let mut world = World::new();

        // No ViewFilter entities exist — system should be a no-op.
        let a = spawn_at(&mut world, Vec3::ZERO);
        let b = spawn_at(&mut world, Vec3::new(10.0, 0.0, 0.0));

        visibility_system(&mut world);

        // No VisibleTo should be written when there are no observers.
        assert!(world.get::<VisibleTo>(a).is_none());
        assert!(world.get::<VisibleTo>(b).is_none());
    }

    #[test]
    fn in_zone_rule_works() {
        let mut world = World::new();

        // Create a zone entity at (10, 0, 10) with radius 5.
        let zone = spawn_at(&mut world, Vec3::new(10.0, 0.0, 10.0));
        world.insert(zone, ZoneRadius(5.0));

        let observer = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            observer,
            ViewFilter::new(vec![VisibilityRule::InZone { zone_entity: zone }]),
        );

        // Entity inside zone.
        let inside = spawn_at(&mut world, Vec3::new(12.0, 0.0, 10.0));
        // Entity outside zone.
        let outside = spawn_at(&mut world, Vec3::new(100.0, 0.0, 0.0));

        visibility_system(&mut world);

        assert!(
            world
                .get::<VisibleTo>(inside)
                .unwrap()
                .0
                .contains(&observer)
        );

        assert!(
            !world
                .get::<VisibleTo>(outside)
                .map(|vt| vt.0.contains(&observer))
                .unwrap_or(false)
        );
    }

    #[test]
    fn parse_rule_strings() {
        assert!(matches!(parse_rule("always"), Some(VisibilityRule::Always)));
        assert!(matches!(
            parse_rule("same-team"),
            Some(VisibilityRule::SameTeam)
        ));
        assert!(matches!(
            parse_rule("within:8"),
            Some(VisibilityRule::WithinRadius { radius }) if (radius - 8.0).abs() < f32::EPSILON
        ));
        assert!(matches!(
            parse_rule("tag:stealth"),
            Some(VisibilityRule::HasTag { tag }) if tag == "stealth"
        ));
        assert!(matches!(
            parse_rule("not:same-team"),
            Some(VisibilityRule::Not(_))
        ));
        assert!(parse_rule("unknown").is_none());
    }
}
