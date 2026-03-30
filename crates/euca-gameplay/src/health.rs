//! Health and damage — the foundation of combat.
//!
//! Components: `Health`, `Dead` (marker).
//! Events: `DamageEvent`, `DeathEvent`.
//! Systems: `apply_damage_system`, `death_check_system`.

use euca_ecs::{Entity, Events, Query, World};

use crate::combat_math::{self, DamageType};
use crate::stats::DamageResistance;

/// Entity has hit points that can be reduced by damage or restored by healing.
///
/// Damage pipeline: `DamageEvent` -> `apply_damage_system` reduces `current`
/// -> `death_check_system` adds `Dead` marker + emits `DeathEvent` when `current <= 0`.
#[derive(Clone, Debug)]
pub struct Health {
    /// Current hit points (clamped to `[0, max]` by systems).
    pub current: f32,
    /// Maximum hit points. Healing cannot exceed this value.
    pub max: f32,
}

impl Health {
    /// Create a fully healed entity with the given maximum HP.
    pub fn new(max: f32) -> Self {
        Self { current: max, max }
    }

    /// Returns `true` when current HP is zero or below.
    pub fn is_dead(&self) -> bool {
        self.current <= 0.0
    }

    /// Health as a fraction of max, clamped to `[0.0, 1.0]`.
    pub fn fraction(&self) -> f32 {
        if self.max > 0.0 {
            (self.current / self.max).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Physical armor value used by Dota 2's armor reduction formula.
///
/// Positive values reduce physical damage; negative values amplify it.
/// See [`combat_math::armor_reduction`] for the exact formula.
#[derive(Clone, Copy, Debug, Default)]
pub struct Armor(pub f32);

/// Multiplicatively-stacking magic resistance sources.
///
/// Each entry is a fraction (e.g. `0.25` for 25% base magic resistance).
/// Multiple sources stack multiplicatively via [`combat_math::magic_resistance_stacked`].
#[derive(Clone, Debug, Default)]
pub struct MagicResistances(pub Vec<f32>);

/// Marker component: entity has died (health reached 0).
/// Stays until respawn or despawn.
#[derive(Clone, Copy, Debug)]
pub struct Dead;

/// Tracks who last dealt damage to this entity (for kill attribution).
#[derive(Clone, Copy, Debug)]
pub struct LastAttacker(pub Option<Entity>);

/// Request to apply damage to an entity. Consumed by `apply_damage_system`.
#[derive(Clone, Debug)]
pub struct DamageEvent {
    /// Entity to receive damage.
    pub target: Entity,
    /// Raw damage amount (before resistance mitigation).
    pub amount: f32,
    /// Who dealt the damage (used for kill attribution via `LastAttacker`).
    pub source: Option<Entity>,
    /// Damage category (e.g. "physical", "magical", "true").
    /// Not an enum — games define categories as data.
    /// Defaults to "physical". The category "true" bypasses all resistance.
    /// Retained for backward compatibility with `DamageResistance`.
    pub category: String,
    /// Typed damage classification used by `combat_math::apply_damage()`.
    /// When the target has [`Armor`] or [`MagicResistances`] components,
    /// this field determines which formula is applied.
    pub damage_type: DamageType,
}

impl DamageEvent {
    /// Create a damage event with default category ("physical") and `DamageType::Physical`.
    pub fn new(target: Entity, amount: f32, source: Option<Entity>) -> Self {
        Self {
            target,
            amount,
            source,
            category: "physical".to_string(),
            damage_type: DamageType::Physical,
        }
    }

    /// Create a damage event with a specific damage category.
    ///
    /// The `damage_type` is inferred from the category string:
    /// - `"magical"` -> `DamageType::Magical`
    /// - `"pure"` or `"true"` -> `DamageType::Pure`
    /// - anything else -> `DamageType::Physical`
    pub fn with_category(
        target: Entity,
        amount: f32,
        source: Option<Entity>,
        category: impl Into<String>,
    ) -> Self {
        let category = category.into();
        let damage_type = category_to_damage_type(&category);
        Self {
            target,
            amount,
            source,
            category,
            damage_type,
        }
    }

    /// Create a damage event with an explicit `DamageType`.
    pub fn typed(
        target: Entity,
        amount: f32,
        source: Option<Entity>,
        damage_type: DamageType,
    ) -> Self {
        let category = match damage_type {
            DamageType::Physical => "physical",
            DamageType::Magical => "magical",
            DamageType::Pure => "pure",
            DamageType::HpRemoval => "true",
        }
        .to_string();
        Self {
            target,
            amount,
            source,
            category,
            damage_type,
        }
    }
}

/// Map a category string to the corresponding `DamageType`.
fn category_to_damage_type(category: &str) -> DamageType {
    match category {
        "magical" => DamageType::Magical,
        "pure" | "true" => DamageType::Pure,
        _ => DamageType::Physical,
    }
}

/// Notification that an entity has died. Emitted by `death_check_system`.
#[derive(Clone, Debug)]
pub struct DeathEvent {
    /// The entity that died.
    pub entity: Entity,
    /// The entity that dealt the killing blow (from `LastAttacker`), if any.
    pub killer: Option<Entity>,
}

/// Apply pending damage events to Health components.
///
/// Damage reduction priority:
/// 1. If the target has [`Armor`] or [`MagicResistances`] components, the
///    Dota 2 formulas from [`combat_math::apply_damage`] are used based on
///    `damage_type`.
/// 2. Otherwise, if the target has a legacy [`DamageResistance`] component,
///    the old formula `effective = raw * (100 / (100 + resistance))` is used
///    based on the `category` string. The category `"true"` bypasses this.
/// 3. If the target has none of these, full damage is dealt.
pub fn apply_damage_system(world: &mut World) {
    // Collect events first to avoid borrow conflicts
    let events: Vec<DamageEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<DamageEvent>().cloned().collect())
        .unwrap_or_default();

    for event in events {
        let target = event.target;

        // Prefer the new combat_math path when the target has Armor or MagicResistances.
        let has_new_components =
            world.get::<Armor>(target).is_some() || world.get::<MagicResistances>(target).is_some();

        let effective_damage = if has_new_components {
            let armor = world.get::<Armor>(target).map(|a| a.0).unwrap_or(0.0);
            let mr = world.get::<MagicResistances>(target);
            let empty: Vec<f32> = Vec::new();
            let resistances = mr.map(|m| m.0.as_slice()).unwrap_or(&empty);
            combat_math::apply_damage(event.amount, event.damage_type, armor, resistances)
        } else {
            // Legacy path: use DamageResistance with category string.
            compute_effective_damage_legacy(
                event.amount,
                &event.category,
                world.get::<DamageResistance>(target),
            )
        };

        if let Some(health) = world.get_mut::<Health>(target) {
            health.current = (health.current - effective_damage).max(0.0);
        }
        // Track who dealt this damage for kill attribution
        if event.source.is_some() {
            world.insert(target, LastAttacker(event.source));
        }
    }
}

/// Legacy damage reduction using the old `DamageResistance` component.
///
/// Formula: `effective = raw * (100.0 / (100.0 + resistance))`.
/// - If `category` is `"true"`, resistance is bypassed (full damage).
/// - If no `DamageResistance` component or no entry for the category, full damage.
fn compute_effective_damage_legacy(
    raw: f32,
    category: &str,
    resistance: Option<&DamageResistance>,
) -> f32 {
    if category == "true" {
        return raw;
    }

    let Some(dr) = resistance else {
        return raw;
    };

    let resist_value = dr.0.get(category).copied().unwrap_or(0.0);
    if resist_value <= 0.0 {
        return raw;
    }

    raw * (100.0 / (100.0 + resist_value)) as f32
}

/// Check for entities with Health <= 0 that aren't already Dead.
/// Adds the Dead marker and emits DeathEvent.
pub fn death_check_system(world: &mut World) {
    // Find entities that just died
    let newly_dead: Vec<(Entity, Option<Entity>)> = {
        let query = Query::<(Entity, &Health)>::new(world);
        query
            .iter()
            .filter(|(e, h)| h.is_dead() && world.get::<Dead>(*e).is_none())
            .map(|(e, _)| {
                let killer = world.get::<LastAttacker>(e).and_then(|la| la.0);
                (e, killer)
            })
            .collect()
    };

    for (entity, killer) in &newly_dead {
        world.insert(*entity, Dead);
        if let Some(events) = world.resource_mut::<Events>() {
            events.send(DeathEvent {
                entity: *entity,
                killer: *killer,
            });
        }
    }
}

/// Apply healing to an entity's Health (clamped to max).
pub fn heal(world: &mut World, entity: Entity, amount: f32) {
    if let Some(health) = world.get_mut::<Health>(entity) {
        health.current = (health.current + amount).min(health.max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_creation() {
        let h = Health::new(100.0);
        assert_eq!(h.current, 100.0);
        assert_eq!(h.max, 100.0);
        assert!(!h.is_dead());
        assert_eq!(h.fraction(), 1.0);
    }

    #[test]
    fn health_dead_at_zero() {
        let h = Health {
            current: 0.0,
            max: 100.0,
        };
        assert!(h.is_dead());
        assert_eq!(h.fraction(), 0.0);
    }

    #[test]
    fn apply_damage_reduces_health() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(100.0));

        // Send damage event
        world
            .resource_mut::<Events>()
            .unwrap()
            .send(DamageEvent::new(entity, 30.0, None));

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 70.0);
    }

    #[test]
    fn damage_cannot_go_below_zero() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(50.0));

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(DamageEvent::new(entity, 999.0, None));

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 0.0);
    }

    #[test]
    fn death_check_marks_dead() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });

        assert!(world.get::<Dead>(entity).is_none());

        death_check_system(&mut world);

        assert!(world.get::<Dead>(entity).is_some());
    }

    #[test]
    fn death_check_ignores_alive() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(50.0));

        death_check_system(&mut world);

        assert!(world.get::<Dead>(entity).is_none());
    }

    #[test]
    fn death_check_does_not_re_mark() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.insert(entity, Dead);

        // Should not panic or double-mark
        death_check_system(&mut world);
        assert!(world.get::<Dead>(entity).is_some());
    }

    #[test]
    fn heal_restores_health() {
        let mut world = World::new();
        let entity = world.spawn(Health {
            current: 30.0,
            max: 100.0,
        });

        heal(&mut world, entity, 50.0);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 80.0);
    }

    #[test]
    fn heal_capped_at_max() {
        let mut world = World::new();
        let entity = world.spawn(Health {
            current: 90.0,
            max: 100.0,
        });

        heal(&mut world, entity, 50.0);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 100.0);
    }

    // ── Damage resistance tests ──

    #[test]
    fn damage_reduced_by_resistance() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(100.0));
        // 50 physical resistance → effective = 100 * (100 / 150) ≈ 66.67
        let mut resist = std::collections::HashMap::new();
        resist.insert("physical".to_string(), 50.0);
        world.insert(entity, DamageResistance(resist));

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(DamageEvent::new(entity, 100.0, None));

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        // 100 - 66.67 = 33.33
        let expected = 100.0 - 100.0 * (100.0 / 150.0) as f32;
        assert!((health.current - expected).abs() < 0.01);
    }

    #[test]
    fn true_damage_bypasses_resistance() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(100.0));
        let mut resist = std::collections::HashMap::new();
        resist.insert("physical".to_string(), 999.0);
        world.insert(entity, DamageResistance(resist));

        // "true" damage ignores all resistance
        world
            .resource_mut::<Events>()
            .unwrap()
            .send(DamageEvent::with_category(entity, 40.0, None, "true"));

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 60.0);
    }

    #[test]
    fn no_resistance_means_full_damage() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        // Entity without DamageResistance
        let entity = world.spawn(Health::new(100.0));

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(DamageEvent::new(entity, 25.0, None));

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 75.0);
    }

    #[test]
    fn unmatched_category_means_full_damage() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(100.0));
        // Has physical resistance but takes magical damage
        let mut resist = std::collections::HashMap::new();
        resist.insert("physical".to_string(), 100.0);
        world.insert(entity, DamageResistance(resist));

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(DamageEvent::with_category(entity, 50.0, None, "magical"));

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 50.0);
    }

    #[test]
    fn backward_compat_default_category_is_physical() {
        let event = DamageEvent::new(euca_ecs::Entity::from_raw(0, 0), 10.0, None);
        assert_eq!(event.category, "physical");
        assert_eq!(event.damage_type, DamageType::Physical);
    }

    // ── combat_math integration tests ──

    #[test]
    fn armor_reduces_physical_damage_dota_formula() {
        // 10 armor → reduction = 0.375 → entity takes 62.5% of physical damage.
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(100.0));
        world.insert(entity, Armor(10.0));

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(DamageEvent::typed(
                entity,
                100.0,
                None,
                DamageType::Physical,
            ));

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        // effective = 100 * (1 - 0.375) = 62.5 → remaining = 37.5
        assert!(
            (health.current - 37.5).abs() < 0.01,
            "10 armor should reduce 100 physical damage to ~62.5 (remaining ~37.5), got {}",
            health.current
        );
    }

    #[test]
    fn magic_resistance_reduces_magical_damage() {
        // 25% magic resistance → entity takes 75% of magical damage.
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(100.0));
        world.insert(entity, MagicResistances(vec![0.25]));

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(DamageEvent::typed(entity, 100.0, None, DamageType::Magical));

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        assert!(
            (health.current - 25.0).abs() < 0.01,
            "25% magic resistance should reduce 100 magical damage to 75 (remaining 25), got {}",
            health.current
        );
    }

    #[test]
    fn pure_damage_ignores_armor_and_magic_resistance() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(100.0));
        world.insert(entity, Armor(50.0));
        world.insert(entity, MagicResistances(vec![0.50]));

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(DamageEvent::typed(entity, 40.0, None, DamageType::Pure));

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(
            health.current, 60.0,
            "Pure damage should ignore both armor and magic resistance"
        );
    }

    #[test]
    fn with_category_infers_damage_type() {
        let phys =
            DamageEvent::with_category(euca_ecs::Entity::from_raw(0, 0), 10.0, None, "physical");
        assert_eq!(phys.damage_type, DamageType::Physical);

        let magic =
            DamageEvent::with_category(euca_ecs::Entity::from_raw(0, 0), 10.0, None, "magical");
        assert_eq!(magic.damage_type, DamageType::Magical);

        let pure = DamageEvent::with_category(euca_ecs::Entity::from_raw(0, 0), 10.0, None, "pure");
        assert_eq!(pure.damage_type, DamageType::Pure);

        let true_dmg =
            DamageEvent::with_category(euca_ecs::Entity::from_raw(0, 0), 10.0, None, "true");
        assert_eq!(true_dmg.damage_type, DamageType::Pure);
    }

    #[test]
    fn legacy_resistance_still_works_without_armor_components() {
        // Entity with old DamageResistance but no Armor/MagicResistances
        // should still use the legacy formula.
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(100.0));
        let mut resist = std::collections::HashMap::new();
        resist.insert("physical".to_string(), 50.0);
        world.insert(entity, DamageResistance(resist));

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(DamageEvent::new(entity, 100.0, None));

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        let expected = 100.0 - 100.0 * (100.0 / 150.0) as f32;
        assert!(
            (health.current - expected).abs() < 0.01,
            "Legacy DamageResistance should still work: expected ~{expected}, got {}",
            health.current
        );
    }
}
