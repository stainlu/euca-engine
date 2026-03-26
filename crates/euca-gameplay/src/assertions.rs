//! Engine-level assertion system — testable expectations as first-class ECS entities.
//!
//! Assertions follow the same pattern as rules: each assertion is an entity
//! with an `Assertion` component. The `evaluate_assertions()` function
//! tests all assertions against the current world state and returns verdicts.
//!
//! ```text
//! euca assert create --name "hero-exists" --condition entity-exists --filter "role:hero"
//! euca assert evaluate
//! → { "results": [{"name": "hero-exists", "passed": true, "message": "Found 2 entities"}] }
//! ```

use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_scene::{GlobalTransform, LocalTransform};
use serde::{Deserialize, Serialize};

use crate::combat::EntityRole;
use crate::health::{Dead, Health};
use crate::teams::{SpawnPoint, Team};
use crate::visibility::Tags;

// ── Core types ──

/// Severity level for assertion failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Must pass — game is broken if this fails.
    Error,
    /// Should pass — degraded experience if this fails.
    Warning,
    /// Informational — nice to have.
    Info,
}

/// Filter for selecting which entities an assertion checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EntityFilter {
    /// Matches all entities.
    Any,
    /// Matches entities on a specific team.
    Team { team: u8 },
    /// Matches entities with a specific role (Hero, Minion, Tower, Structure).
    Role { role: String },
    /// Matches entities with a specific tag.
    Tag { tag: String },
    /// Matches entities that have a specific component.
    HasComponent { component: String },
    /// All filters must match (intersection).
    And { filters: Vec<EntityFilter> },
}

/// What condition to check.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssertCondition {
    /// At least one entity matching the filter exists.
    EntityExists { filter: EntityFilter },
    /// Entity count matching filter is within [min, max].
    EntityCount {
        filter: EntityFilter,
        #[serde(default)]
        min: Option<u32>,
        #[serde(default)]
        max: Option<u32>,
    },
    /// A numeric field on matching entities satisfies a comparison.
    FieldCheck {
        filter: EntityFilter,
        field: String,
        op: CompareOp,
        value: f64,
    },
    /// Every team that has entities also has at least one SpawnPoint.
    AllTeamsHaveSpawnPoints,
    /// No two entities matching the filter are closer than `min_distance`.
    NoOverlap {
        filter: EntityFilter,
        min_distance: f32,
    },
    /// No entity matching the filter has the `Dead` component.
    NoneAreDead { filter: EntityFilter },
    /// Every entity with Health has Health.current > 0 (unless Dead).
    NoZeroHealthAlive,
    /// Every entity with MeshRenderer also has GlobalTransform (i.e. is renderable).
    AllRenderableHaveTransform,
    /// Game is in a specific phase.
    GamePhase { phase: String },
    /// Total entity count is below a budget.
    EntityBudget { max: u32 },
}

/// Comparison operator for field checks.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
    NotEqual,
}

/// An assertion component — attach to an entity to register a testable expectation.
#[derive(Clone, Debug)]
pub struct Assertion {
    /// Human-readable name for this assertion.
    pub name: String,
    /// What to check.
    pub condition: AssertCondition,
    /// How critical a failure is.
    pub severity: Severity,
    /// Last evaluation result (updated by `evaluate_assertions`).
    pub last_result: Option<AssertResult>,
}

/// Result of evaluating a single assertion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssertResult {
    /// Whether the assertion passed.
    pub passed: bool,
    /// Human-readable explanation.
    pub message: String,
    /// World tick when this was evaluated.
    pub tick: u64,
}

/// Aggregate result from evaluating all assertions.
#[derive(Clone, Debug, Serialize)]
pub struct EvaluationReport {
    /// Total assertions evaluated.
    pub total: usize,
    /// Number that passed.
    pub passed: usize,
    /// Number that failed.
    pub failed: usize,
    /// Per-assertion results.
    pub results: Vec<AssertionResult>,
}

/// A single assertion's result with metadata.
#[derive(Clone, Debug, Serialize)]
pub struct AssertionResult {
    /// Entity index of the assertion.
    pub entity_id: u32,
    /// Assertion name.
    pub name: String,
    /// Severity level.
    pub severity: Severity,
    /// Whether it passed.
    pub passed: bool,
    /// Explanation.
    pub message: String,
}

// ── Filter matching ──

/// Returns true if the given entity matches the filter.
pub fn matches_filter(entity: Entity, filter: &EntityFilter, world: &World) -> bool {
    match filter {
        EntityFilter::Any => true,
        EntityFilter::Team { team } => world.get::<Team>(entity).is_some_and(|t| t.0 == *team),
        EntityFilter::Role { role } => {
            let expected = parse_role(role);
            match expected {
                Some(r) => world.get::<EntityRole>(entity).is_some_and(|er| *er == r),
                None => false,
            }
        }
        EntityFilter::Tag { tag } => world
            .get::<Tags>(entity)
            .is_some_and(|tags| tags.0.contains(tag)),
        EntityFilter::HasComponent { component } => has_component(entity, component, world),
        EntityFilter::And { filters } => filters.iter().all(|f| matches_filter(entity, f, world)),
    }
}

fn parse_role(s: &str) -> Option<EntityRole> {
    match s.to_lowercase().as_str() {
        "hero" => Some(EntityRole::Hero),
        "minion" => Some(EntityRole::Minion),
        "tower" => Some(EntityRole::Tower),
        "structure" => Some(EntityRole::Structure),
        _ => None,
    }
}

fn has_component(entity: Entity, name: &str, world: &World) -> bool {
    match name.to_lowercase().as_str() {
        "health" => world.get::<Health>(entity).is_some(),
        "team" => world.get::<Team>(entity).is_some(),
        "dead" => world.get::<Dead>(entity).is_some(),
        "spawnpoint" | "spawn_point" => world.get::<SpawnPoint>(entity).is_some(),
        "globaltransform" | "global_transform" => world.get::<GlobalTransform>(entity).is_some(),
        "localtransform" | "local_transform" => world.get::<LocalTransform>(entity).is_some(),
        "entityrole" | "entity_role" | "role" => world.get::<EntityRole>(entity).is_some(),
        "tags" => world.get::<Tags>(entity).is_some(),
        "velocity" => world.get::<euca_physics::Velocity>(entity).is_some(),
        "meshrenderer" | "mesh_renderer" => {
            world.get::<euca_render::MeshRenderer>(entity).is_some()
        }
        _ => false,
    }
}

// ── Field extraction ──

fn extract_field(entity: Entity, field: &str, world: &World) -> Option<f64> {
    match field.to_lowercase().as_str() {
        "health" | "health.current" => world.get::<Health>(entity).map(|h| h.current as f64),
        "health.max" => world.get::<Health>(entity).map(|h| h.max as f64),
        "health.percent" | "health.pct" => world.get::<Health>(entity).map(|h| {
            if h.max > 0.0 {
                (h.current / h.max * 100.0) as f64
            } else {
                0.0
            }
        }),
        "team" => world.get::<Team>(entity).map(|t| t.0 as f64),
        "position.x" | "pos.x" | "x" => get_position(entity, world).map(|p| p.x as f64),
        "position.y" | "pos.y" | "y" => get_position(entity, world).map(|p| p.y as f64),
        "position.z" | "pos.z" | "z" => get_position(entity, world).map(|p| p.z as f64),
        "gold" => world
            .get::<crate::economy::Gold>(entity)
            .map(|g| g.0 as f64),
        "level" => world
            .get::<crate::leveling::Level>(entity)
            .map(|l| l.level as f64),
        _ => None,
    }
}

fn get_position(entity: Entity, world: &World) -> Option<Vec3> {
    world
        .get::<GlobalTransform>(entity)
        .map(|gt| gt.0.translation)
        .or_else(|| {
            world
                .get::<LocalTransform>(entity)
                .map(|lt| lt.0.translation)
        })
}

fn compare(lhs: f64, op: CompareOp, rhs: f64) -> bool {
    match op {
        CompareOp::Greater => lhs > rhs,
        CompareOp::GreaterEqual => lhs >= rhs,
        CompareOp::Less => lhs < rhs,
        CompareOp::LessEqual => lhs <= rhs,
        CompareOp::Equal => (lhs - rhs).abs() < f64::EPSILON,
        CompareOp::NotEqual => (lhs - rhs).abs() >= f64::EPSILON,
    }
}

fn op_symbol(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Greater => ">",
        CompareOp::GreaterEqual => ">=",
        CompareOp::Less => "<",
        CompareOp::LessEqual => "<=",
        CompareOp::Equal => "==",
        CompareOp::NotEqual => "!=",
    }
}

// ── Condition evaluation ──

fn evaluate_condition(condition: &AssertCondition, world: &World) -> AssertResult {
    let tick = world
        .resource::<crate::game_state::GameState>()
        .map(|gs| gs.elapsed as u64)
        .unwrap_or(0);

    match condition {
        AssertCondition::EntityExists { filter } => {
            let count = count_matching(filter, world);
            if count > 0 {
                AssertResult {
                    passed: true,
                    message: format!("Found {count} matching entities"),
                    tick,
                }
            } else {
                AssertResult {
                    passed: false,
                    message: "No matching entities found".into(),
                    tick,
                }
            }
        }

        AssertCondition::EntityCount { filter, min, max } => {
            let count = count_matching(filter, world);
            let min_ok = min.is_none_or(|m| count >= m);
            let max_ok = max.is_none_or(|m| count <= m);
            if min_ok && max_ok {
                AssertResult {
                    passed: true,
                    message: format!(
                        "Count {count} is within bounds (min={}, max={})",
                        min.map_or("none".into(), |m| m.to_string()),
                        max.map_or("none".into(), |m| m.to_string())
                    ),
                    tick,
                }
            } else {
                AssertResult {
                    passed: false,
                    message: format!(
                        "Count {count} out of bounds (min={}, max={})",
                        min.map_or("none".into(), |m| m.to_string()),
                        max.map_or("none".into(), |m| m.to_string())
                    ),
                    tick,
                }
            }
        }

        AssertCondition::FieldCheck {
            filter,
            field,
            op,
            value,
        } => {
            let query = Query::<Entity>::new(world);
            let mut checked = 0u32;
            let mut failures = Vec::new();
            for entity in query.iter() {
                if !matches_filter(entity, filter, world) {
                    continue;
                }
                if let Some(actual) = extract_field(entity, field, world) {
                    checked += 1;
                    if !compare(actual, *op, *value) {
                        failures.push(format!(
                            "E{}: {field}={actual} (expected {}{value})",
                            entity.index(),
                            op_symbol(*op)
                        ));
                    }
                }
            }
            if failures.is_empty() {
                AssertResult {
                    passed: true,
                    message: format!(
                        "All {checked} entities satisfy {field} {}{value}",
                        op_symbol(*op)
                    ),
                    tick,
                }
            } else {
                AssertResult {
                    passed: false,
                    message: format!(
                        "{} of {checked} failed: {}",
                        failures.len(),
                        failures.join("; ")
                    ),
                    tick,
                }
            }
        }

        AssertCondition::AllTeamsHaveSpawnPoints => {
            let mut teams_seen: std::collections::HashSet<u8> = std::collections::HashSet::new();
            let mut teams_with_spawns: std::collections::HashSet<u8> =
                std::collections::HashSet::new();

            let team_query = Query::<(Entity, &Team)>::new(world);
            for (_, team) in team_query.iter() {
                teams_seen.insert(team.0);
            }

            let sp_query = Query::<(Entity, &SpawnPoint)>::new(world);
            for (_, sp) in sp_query.iter() {
                teams_with_spawns.insert(sp.team);
            }

            let missing: Vec<u8> = teams_seen.difference(&teams_with_spawns).copied().collect();
            if missing.is_empty() {
                AssertResult {
                    passed: true,
                    message: format!("All {} teams have spawn points", teams_seen.len()),
                    tick,
                }
            } else {
                AssertResult {
                    passed: false,
                    message: format!("Teams without spawn points: {:?}", missing),
                    tick,
                }
            }
        }

        AssertCondition::NoOverlap {
            filter,
            min_distance,
        } => {
            let positions: Vec<(Entity, Vec3)> = {
                let query = Query::<Entity>::new(world);
                query
                    .iter()
                    .filter(|e| matches_filter(*e, filter, world))
                    .filter_map(|e| get_position(e, world).map(|p| (e, p)))
                    .collect()
            };

            let mut overlaps = Vec::new();
            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    let dist = (positions[i].1 - positions[j].1).length();
                    if dist < *min_distance {
                        overlaps.push(format!(
                            "E{} and E{} are {dist:.2} apart (min={min_distance})",
                            positions[i].0.index(),
                            positions[j].0.index()
                        ));
                    }
                }
            }

            if overlaps.is_empty() {
                AssertResult {
                    passed: true,
                    message: format!("No overlaps among {} entities", positions.len()),
                    tick,
                }
            } else {
                AssertResult {
                    passed: false,
                    message: format!("{} overlaps: {}", overlaps.len(), overlaps.join("; ")),
                    tick,
                }
            }
        }

        AssertCondition::NoneAreDead { filter } => {
            let query = Query::<Entity>::new(world);
            let mut dead_entities = Vec::new();
            for entity in query.iter() {
                if matches_filter(entity, filter, world) && world.get::<Dead>(entity).is_some() {
                    dead_entities.push(entity.index());
                }
            }
            if dead_entities.is_empty() {
                AssertResult {
                    passed: true,
                    message: "No matching dead entities".into(),
                    tick,
                }
            } else {
                AssertResult {
                    passed: false,
                    message: format!("{} dead entities: {:?}", dead_entities.len(), dead_entities),
                    tick,
                }
            }
        }

        AssertCondition::NoZeroHealthAlive => {
            let query = Query::<(Entity, &Health)>::new(world);
            let mut violations = Vec::new();
            for (entity, health) in query.iter() {
                if health.current <= 0.0 && world.get::<Dead>(entity).is_none() {
                    violations.push(entity.index());
                }
            }
            if violations.is_empty() {
                AssertResult {
                    passed: true,
                    message: "No alive entities with zero health".into(),
                    tick,
                }
            } else {
                AssertResult {
                    passed: false,
                    message: format!(
                        "{} alive entities with health <= 0: {:?}",
                        violations.len(),
                        violations
                    ),
                    tick,
                }
            }
        }

        AssertCondition::AllRenderableHaveTransform => {
            let query = Query::<Entity>::new(world);
            let mut missing = Vec::new();
            for entity in query.iter() {
                if world.get::<euca_render::MeshRenderer>(entity).is_some()
                    && world.get::<GlobalTransform>(entity).is_none()
                {
                    missing.push(entity.index());
                }
            }
            if missing.is_empty() {
                AssertResult {
                    passed: true,
                    message: "All renderable entities have transforms".into(),
                    tick,
                }
            } else {
                AssertResult {
                    passed: false,
                    message: format!(
                        "{} entities with MeshRenderer but no GlobalTransform: {:?}",
                        missing.len(),
                        missing
                    ),
                    tick,
                }
            }
        }

        AssertCondition::GamePhase { phase } => {
            let current =
                world
                    .resource::<crate::game_state::GameState>()
                    .map(|gs| match &gs.phase {
                        crate::game_state::GamePhase::Lobby => "lobby",
                        crate::game_state::GamePhase::Countdown { .. } => "countdown",
                        crate::game_state::GamePhase::Playing => "playing",
                        crate::game_state::GamePhase::PostMatch { .. } => "post_match",
                    });
            match current {
                Some(actual) if actual == phase.as_str() => AssertResult {
                    passed: true,
                    message: format!("Game is in '{phase}' phase"),
                    tick,
                },
                Some(actual) => AssertResult {
                    passed: false,
                    message: format!("Expected phase '{phase}', got '{actual}'"),
                    tick,
                },
                None => AssertResult {
                    passed: false,
                    message: "No GameState resource — game not initialized".into(),
                    tick,
                },
            }
        }

        AssertCondition::EntityBudget { max } => {
            let count = world.entity_count();
            if count <= *max {
                AssertResult {
                    passed: true,
                    message: format!("Entity count {count} within budget of {max}"),
                    tick,
                }
            } else {
                AssertResult {
                    passed: false,
                    message: format!("Entity count {count} exceeds budget of {max}"),
                    tick,
                }
            }
        }
    }
}

fn count_matching(filter: &EntityFilter, world: &World) -> u32 {
    let query = Query::<Entity>::new(world);
    query
        .iter()
        .filter(|e| matches_filter(*e, filter, world))
        .count() as u32
}

// ── Public API ──

/// Evaluate all assertion entities in the world and return an aggregate report.
pub fn evaluate_assertions(world: &mut World) -> EvaluationReport {
    // Collect assertion data (avoids borrow conflict during mutation).
    let assertions: Vec<(Entity, String, AssertCondition, Severity)> = {
        let query = Query::<(Entity, &Assertion)>::new(world);
        query
            .iter()
            .map(|(e, a)| (e, a.name.clone(), a.condition.clone(), a.severity))
            .collect()
    };

    let mut results = Vec::with_capacity(assertions.len());
    let mut passed = 0usize;
    let mut failed = 0usize;

    for (entity, name, condition, severity) in assertions {
        let result = evaluate_condition(&condition, world);

        if result.passed {
            passed += 1;
        } else {
            failed += 1;
        }

        let assertion_result = AssertionResult {
            entity_id: entity.index(),
            name: name.clone(),
            severity,
            passed: result.passed,
            message: result.message.clone(),
        };
        results.push(assertion_result);

        // Update the assertion entity with the latest result.
        if let Some(a) = world.get_mut::<Assertion>(entity) {
            a.last_result = Some(result);
        }
    }

    EvaluationReport {
        total: results.len(),
        passed,
        failed,
        results,
    }
}

/// Parse an entity filter from a CLI-style string.
///
/// Supported formats:
/// - `"any"` → matches all
/// - `"team:1"` → matches team 1
/// - `"role:hero"` → matches heroes
/// - `"tag:ground"` → matches entities with tag "ground"
/// - `"component:Health"` → matches entities with Health component
/// - Multiple filters joined by `+` → AND logic
pub fn parse_entity_filter(s: &str) -> Option<EntityFilter> {
    // Check for AND logic (+ separator)
    if s.contains('+') {
        let parts: Vec<&str> = s.split('+').collect();
        let filters: Vec<EntityFilter> = parts
            .iter()
            .filter_map(|p| parse_single_filter(p.trim()))
            .collect();
        if filters.len() == parts.len() {
            return Some(EntityFilter::And { filters });
        }
        return None;
    }
    parse_single_filter(s)
}

fn parse_single_filter(s: &str) -> Option<EntityFilter> {
    if s == "any" || s == "all" || s == "*" {
        return Some(EntityFilter::Any);
    }
    if let Some(rest) = s.strip_prefix("team:") {
        let team: u8 = rest.parse().ok()?;
        return Some(EntityFilter::Team { team });
    }
    if let Some(rest) = s.strip_prefix("role:") {
        return Some(EntityFilter::Role {
            role: rest.to_string(),
        });
    }
    if let Some(rest) = s.strip_prefix("tag:") {
        return Some(EntityFilter::Tag {
            tag: rest.to_string(),
        });
    }
    if let Some(rest) = s.strip_prefix("component:") {
        return Some(EntityFilter::HasComponent {
            component: rest.to_string(),
        });
    }
    None
}

/// Parse a condition type from a CLI-style string.
///
/// Supported formats:
/// - `"entity-exists"` → EntityExists
/// - `"entity-count"` → EntityCount
/// - `"all-teams-have-spawns"` → AllTeamsHaveSpawnPoints
/// - `"no-overlap"` → NoOverlap
/// - `"none-dead"` → NoneAreDead
/// - `"no-zero-health"` → NoZeroHealthAlive
/// - `"renderable"` → AllRenderableHaveTransform
/// - `"game-phase"` → GamePhase
/// - `"entity-budget"` → EntityBudget
pub fn parse_condition_type(s: &str) -> Option<&str> {
    match s {
        "entity-exists" | "exists" => Some("entity_exists"),
        "entity-count" | "count" => Some("entity_count"),
        "field-check" | "field" => Some("field_check"),
        "all-teams-have-spawns" | "spawns" => Some("all_teams_have_spawn_points"),
        "no-overlap" | "overlap" => Some("no_overlap"),
        "none-dead" | "alive" => Some("none_are_dead"),
        "no-zero-health" => Some("no_zero_health_alive"),
        "renderable" | "all-renderable" => Some("all_renderable_have_transform"),
        "game-phase" | "phase" => Some("game_phase"),
        "entity-budget" | "budget" => Some("entity_budget"),
        _ => None,
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    fn setup_world() -> World {
        let mut world = World::new();
        world.insert_resource(crate::game_state::GameState::new(
            crate::game_state::MatchConfig {
                mode: "deathmatch".into(),
                score_limit: 10,
                time_limit: 300.0,
                respawn_delay: 3.0,
            },
        ));
        world
    }

    #[test]
    fn entity_exists_passes_when_present() {
        let mut world = setup_world();
        let e = world.spawn(Health {
            current: 100.0,
            max: 100.0,
        });
        world.insert(e, Team(1));
        world.insert(e, EntityRole::Hero);
        world.spawn(Assertion {
            name: "hero-exists".into(),
            condition: AssertCondition::EntityExists {
                filter: EntityFilter::Role {
                    role: "hero".into(),
                },
            },
            severity: Severity::Error,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert_eq!(report.total, 1);
        assert_eq!(report.passed, 1);
        assert!(report.results[0].passed);
    }

    #[test]
    fn entity_exists_fails_when_absent() {
        let mut world = setup_world();
        world.spawn(Assertion {
            name: "hero-exists".into(),
            condition: AssertCondition::EntityExists {
                filter: EntityFilter::Role {
                    role: "hero".into(),
                },
            },
            severity: Severity::Error,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert_eq!(report.failed, 1);
        assert!(!report.results[0].passed);
    }

    #[test]
    fn entity_count_within_bounds() {
        let mut world = setup_world();
        let e1 = world.spawn(Health {
            current: 100.0,
            max: 100.0,
        });
        world.insert(e1, Team(1));
        let e2 = world.spawn(Health {
            current: 100.0,
            max: 100.0,
        });
        world.insert(e2, Team(1));
        world.spawn(Assertion {
            name: "team1-count".into(),
            condition: AssertCondition::EntityCount {
                filter: EntityFilter::Team { team: 1 },
                min: Some(1),
                max: Some(5),
            },
            severity: Severity::Warning,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert!(report.results[0].passed);
    }

    #[test]
    fn entity_count_below_minimum() {
        let mut world = setup_world();
        world.spawn(Assertion {
            name: "team1-count".into(),
            condition: AssertCondition::EntityCount {
                filter: EntityFilter::Team { team: 1 },
                min: Some(1),
                max: None,
            },
            severity: Severity::Error,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert!(!report.results[0].passed);
    }

    #[test]
    fn field_check_health_above_zero() {
        let mut world = setup_world();
        let e1 = world.spawn(Health {
            current: 50.0,
            max: 100.0,
        });
        world.insert(e1, Team(1));
        let e2 = world.spawn(Health {
            current: 80.0,
            max: 100.0,
        });
        world.insert(e2, Team(1));
        world.spawn(Assertion {
            name: "team1-healthy".into(),
            condition: AssertCondition::FieldCheck {
                filter: EntityFilter::Team { team: 1 },
                field: "health".into(),
                op: CompareOp::Greater,
                value: 0.0,
            },
            severity: Severity::Error,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert!(report.results[0].passed);
    }

    #[test]
    fn all_teams_have_spawn_points() {
        let mut world = setup_world();
        world.spawn(Team(1));
        world.spawn(Team(2));
        world.spawn(SpawnPoint { team: 1 });
        // Team 2 has no spawn point
        world.spawn(Assertion {
            name: "all-spawns".into(),
            condition: AssertCondition::AllTeamsHaveSpawnPoints,
            severity: Severity::Error,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert!(!report.results[0].passed);
        assert!(report.results[0].message.contains("2"));
    }

    #[test]
    fn no_overlap_detects_close_entities() {
        let mut world = setup_world();
        let e1 = world.spawn(Team(1));
        world.insert(e1, LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(e1, GlobalTransform(Transform::from_translation(Vec3::ZERO)));

        let e2 = world.spawn(Team(1));
        world.insert(
            e2,
            LocalTransform(Transform::from_translation(Vec3::new(0.1, 0.0, 0.0))),
        );
        world.insert(
            e2,
            GlobalTransform(Transform::from_translation(Vec3::new(0.1, 0.0, 0.0))),
        );

        world.spawn(Assertion {
            name: "no-overlap".into(),
            condition: AssertCondition::NoOverlap {
                filter: EntityFilter::Team { team: 1 },
                min_distance: 1.0,
            },
            severity: Severity::Warning,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert!(!report.results[0].passed);
    }

    #[test]
    fn none_dead_passes_when_all_alive() {
        let mut world = setup_world();
        let e = world.spawn(Health {
            current: 100.0,
            max: 100.0,
        });
        world.insert(e, Team(1));
        world.spawn(Assertion {
            name: "team1-alive".into(),
            condition: AssertCondition::NoneAreDead {
                filter: EntityFilter::Team { team: 1 },
            },
            severity: Severity::Error,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert!(report.results[0].passed);
    }

    #[test]
    fn none_dead_fails_when_dead_entity() {
        let mut world = setup_world();
        let e = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.insert(e, Team(1));
        world.insert(e, Dead);
        world.spawn(Assertion {
            name: "team1-alive".into(),
            condition: AssertCondition::NoneAreDead {
                filter: EntityFilter::Team { team: 1 },
            },
            severity: Severity::Error,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert!(!report.results[0].passed);
    }

    #[test]
    fn no_zero_health_alive_detects_violation() {
        let mut world = setup_world();
        // Alive entity with 0 health = violation
        world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.spawn(Assertion {
            name: "no-zero-hp".into(),
            condition: AssertCondition::NoZeroHealthAlive,
            severity: Severity::Error,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert!(!report.results[0].passed);
    }

    #[test]
    fn entity_budget_within_limit() {
        let mut world = setup_world();
        world.spawn(Team(1));
        world.spawn(Team(2));
        world.spawn(Assertion {
            name: "budget".into(),
            condition: AssertCondition::EntityBudget { max: 100 },
            severity: Severity::Warning,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert!(report.results[0].passed);
    }

    #[test]
    fn parse_entity_filter_works() {
        assert!(matches!(
            parse_entity_filter("any"),
            Some(EntityFilter::Any)
        ));
        assert!(matches!(
            parse_entity_filter("team:1"),
            Some(EntityFilter::Team { team: 1 })
        ));
        assert!(matches!(
            parse_entity_filter("role:hero"),
            Some(EntityFilter::Role { .. })
        ));
        assert!(matches!(
            parse_entity_filter("tag:ground"),
            Some(EntityFilter::Tag { .. })
        ));
        assert!(matches!(
            parse_entity_filter("component:Health"),
            Some(EntityFilter::HasComponent { .. })
        ));
        assert!(parse_entity_filter("invalid").is_none());
    }

    #[test]
    fn parse_and_filter() {
        let filter = parse_entity_filter("team:1+role:hero").unwrap();
        match filter {
            EntityFilter::And { filters } => assert_eq!(filters.len(), 2),
            _ => panic!("Expected And filter"),
        }
    }

    #[test]
    fn and_filter_matches_correctly() {
        let mut world = setup_world();
        let e = world.spawn(Team(1));
        world.insert(
            e,
            Health {
                current: 100.0,
                max: 100.0,
            },
        );
        world.insert(e, EntityRole::Hero);

        let filter = EntityFilter::And {
            filters: vec![
                EntityFilter::Team { team: 1 },
                EntityFilter::Role {
                    role: "hero".into(),
                },
            ],
        };
        assert!(matches_filter(e, &filter, &world));

        let filter_wrong = EntityFilter::And {
            filters: vec![
                EntityFilter::Team { team: 2 }, // wrong team
                EntityFilter::Role {
                    role: "hero".into(),
                },
            ],
        };
        assert!(!matches_filter(e, &filter_wrong, &world));
    }

    #[test]
    fn multiple_assertions_evaluated() {
        let mut world = setup_world();
        let e = world.spawn(Health {
            current: 100.0,
            max: 100.0,
        });
        world.insert(e, Team(1));
        world.spawn(Assertion {
            name: "assert-1".into(),
            condition: AssertCondition::EntityExists {
                filter: EntityFilter::Team { team: 1 },
            },
            severity: Severity::Error,
            last_result: None,
        });
        world.spawn(Assertion {
            name: "assert-2".into(),
            condition: AssertCondition::EntityExists {
                filter: EntityFilter::Team { team: 99 }, // no such team
            },
            severity: Severity::Warning,
            last_result: None,
        });

        let report = evaluate_assertions(&mut world);
        assert_eq!(report.total, 2);
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
    }

    #[test]
    fn evaluate_updates_last_result() {
        let mut world = setup_world();
        let e = world.spawn(Health {
            current: 100.0,
            max: 100.0,
        });
        world.insert(e, Team(1));
        let assert_entity = world.spawn(Assertion {
            name: "check".into(),
            condition: AssertCondition::EntityExists {
                filter: EntityFilter::Team { team: 1 },
            },
            severity: Severity::Error,
            last_result: None,
        });

        evaluate_assertions(&mut world);

        let a = world.get::<Assertion>(assert_entity).unwrap();
        assert!(a.last_result.is_some());
        assert!(a.last_result.as_ref().unwrap().passed);
    }
}
