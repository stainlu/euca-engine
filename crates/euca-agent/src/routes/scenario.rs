//! Scenario primitive — declarative game setup as a single JSON document.
//!
//! A scenario describes the **complete** state needed to bring a fresh
//! world to life: templates, entities, rules, assertions, camera, game
//! mode. It replaces the imperative 28-command MOBA bash script with a
//! file an agent can save, edit, and replay against either the main
//! world or a fork.
//!
//! # Why this exists
//!
//! Before scenarios, setting up a MOBA required ordering 28 individual
//! `euca template/spawn/rule/assert` commands. Any one of them could
//! fail mid-setup, leaving the world half-built; agents had to figure
//! out which step failed and resume from a partial state. Scenarios
//! make setup **atomic and declarative**: the engine receives the
//! entire description and applies it as one operation.
//!
//! # Composition with fork
//!
//! Scenarios pair naturally with [`fork`](super::fork): an agent can
//! save a baseline scenario, fork the world, apply a modified scenario
//! to the fork (e.g. with hero HP doubled), step the fork forward, and
//! compare. The main world stays untouched.
//!
//! # Format versioning
//!
//! Scenarios are version-tagged. **v2** (this module) carries
//! `templates`, typed `actions`, and `assertions` in addition to the
//! existing `entities`, `rules`, `camera`, and `game` sections of v1.
//! The level loader auto-detects v1 (loose JSON, string actions) vs v2
//! (struct-based, typed actions) and dispatches to the right code path.

use std::collections::HashMap;

use axum::Json;
use axum::extract::{Path, State};
use euca_gameplay::{AssertCondition, GameAction, RuleCondition, Severity};
use serde::{Deserialize, Serialize};

use crate::state::SharedWorld;

use super::SpawnRequest;

// ── Scenario format types ───────────────────────────────────────────────────

/// Top-level scenario document. Round-trips through `POST /scenario` and
/// `GET /scenario` and can be applied to a fork via
/// `POST /fork/{id}/scenario`.
///
/// All fields are optional; an empty scenario is valid and produces an
/// empty world.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScenarioSpec {
    /// Format version. v2 indicates this scenario format with typed
    /// actions, templates, and assertions. Defaults to 2.
    #[serde(default = "default_version")]
    pub version: u32,

    /// Human-readable scenario name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Camera configuration applied to the renderer when loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<ScenarioCamera>,

    /// Game-mode configuration (mode, score limit, time limit, etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub game: Option<ScenarioGame>,

    /// Reusable entity templates. Inserted into `TemplateRegistry` before
    /// `entities` are spawned so entities can reference them by name.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub templates: HashMap<String, SpawnRequest>,

    /// Entities to spawn. Each entry is either a fully-inlined
    /// `SpawnRequest` or a template reference with optional overrides.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<ScenarioEntity>,

    /// Game rules to register (typed conditions + typed actions).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rules: Vec<ScenarioRule>,

    /// Assertions to register on the world.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assertions: Vec<AssertionSpec>,
}

fn default_version() -> u32 {
    2
}

/// Camera state embedded in a scenario.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScenarioCamera {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eye: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fov_y: Option<f32>,
}

/// Game-mode configuration embedded in a scenario.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScenarioGame {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_limit: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_limit: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub respawn_delay: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_start: Option<bool>,
}

/// One entity in the scenario's `entities` list. Either references a
/// template by name (with optional overrides) or inlines a complete
/// `SpawnRequest`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScenarioEntity {
    /// Reference an entry from the scenario's `templates` map.
    Template {
        /// Template name from the `templates` section.
        template: String,
        /// Optional position override (templates often share a base
        /// definition but differ in placement).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        position: Option<[f32; 3]>,
        /// Optional field-level overrides applied on top of the
        /// template before spawning.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        overrides: Option<SpawnRequest>,
    },
    /// Inline a fully-specified entity. Use this for one-offs that
    /// don't justify a template.
    Inline(SpawnRequest),
}

/// One rule in the scenario's `rules` list. Uses the typed
/// `RuleCondition` and `GameAction` enums so authors don't have to
/// build string DSLs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScenarioRule {
    /// What triggers the rule.
    pub when: RuleCondition,
    /// Which entities this rule watches. Defaults to "any".
    #[serde(default = "default_filter_string")]
    pub filter: String,
    /// Actions to execute when the rule fires.
    pub actions: Vec<GameAction>,
}

fn default_filter_string() -> String {
    "any".to_string()
}

/// Serializable assertion definition. Mirrors the runtime `Assertion`
/// component but excludes `last_result` (which is runtime-only state).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssertionSpec {
    /// Human-readable name.
    pub name: String,
    /// What to check.
    pub condition: AssertCondition,
    /// How critical a failure is. Defaults to "error".
    #[serde(default = "default_severity")]
    pub severity: Severity,
}

fn default_severity() -> Severity {
    Severity::Error
}

// ── Apply / extract ─────────────────────────────────────────────────────────

/// Apply a `ScenarioSpec` to an existing world. **Destroys all current
/// entities and resources** before loading — equivalent to "reset and
/// build this scenario from scratch."
///
/// Returns the number of entities spawned.
pub fn apply_scenario(w: &mut euca_ecs::World, scenario: &ScenarioSpec) -> u32 {
    // 1. Wipe existing entities. We despawn rather than recreate the
    //    world so that resources (renderer, audio handles, etc.) stay
    //    intact — the same convention as `scene_load`.
    let existing: Vec<euca_ecs::Entity> = w.all_entities();
    for e in existing {
        w.despawn(e);
    }

    // 2. Install templates into the registry before spawning entities,
    //    so template references resolve.
    if !scenario.templates.is_empty() {
        let mut registry = w
            .resource_mut::<super::TemplateRegistry>()
            .map(|r| std::mem::take(r))
            .unwrap_or_default();
        for (name, req) in &scenario.templates {
            registry.templates.insert(name.clone(), req.clone());
        }
        w.insert_resource(registry);
    }

    // 3. Spawn entities.
    let mut count = 0u32;
    for entry in &scenario.entities {
        let resolved: SpawnRequest = match entry {
            ScenarioEntity::Inline(req) => req.clone(),
            ScenarioEntity::Template {
                template,
                position,
                overrides,
            } => {
                let base = w
                    .resource::<super::TemplateRegistry>()
                    .and_then(|r| r.templates.get(template).cloned());
                let mut merged = match base {
                    Some(b) => b,
                    None => {
                        log::warn!(
                            "scenario references unknown template '{template}'; skipping entity"
                        );
                        continue;
                    }
                };
                if let Some(pos) = position {
                    merged.position = Some(*pos);
                }
                if let Some(over) = overrides {
                    merge_spawn_overrides(&mut merged, over);
                }
                merged
            }
        };
        super::level::spawn_entity(w, &resolved);
        count += 1;
    }

    // 4. Register rules.
    for rule in &scenario.rules {
        apply_rule(w, rule);
    }

    // 5. Register assertions.
    for assertion in &scenario.assertions {
        w.spawn(euca_gameplay::Assertion {
            name: assertion.name.clone(),
            condition: assertion.condition.clone(),
            severity: assertion.severity,
            last_result: None,
        });
    }

    // 6. Apply camera config.
    if let Some(cam_cfg) = &scenario.camera
        && let Some(cam) = w.resource_mut::<euca_render::Camera>()
    {
        if let Some(eye) = cam_cfg.eye {
            cam.eye = euca_math::Vec3::new(eye[0], eye[1], eye[2]);
        }
        if let Some(target) = cam_cfg.target {
            cam.target = euca_math::Vec3::new(target[0], target[1], target[2]);
        }
        if let Some(fov) = cam_cfg.fov_y {
            cam.fov_y = fov;
        }
    }

    // 7. Apply game-mode config.
    if let Some(game) = &scenario.game {
        let config = euca_gameplay::MatchConfig {
            mode: game.mode.clone().unwrap_or_else(|| "deathmatch".into()),
            score_limit: game.score_limit.unwrap_or(10),
            time_limit: game.time_limit.unwrap_or(300.0),
            respawn_delay: game.respawn_delay.unwrap_or(3.0),
        };
        let mut state = euca_gameplay::GameState::new(config);
        if game.auto_start.unwrap_or(true) {
            state.start();
        }
        w.insert_resource(state);
    }

    count
}

/// Merge override fields from `over` into `base`, overwriting only the
/// fields that the override actually specifies.
fn merge_spawn_overrides(base: &mut SpawnRequest, over: &SpawnRequest) {
    if over.agent_id.is_some() {
        base.agent_id = over.agent_id;
    }
    if over.mesh.is_some() {
        base.mesh = over.mesh.clone();
    }
    if over.color.is_some() {
        base.color = over.color.clone();
    }
    if over.position.is_some() {
        base.position = over.position;
    }
    if over.scale.is_some() {
        base.scale = over.scale;
    }
    if over.velocity.is_some() {
        base.velocity = over.velocity.clone();
    }
    if over.collider.is_some() {
        base.collider = over.collider.clone();
    }
    if over.physics_body.is_some() {
        base.physics_body = over.physics_body.clone();
    }
    if over.health.is_some() {
        base.health = over.health;
    }
    if over.team.is_some() {
        base.team = over.team;
    }
    if over.combat.is_some() {
        base.combat = over.combat;
    }
    if over.combat_damage.is_some() {
        base.combat_damage = over.combat_damage;
    }
    if over.combat_range.is_some() {
        base.combat_range = over.combat_range;
    }
    if over.combat_speed.is_some() {
        base.combat_speed = over.combat_speed;
    }
    if over.combat_cooldown.is_some() {
        base.combat_cooldown = over.combat_cooldown;
    }
    if over.combat_style.is_some() {
        base.combat_style = over.combat_style.clone();
    }
    if over.ai_patrol.is_some() {
        base.ai_patrol = over.ai_patrol.clone();
    }
    if over.gold.is_some() {
        base.gold = over.gold;
    }
    if over.gold_bounty.is_some() {
        base.gold_bounty = over.gold_bounty;
    }
    if over.xp_bounty.is_some() {
        base.xp_bounty = over.xp_bounty;
    }
    if over.role.is_some() {
        base.role = over.role.clone();
    }
    if over.spawn_point.is_some() {
        base.spawn_point = over.spawn_point;
    }
    if over.player.is_some() {
        base.player = over.player;
    }
    if over.building_type.is_some() {
        base.building_type = over.building_type.clone();
    }
    if over.lane.is_some() {
        base.lane = over.lane.clone();
    }
}

fn apply_rule(w: &mut euca_ecs::World, rule: &ScenarioRule) {
    let filter = euca_gameplay::parse_filter(&rule.filter).unwrap_or(euca_gameplay::RuleFilter::Any);
    let actions = std::sync::Arc::new(rule.actions.clone());
    match rule.when.clone() {
        RuleCondition::Death => {
            w.spawn(euca_gameplay::OnDeathRule { filter, actions });
        }
        RuleCondition::Timer { interval } => {
            w.spawn(euca_gameplay::TimerRule {
                interval,
                elapsed: 0.0,
                repeat: true,
                actions,
            });
        }
        RuleCondition::HealthBelow { threshold } => {
            w.spawn(euca_gameplay::HealthBelowRule {
                filter,
                threshold,
                triggered_entities: std::collections::HashSet::new(),
                actions,
            });
        }
        RuleCondition::Score { threshold } => {
            w.spawn(euca_gameplay::OnScoreRule {
                score_threshold: threshold,
                triggered: false,
                actions,
            });
        }
        RuleCondition::Phase { phase } => {
            w.spawn(euca_gameplay::OnPhaseRule {
                phase,
                triggered: false,
                actions,
            });
        }
    }
}

/// Extract the current world state into a `ScenarioSpec` for export.
/// Round-trippable through `apply_scenario`.
pub fn extract_scenario(w: &euca_ecs::World) -> ScenarioSpec {
    use euca_ecs::{Entity, Query};

    // Templates from the registry.
    let templates = w
        .resource::<super::TemplateRegistry>()
        .map(|r| r.templates.clone())
        .unwrap_or_default();

    // Entities — extract via the existing read_entity_data helper, then
    // convert to scenario inline form using SpawnRequest fields we can
    // reconstruct.
    let entities: Vec<ScenarioEntity> = {
        let query = Query::<Entity>::new(w);
        query
            .iter()
            .filter_map(|e| {
                // Skip rule and assertion entities (they're emitted in
                // their own sections).
                if w.get::<euca_gameplay::OnDeathRule>(e).is_some()
                    || w.get::<euca_gameplay::TimerRule>(e).is_some()
                    || w.get::<euca_gameplay::HealthBelowRule>(e).is_some()
                    || w.get::<euca_gameplay::OnScoreRule>(e).is_some()
                    || w.get::<euca_gameplay::OnPhaseRule>(e).is_some()
                    || w.get::<euca_gameplay::Assertion>(e).is_some()
                {
                    return None;
                }
                Some(ScenarioEntity::Inline(spawn_request_from_world(w, e)))
            })
            .collect()
    };

    // Rules — query each rule component type and convert to ScenarioRule.
    let mut rules = Vec::new();
    {
        let q = Query::<&euca_gameplay::OnDeathRule>::new(w);
        for r in q.iter() {
            rules.push(ScenarioRule {
                when: RuleCondition::Death,
                filter: filter_to_string(&r.filter),
                actions: (*r.actions).clone(),
            });
        }
    }
    {
        let q = Query::<&euca_gameplay::TimerRule>::new(w);
        for r in q.iter() {
            rules.push(ScenarioRule {
                when: RuleCondition::Timer { interval: r.interval },
                filter: "any".to_string(),
                actions: (*r.actions).clone(),
            });
        }
    }
    {
        let q = Query::<&euca_gameplay::HealthBelowRule>::new(w);
        for r in q.iter() {
            rules.push(ScenarioRule {
                when: RuleCondition::HealthBelow { threshold: r.threshold },
                filter: filter_to_string(&r.filter),
                actions: (*r.actions).clone(),
            });
        }
    }
    {
        let q = Query::<&euca_gameplay::OnScoreRule>::new(w);
        for r in q.iter() {
            rules.push(ScenarioRule {
                when: RuleCondition::Score { threshold: r.score_threshold },
                filter: "any".to_string(),
                actions: (*r.actions).clone(),
            });
        }
    }
    {
        let q = Query::<&euca_gameplay::OnPhaseRule>::new(w);
        for r in q.iter() {
            rules.push(ScenarioRule {
                when: RuleCondition::Phase { phase: r.phase.clone() },
                filter: "any".to_string(),
                actions: (*r.actions).clone(),
            });
        }
    }

    // Assertions.
    let assertions: Vec<AssertionSpec> = {
        let q = Query::<&euca_gameplay::Assertion>::new(w);
        q.iter()
            .map(|a| AssertionSpec {
                name: a.name.clone(),
                condition: a.condition.clone(),
                severity: a.severity,
            })
            .collect()
    };

    // Camera + game state.
    let camera = w.resource::<euca_render::Camera>().map(|cam| ScenarioCamera {
        eye: Some([cam.eye.x, cam.eye.y, cam.eye.z]),
        target: Some([cam.target.x, cam.target.y, cam.target.z]),
        fov_y: Some(cam.fov_y),
    });

    let game = w.resource::<euca_gameplay::GameState>().map(|s| ScenarioGame {
        mode: Some(s.config.mode.clone()),
        score_limit: Some(s.config.score_limit),
        time_limit: Some(s.config.time_limit),
        respawn_delay: Some(s.config.respawn_delay),
        auto_start: Some(true),
    });

    ScenarioSpec {
        version: 2,
        name: None,
        camera,
        game,
        templates,
        entities,
        rules,
        assertions,
    }
}

/// Reconstruct a `SpawnRequest` from an entity in the world. Reads the
/// component fields we know how to round-trip. Returns a SpawnRequest
/// with `None` for fields whose components are absent.
fn spawn_request_from_world(w: &euca_ecs::World, e: euca_ecs::Entity) -> SpawnRequest {
    use euca_scene::LocalTransform;

    let mut req = SpawnRequest::default();

    if let Some(t) = w.get::<LocalTransform>(e) {
        let tr = t.0.translation;
        req.position = Some([tr.x, tr.y, tr.z]);
        let s = t.0.scale;
        req.scale = Some([s.x, s.y, s.z]);
    }
    if let Some(h) = w.get::<euca_gameplay::Health>(e) {
        req.health = Some(h.max);
    }
    if let Some(team) = w.get::<euca_gameplay::Team>(e) {
        req.team = Some(team.0);
    }

    req
}

fn filter_to_string(filter: &euca_gameplay::RuleFilter) -> String {
    match filter {
        euca_gameplay::RuleFilter::Any => "any".to_string(),
        euca_gameplay::RuleFilter::Entity(id) => format!("entity:{id}"),
        euca_gameplay::RuleFilter::Team(t) => format!("team:{t}"),
    }
}

// ── HTTP handlers ───────────────────────────────────────────────────────────

/// `POST /scenario` — apply a scenario to the main world. Wipes the
/// current main world and loads the scenario fresh.
pub async fn scenario_apply_main(
    State(shared): State<SharedWorld>,
    Json(scenario): Json<ScenarioSpec>,
) -> Json<serde_json::Value> {
    let count = shared.with(|w, _| apply_scenario(w, &scenario));
    Json(serde_json::json!({
        "ok": true,
        "entities_spawned": count,
        "templates": scenario.templates.len(),
        "rules": scenario.rules.len(),
        "assertions": scenario.assertions.len(),
    }))
}

/// `GET /scenario` — export the current main world as a scenario JSON.
pub async fn scenario_export_main(
    State(shared): State<SharedWorld>,
) -> Json<serde_json::Value> {
    let scenario = shared.with(|w, _| extract_scenario(w));
    Json(serde_json::to_value(scenario).unwrap_or(serde_json::json!({})))
}

/// `POST /fork/{id}/scenario` — apply a scenario to a fork. Wipes the
/// fork's current state and loads the scenario fresh; the main world
/// is untouched.
pub async fn scenario_apply_fork(
    State(shared): State<SharedWorld>,
    Path(fork_id): Path<String>,
    Json(scenario): Json<ScenarioSpec>,
) -> Json<serde_json::Value> {
    let result = shared.with_fork(&fork_id, |w, _| apply_scenario(w, &scenario));
    match result {
        Some(count) => Json(serde_json::json!({
            "ok": true,
            "fork_id": fork_id,
            "entities_spawned": count,
            "templates": scenario.templates.len(),
            "rules": scenario.rules.len(),
            "assertions": scenario.assertions.len(),
        })),
        None => Json(serde_json::json!({
            "ok": false,
            "error": format!("fork '{fork_id}' not found"),
        })),
    }
}
