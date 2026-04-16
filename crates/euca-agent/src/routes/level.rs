use axum::Json;
use axum::extract::State;

use euca_ecs::{Entity, Query};
use euca_math::Vec3;
use euca_physics::Velocity;
use euca_render::{MaterialRef, MeshRenderer};
use euca_scene::{GlobalTransform, LocalTransform};

use crate::state::{Owner, SharedWorld};

use super::{
    DefaultAssets, MeshResolution, MessageResponse, RichEntityData, SpawnRequest, SpawnResponse,
    apply_collider, apply_physics_body, apply_velocity, read_entity_data, resolve_mesh,
};

// ---------------------------------------------------------------------------
// New LevelData format (terrain-aware levels with width/height)
// ---------------------------------------------------------------------------

/// Camera configuration embedded in a new-format level.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LevelCamera {
    #[serde(default)]
    pub eye: Option<[f32; 3]>,
    #[serde(default)]
    pub target: Option<[f32; 3]>,
    #[serde(default)]
    pub fov_y: Option<f32>,
}

/// New level data format that carries terrain dimensions and metadata
/// alongside the entity list and camera config.
///
/// Detected by the presence of both `width` and `height` fields at the
/// top level of the JSON document.
#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct LevelData {
    /// Human-readable level name.
    #[serde(default)]
    pub name: Option<String>,
    /// Terrain width in grid cells.
    #[serde(default)]
    pub width: Option<u32>,
    /// Terrain height (depth) in grid cells.
    #[serde(default)]
    pub height: Option<u32>,
    /// Entities to spawn.
    #[serde(default)]
    pub entities: Vec<SpawnRequest>,
    /// Optional camera override.
    #[serde(default)]
    pub camera: Option<LevelCamera>,
}

/// Level file format request.
#[derive(serde::Deserialize)]
pub struct LevelLoadRequest {
    pub path: String,
}

/// Spawn an entity from a `SpawnRequest`, applying all components.
///
/// Single source of truth for entity creation, used by `POST /spawn`
/// (via entity.rs) and the level loader.
pub(crate) fn spawn_entity(w: &mut euca_ecs::World, req: &SpawnRequest) -> SpawnResponse {
    let pos = req.position.unwrap_or([0.0, 0.0, 0.0]);
    let scl = req.scale.unwrap_or([1.0, 1.0, 1.0]);
    let mut transform = euca_math::Transform::from_translation(Vec3::new(pos[0], pos[1], pos[2]));
    transform.scale = Vec3::new(scl[0], scl[1], scl[2]);

    let entity = w.spawn(LocalTransform(transform));
    w.insert(entity, GlobalTransform::default());

    if let Some(agent_id) = req.agent_id {
        w.insert(entity, Owner(agent_id));
    }

    if let Some(mesh_name) = &req.mesh {
        match resolve_mesh(w, entity, mesh_name) {
            MeshResolution::Ready(handle) => {
                w.insert(entity, MeshRenderer { mesh: handle });
                let mat = w.resource::<DefaultAssets>().cloned().map(|assets| {
                    req.color
                        .as_deref()
                        .and_then(|c| assets.material(c))
                        .unwrap_or(assets.default_material)
                });
                if let Some(mat) = mat {
                    w.insert(entity, MaterialRef { handle: mat });
                }
            }
            MeshResolution::Pending => {
                if let Some(assets) = w.resource::<DefaultAssets>().cloned() {
                    let mat = req
                        .color
                        .as_deref()
                        .and_then(|c| assets.material(c))
                        .unwrap_or(assets.default_material);
                    w.insert(entity, MaterialRef { handle: mat });
                }
            }
            MeshResolution::NotFound => {}
            MeshResolution::LoadError(err) => {
                log::warn!("Failed to load mesh '{}': {}", mesh_name, err);
            }
        }
    }

    if let Some(v) = &req.velocity {
        apply_velocity(w, entity, v);
    }
    if let Some(c) = &req.collider {
        apply_collider(w, entity, c);
    }
    if let Some(pb) = &req.physics_body {
        apply_physics_body(w, entity, pb);
        if pb != "Static" && w.get::<Velocity>(entity).is_none() {
            w.insert(entity, Velocity::default());
        }
    }

    if let Some(max_health) = req.health {
        w.insert(entity, euca_gameplay::Health::new(max_health));
    }
    if let Some(team_id) = req.team {
        w.insert(entity, euca_gameplay::Team(team_id));
    }
    if req.combat == Some(true) {
        let mut ac = euca_gameplay::AutoCombat::new();
        if let Some(d) = req.combat_damage {
            ac.damage = d;
        }
        if let Some(r) = req.combat_range {
            ac.range = r;
            ac.detect_range = r.max(ac.detect_range);
        }
        if let Some(s) = req.combat_speed {
            ac.speed = s;
        }
        if let Some(c) = req.combat_cooldown {
            ac.cooldown = c;
        }
        if let Some(ref style) = req.combat_style
            && style == "stationary"
        {
            ac.attack_style = euca_gameplay::AttackStyle::Stationary;
            ac.speed = 0.0;
        }
        w.insert(entity, ac);
        if w.get::<Velocity>(entity).is_none() {
            w.insert(entity, Velocity::default());
        }
    }
    if let Some(ref waypoints) = req.ai_patrol {
        let wps: Vec<euca_math::Vec3> = waypoints
            .iter()
            .map(|wp| euca_math::Vec3::new(wp[0], wp[1], wp[2]))
            .collect();
        let speed = req.combat_speed.unwrap_or(3.0);
        w.insert(entity, euca_gameplay::AiGoal::patrol(wps, speed));
    }
    if req.combat.unwrap_or(false)
        && let Some(team_id) = req.team
    {
        let dir = if team_id == 1 {
            euca_math::Vec3::new(1.0, 0.0, 0.0)
        } else {
            euca_math::Vec3::new(-1.0, 0.0, 0.0)
        };
        w.insert(entity, euca_gameplay::MarchDirection(dir));
    }

    if let Some(g) = req.gold {
        w.insert(entity, euca_gameplay::Gold(g));
        if w.get::<euca_gameplay::Level>(entity).is_none() {
            w.insert(entity, euca_gameplay::Level::new(1));
        }
    }
    if let Some(b) = req.gold_bounty {
        w.insert(entity, euca_gameplay::GoldBounty(b));
    }
    if let Some(xp) = req.xp_bounty {
        w.insert(entity, euca_gameplay::XpBounty(xp));
    }
    if let Some(ref role) = req.role {
        let r = match role.as_str() {
            "hero" => euca_gameplay::EntityRole::Hero,
            "minion" => euca_gameplay::EntityRole::Minion,
            "tower" => euca_gameplay::EntityRole::Tower,
            "structure" => euca_gameplay::EntityRole::Structure,
            _ => euca_gameplay::EntityRole::Minion,
        };
        w.insert(entity, r);
    }
    if let Some(sp_team) = req.spawn_point {
        w.insert(entity, euca_gameplay::SpawnPoint { team: sp_team });
    }
    if req.player.unwrap_or(false) {
        w.insert(entity, euca_gameplay::player::PlayerHero);
        w.insert(entity, euca_gameplay::player::PlayerCommandQueue::default());
        if let Some(cam) = w.resource_mut::<euca_gameplay::camera::MobaCamera>() {
            cam.follow_entity = Some(entity);
        }
    }

    // Building components: BuildingStats, BackdoorProtection, TowerAggro.
    if let Some(ref bt_str) = req.building_type {
        let building_type = match bt_str.as_str() {
            "tier1_tower" => Some(euca_moba::BuildingType::Tier1Tower),
            "tier2_tower" => Some(euca_moba::BuildingType::Tier2Tower),
            "tier3_tower" => Some(euca_moba::BuildingType::Tier3Tower),
            "tier4_tower" => Some(euca_moba::BuildingType::Tier4Tower),
            "melee_barracks" => Some(euca_moba::BuildingType::MeleeBarracks),
            "ranged_barracks" => Some(euca_moba::BuildingType::RangedBarracks),
            "ancient" => Some(euca_moba::BuildingType::Ancient),
            "fountain" => Some(euca_moba::BuildingType::Fountain),
            "effigy" => Some(euca_moba::BuildingType::Effigy),
            "outpost" => Some(euca_moba::BuildingType::Outpost),
            _ => {
                log::warn!("Unknown building_type '{bt_str}', ignoring");
                None
            }
        };

        let lane = req.lane.as_deref().and_then(|l| match l {
            "top" => Some(euca_moba::Lane::Top),
            "mid" => Some(euca_moba::Lane::Mid),
            "bot" => Some(euca_moba::Lane::Bot),
            _ => {
                log::warn!("Unknown lane '{l}', ignoring");
                None
            }
        });

        if let Some(bt) = building_type {
            let team_id = req.team.unwrap_or(0) as u32;
            let bs = euca_moba::building_stats(bt, team_id, lane);

            // Override the Health component with canonical building HP.
            w.insert(entity, euca_gameplay::Health::new(bs.max_hp));

            // Override AutoCombat with building-canonical values if tower.
            if let (Some(dmg), Some(rng), Some(spd)) =
                (bs.attack_damage, bs.attack_range, bs.attack_speed)
            {
                w.insert(entity, euca_gameplay::AutoCombat::stationary(dmg, rng, spd));
            }

            w.insert(entity, bs);

            // All towers and barracks get backdoor protection.
            match bt {
                euca_moba::BuildingType::Tier1Tower
                | euca_moba::BuildingType::Tier2Tower
                | euca_moba::BuildingType::Tier3Tower
                | euca_moba::BuildingType::Tier4Tower
                | euca_moba::BuildingType::MeleeBarracks
                | euca_moba::BuildingType::RangedBarracks
                | euca_moba::BuildingType::Ancient => {
                    w.insert(entity, euca_moba::BackdoorProtection::default());
                }
                _ => {}
            }

            // Towers get TowerAggro.
            match bt {
                euca_moba::BuildingType::Tier1Tower
                | euca_moba::BuildingType::Tier2Tower
                | euca_moba::BuildingType::Tier3Tower
                | euca_moba::BuildingType::Tier4Tower => {
                    w.insert(entity, euca_moba::TowerAggro::default());
                }
                _ => {}
            }
        }
    }

    SpawnResponse {
        entity_id: entity.index(),
        entity_generation: entity.generation(),
    }
}

/// Load a level definition into the world, spawning entities, rules, and
/// configuring camera and game state.
///
/// `level` should be a JSON value matching the level file format:
/// ```json
/// { "entities": [...], "rules": [...], "camera": {...}, "game": {...} }
/// ```
///
/// Returns the number of entities created.
pub fn load_level_into_world(w: &mut euca_ecs::World, level: &serde_json::Value) -> u32 {
    let entity_defs: Vec<SpawnRequest> = level
        .get("entities")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let mut count = 0u32;
    for entity_def in &entity_defs {
        spawn_entity(w, entity_def);
        count += 1;
    }

    if let Some(rules) = level.get("rules")
        && let Some(rules_arr) = rules.as_array()
    {
        for rule_val in rules_arr {
            create_rule_from_value(w, rule_val);
        }
    }

    if let Some(cam) = level.get("camera")
        && let Some(cam_res) = w.resource_mut::<euca_render::Camera>()
    {
        if let Some(eye) = cam.get("eye").and_then(|v| v.as_array())
            && eye.len() == 3
        {
            cam_res.eye = Vec3::new(
                eye[0].as_f64().unwrap_or(0.0) as f32,
                eye[1].as_f64().unwrap_or(0.0) as f32,
                eye[2].as_f64().unwrap_or(0.0) as f32,
            );
        }
        if let Some(target) = cam.get("target").and_then(|v| v.as_array())
            && target.len() == 3
        {
            cam_res.target = Vec3::new(
                target[0].as_f64().unwrap_or(0.0) as f32,
                target[1].as_f64().unwrap_or(0.0) as f32,
                target[2].as_f64().unwrap_or(0.0) as f32,
            );
        }
        if let Some(fov) = cam.get("fov_y").and_then(|v| v.as_f64()) {
            cam_res.fov_y = fov as f32;
        }
    }

    if let Some(game) = level.get("game") {
        let mode = game
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("deathmatch")
            .to_string();
        let score_limit = game
            .get("score_limit")
            .and_then(|v| v.as_i64())
            .unwrap_or(10) as i32;
        let time_limit = game
            .get("time_limit")
            .and_then(|v| v.as_f64())
            .unwrap_or(300.0) as f32;
        let respawn_delay = game
            .get("respawn_delay")
            .and_then(|v| v.as_f64())
            .unwrap_or(3.0) as f32;

        let config = euca_gameplay::MatchConfig {
            mode,
            score_limit,
            time_limit,
            respawn_delay,
        };
        let mut state = euca_gameplay::GameState::new(config);
        if game
            .get("auto_start")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
        {
            state.start();
        }
        w.insert_resource(state);
    }

    count
}

/// Auto-detect old vs new level format and load accordingly.
///
/// **New format** is identified by the presence of both `width` and `height`
/// fields at the JSON root.  Everything else falls through to the existing
/// [`load_level_into_world`] path (backward compatible).
///
/// Returns the number of entities created.
pub fn load_level_auto(w: &mut euca_ecs::World, level: &serde_json::Value) -> u32 {
    let is_new_format = level.get("width").is_some() && level.get("height").is_some();

    if !is_new_format {
        return load_level_into_world(w, level);
    }

    // -- New format path --------------------------------------------------

    let data: LevelData = match serde_json::from_value(level.clone()) {
        Ok(d) => d,
        Err(e) => {
            log::error!("Failed to deserialize new LevelData format: {e}");
            return 0;
        }
    };

    let level_name = data.name.as_deref().unwrap_or("<unnamed>");
    let width = data.width.unwrap_or(0);
    let height = data.height.unwrap_or(0);
    log::info!("Loading level '{}' ({}x{})", level_name, width, height,);

    // Store the LevelData as a world resource so other systems can query it.
    w.insert_resource(data.clone());

    // Spawn entities.
    let mut count = 0u32;
    for entity_def in &data.entities {
        spawn_entity(w, entity_def);
        count += 1;
    }

    // Apply camera config.
    if let Some(cam_cfg) = &data.camera
        && let Some(cam) = w.resource_mut::<euca_render::Camera>()
    {
        if let Some(eye) = cam_cfg.eye {
            cam.eye = Vec3::new(eye[0], eye[1], eye[2]);
        }
        if let Some(target) = cam_cfg.target {
            cam.target = Vec3::new(target[0], target[1], target[2]);
        }
        if let Some(fov_y) = cam_cfg.fov_y {
            cam.fov_y = fov_y;
        }
    }

    count
}

/// POST /level/load — load a level definition from a JSON file.
pub async fn level_load(
    State(world): State<SharedWorld>,
    Json(req): Json<LevelLoadRequest>,
) -> Json<serde_json::Value> {
    let data = match std::fs::read_to_string(&req.path) {
        Ok(s) => s,
        Err(e) => {
            return Json(serde_json::json!({
                "ok": false,
                "error": format!("Cannot read {}: {e}", req.path),
            }));
        }
    };

    let level: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => {
            return Json(serde_json::json!({
                "ok": false,
                "error": format!("Invalid JSON: {e}"),
            }));
        }
    };

    let entities_created = world.with(|w, _| load_level_into_world(w, &level));

    Json(serde_json::json!({
        "ok": true,
        "entities_created": entities_created,
    }))
}

fn create_rule_from_value(w: &mut euca_ecs::World, rule: &serde_json::Value) {
    let when_str = rule.get("when").and_then(|v| v.as_str()).unwrap_or("");
    let filter_str = rule.get("filter").and_then(|v| v.as_str()).unwrap_or("any");
    let action_strs: Vec<String> = rule
        .get("actions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let condition = match euca_gameplay::parse_when(when_str) {
        Some(c) => c,
        None => return,
    };

    let filter = euca_gameplay::parse_filter(filter_str).unwrap_or(euca_gameplay::RuleFilter::Any);

    let actions: Vec<euca_gameplay::GameAction> = action_strs
        .iter()
        .filter_map(|s| euca_gameplay::parse_action(s))
        .collect();

    if actions.is_empty() {
        return;
    }

    let actions = std::sync::Arc::new(actions);

    match condition {
        euca_gameplay::RuleCondition::Death => {
            w.spawn(euca_gameplay::OnDeathRule { filter, actions });
        }
        euca_gameplay::RuleCondition::Timer { interval } => {
            w.spawn(euca_gameplay::TimerRule {
                interval,
                elapsed: 0.0,
                repeat: true,
                actions,
            });
        }
        euca_gameplay::RuleCondition::HealthBelow { threshold } => {
            w.spawn(euca_gameplay::HealthBelowRule {
                filter,
                threshold,
                triggered_entities: std::collections::HashSet::new(),
                actions,
            });
        }
        euca_gameplay::RuleCondition::Score { threshold } => {
            w.spawn(euca_gameplay::OnScoreRule {
                score_threshold: threshold,
                triggered: false,
                actions,
            });
        }
        euca_gameplay::RuleCondition::Phase { phase } => {
            w.spawn(euca_gameplay::OnPhaseRule {
                phase,
                triggered: false,
                actions,
            });
        }
    }
}

/// POST /level/save — save current world state to a level JSON file.
pub async fn level_save(
    State(world): State<SharedWorld>,
    Json(req): Json<LevelLoadRequest>,
) -> Json<MessageResponse> {
    let level_data = world.with_world(|w| {
        let entities: Vec<RichEntityData> = {
            let query = Query::<Entity>::new(w);
            query.iter().map(|e| read_entity_data(w, e)).collect()
        };

        let camera = w.resource::<euca_render::Camera>().map(|cam| {
            serde_json::json!({
                "eye": [cam.eye.x, cam.eye.y, cam.eye.z],
                "target": [cam.target.x, cam.target.y, cam.target.z],
                "fov_y": cam.fov_y,
            })
        });

        let game = w.resource::<euca_gameplay::GameState>().map(|state| {
            serde_json::json!({
                "mode": state.config.mode,
                "score_limit": state.config.score_limit,
                "time_limit": state.config.time_limit,
                "respawn_delay": state.config.respawn_delay,
            })
        });

        let mut rules = Vec::new();

        let death_rules = Query::<(Entity, &euca_gameplay::OnDeathRule)>::new(w);
        for (_e, r) in death_rules.iter() {
            rules.push(serde_json::json!({
                "when": "death",
                "filter": format_filter(&r.filter),
                "actions": format_actions(&r.actions),
            }));
        }

        let timer_rules = Query::<(Entity, &euca_gameplay::TimerRule)>::new(w);
        for (_e, t) in timer_rules.iter() {
            rules.push(serde_json::json!({
                "when": format!("timer:{}", t.interval),
                "actions": format_actions(&t.actions),
            }));
        }

        let health_rules = Query::<(Entity, &euca_gameplay::HealthBelowRule)>::new(w);
        for (_e, h) in health_rules.iter() {
            rules.push(serde_json::json!({
                "when": format!("health-below:{}", h.threshold),
                "filter": format_filter(&h.filter),
                "actions": format_actions(&h.actions),
            }));
        }

        let mut level = serde_json::json!({
            "version": 1,
            "entities": entities,
        });

        if !rules.is_empty() {
            level["rules"] = serde_json::json!(rules);
        }
        if let Some(cam) = camera {
            level["camera"] = cam;
        }
        if let Some(g) = game {
            level["game"] = g;
        }

        level
    });

    match std::fs::write(
        &req.path,
        serde_json::to_string_pretty(&level_data).expect("level data serialization failed"),
    ) {
        Ok(()) => Json(MessageResponse {
            ok: true,
            message: Some(format!("Level saved to {}", req.path)),
        }),
        Err(e) => Json(MessageResponse {
            ok: false,
            message: Some(format!("Save failed: {e}")),
        }),
    }
}

fn format_filter(filter: &euca_gameplay::RuleFilter) -> String {
    match filter {
        euca_gameplay::RuleFilter::Any => "any".to_string(),
        euca_gameplay::RuleFilter::Entity(id) => format!("entity:{id}"),
        euca_gameplay::RuleFilter::Team(id) => format!("team:{id}"),
    }
}

fn format_actions(actions: &[euca_gameplay::GameAction]) -> Vec<String> {
    actions.iter().map(|a| format!("{a:?}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_ecs::World;

    /// Old-format JSON (no `width`/`height`) must fall through to
    /// `load_level_into_world` and still work.
    #[test]
    fn auto_detect_old_format() {
        let mut world = World::new();
        let level = serde_json::json!({
            "entities": [
                { "position": [1.0, 2.0, 3.0] },
                { "position": [4.0, 5.0, 6.0] },
            ],
        });

        let count = load_level_auto(&mut world, &level);
        assert_eq!(count, 2, "old format should spawn 2 entities");
        // LevelData resource must NOT be present for legacy path.
        assert!(
            world.resource::<LevelData>().is_none(),
            "old format should not store LevelData resource"
        );
    }

    /// New-format JSON (has `width` + `height`) must be detected and stored
    /// as a `LevelData` world resource.
    #[test]
    fn auto_detect_new_format() {
        let mut world = World::new();
        let level = serde_json::json!({
            "name": "Test Arena",
            "width": 256,
            "height": 256,
            "entities": [
                { "position": [0.0, 0.0, 0.0] },
            ],
        });

        let count = load_level_auto(&mut world, &level);
        assert_eq!(count, 1);

        let ld = world
            .resource::<LevelData>()
            .expect("LevelData should be stored");
        assert_eq!(ld.name.as_deref(), Some("Test Arena"));
        assert_eq!(ld.width, Some(256));
        assert_eq!(ld.height, Some(256));
    }

    /// Entities in the new format must actually be spawned into the world.
    #[test]
    fn new_format_entities_spawn() {
        let mut world = World::new();
        let level = serde_json::json!({
            "width": 64,
            "height": 64,
            "entities": [
                { "position": [1.0, 0.0, 0.0] },
                { "position": [2.0, 0.0, 0.0] },
                { "position": [3.0, 0.0, 0.0] },
            ],
        });

        let count = load_level_auto(&mut world, &level);
        assert_eq!(count, 3);

        // Verify entities actually exist by querying LocalTransform.
        let query = euca_ecs::Query::<(Entity, &LocalTransform)>::new(&world);
        let spawned: Vec<_> = query.iter().collect();
        assert_eq!(spawned.len(), 3, "3 entities should be queryable");
    }

    /// Camera config in the new format must be applied to the world resource.
    #[test]
    fn new_format_camera_applied() {
        let mut world = World::new();
        world.insert_resource(euca_render::Camera::default());

        let level = serde_json::json!({
            "width": 64,
            "height": 64,
            "entities": [],
            "camera": {
                "eye": [10.0, 20.0, 30.0],
                "target": [0.0, 0.0, 0.0],
                "fov_y": 1.2,
            },
        });

        load_level_auto(&mut world, &level);

        let cam = world
            .resource::<euca_render::Camera>()
            .expect("Camera should exist");
        assert!((cam.eye.x - 10.0).abs() < f32::EPSILON);
        assert!((cam.eye.y - 20.0).abs() < f32::EPSILON);
        assert!((cam.eye.z - 30.0).abs() < f32::EPSILON);
        assert!((cam.target.x).abs() < f32::EPSILON);
        assert!((cam.target.y).abs() < f32::EPSILON);
        assert!((cam.target.z).abs() < f32::EPSILON);
        assert!((cam.fov_y - 1.2).abs() < f32::EPSILON);
    }
}
