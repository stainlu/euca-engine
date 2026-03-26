//! Engine diagnostic endpoints — health checks for game state.

use axum::Json;
use axum::extract::State;
use euca_ecs::{Entity, Events, Query, World};

use crate::state::SharedWorld;

/// GET /diagnose — scan all entities and report problems
pub async fn diagnose(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let result = world.with_world(run_diagnostics);
    Json(result)
}

fn run_diagnostics(w: &World) -> serde_json::Value {
    let mut warnings: Vec<String> = Vec::new();
    let mut info: Vec<String> = Vec::new();

    let mut dynamic_count = 0u32;
    let mut kinematic_count = 0u32;
    let mut static_count = 0u32;
    let mut teams_with_spawn_points: std::collections::HashSet<u8> =
        std::collections::HashSet::new();
    let mut teams_seen: std::collections::HashSet<u8> = std::collections::HashSet::new();
    let mut positions: Vec<(u32, euca_math::Vec3)> = Vec::new();

    // Scan spawn points
    let query = Query::<(Entity, &euca_gameplay::SpawnPoint)>::new(w);
    for (_, sp) in query.iter() {
        teams_with_spawn_points.insert(sp.team);
    }

    // Scan all entities
    let query = Query::<Entity>::new(w);
    for entity in query.iter() {
        let eid = entity.index();

        // Check physics body type
        if let Some(pb) = w.get::<euca_physics::PhysicsBody>(entity) {
            match pb.body_type {
                euca_physics::RigidBodyType::Dynamic => dynamic_count += 1,
                euca_physics::RigidBodyType::Kinematic => kinematic_count += 1,
                euca_physics::RigidBodyType::Static => static_count += 1,
            }
        }

        // Track teams
        if let Some(team) = w.get::<euca_gameplay::Team>(entity) {
            teams_seen.insert(team.0);
        }

        // Check: AutoCombat without Velocity
        if w.get::<euca_gameplay::AutoCombat>(entity).is_some()
            && w.get::<euca_physics::Velocity>(entity).is_none()
        {
            let role = w
                .get::<euca_gameplay::EntityRole>(entity)
                .map(|r| format!("{r:?}"))
                .unwrap_or_default();
            // Static towers don't need Velocity (they're stationary)
            let is_static = w
                .get::<euca_physics::PhysicsBody>(entity)
                .is_some_and(|pb| pb.body_type == euca_physics::RigidBodyType::Static);
            if !is_static {
                warnings.push(format!(
                    "E{eid} ({role}): has AutoCombat but no Velocity — cannot move"
                ));
            }
        }

        // Check: Velocity without PhysicsBody
        if w.get::<euca_physics::Velocity>(entity).is_some()
            && w.get::<euca_physics::PhysicsBody>(entity).is_none()
        {
            warnings.push(format!(
                "E{eid}: has Velocity but no PhysicsBody — physics won't integrate"
            ));
        }

        // Check: Dead without RespawnTimer
        if w.get::<euca_gameplay::Dead>(entity).is_some()
            && w.get::<euca_gameplay::RespawnTimer>(entity).is_none()
        {
            let has_team = w.get::<euca_gameplay::Team>(entity).is_some();
            if has_team {
                warnings.push(format!(
                    "E{eid}: Dead with Team but no RespawnTimer — stuck dead"
                ));
            }
        }

        // Check: Health entity without GlobalTransform (won't render health bar)
        if w.get::<euca_gameplay::Health>(entity).is_some()
            && w.get::<euca_scene::GlobalTransform>(entity).is_none()
        {
            warnings.push(format!(
                "E{eid}: has Health but no GlobalTransform — health bar won't render"
            ));
        }

        // Check: MeshRenderer without GlobalTransform (won't render)
        if w.get::<euca_render::MeshRenderer>(entity).is_some()
            && w.get::<euca_scene::GlobalTransform>(entity).is_none()
        {
            warnings.push(format!(
                "E{eid}: has MeshRenderer but no GlobalTransform — invisible"
            ));
        }

        // Check: MeshRenderer without MaterialRef (will use default — likely unintended)
        if w.get::<euca_render::MeshRenderer>(entity).is_some()
            && w.get::<euca_render::MaterialRef>(entity).is_none()
        {
            warnings.push(format!(
                "E{eid}: has MeshRenderer but no MaterialRef — missing material"
            ));
        }

        // Check: alive entity with zero health (should be Dead)
        if let Some(health) = w.get::<euca_gameplay::Health>(entity)
            && health.current <= 0.0
            && w.get::<euca_gameplay::Dead>(entity).is_none()
        {
            warnings.push(format!(
                "E{eid}: Health={:.0}/{:.0} but not Dead — death_check_system may not be running",
                health.current, health.max
            ));
        }

        // Track positions for overlap detection
        if let Some(gt) = w.get::<euca_scene::GlobalTransform>(entity) {
            let pos = gt.0.translation;
            if w.get::<euca_render::MeshRenderer>(entity).is_some() {
                positions.push((eid, pos));
            }
        }
    }

    // Check: teams without spawn points
    for team in &teams_seen {
        if !teams_with_spawn_points.contains(team) {
            warnings.push(format!(
                "No SpawnPoint for team {team} — heroes will respawn at fallback (0,2,0)"
            ));
        }
    }

    // Check: entities at same position (z-fighting / accidental overlap)
    let overlap_threshold = 0.01f32;
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let dist = (positions[i].1 - positions[j].1).length();
            if dist < overlap_threshold {
                warnings.push(format!(
                    "E{} and E{} are at the same position ({:.1},{:.1},{:.1}) — z-fighting",
                    positions[i].0,
                    positions[j].0,
                    positions[i].1.x,
                    positions[i].1.y,
                    positions[i].1.z,
                ));
            }
        }
    }

    // Check: GameState
    if w.resource::<euca_gameplay::GameState>().is_none() {
        info.push("No GameState — scoring and respawn won't work".into());
    }

    // Assertion summary
    {
        let query = Query::<(Entity, &euca_gameplay::assertions::Assertion)>::new(w);
        let assertion_count = query.iter().count();
        if assertion_count > 0 {
            let with_results = query
                .iter()
                .filter(|(_, a)| a.last_result.is_some())
                .count();
            let passed = query
                .iter()
                .filter(|(_, a)| a.last_result.as_ref().is_some_and(|r| r.passed))
                .count();
            let failed = with_results - passed;
            info.push(format!(
                "Assertions: {assertion_count} defined, {with_results} evaluated ({passed} pass, {failed} fail)"
            ));
        }
    }

    // Physics summary
    info.push(format!(
        "Physics: {dynamic_count} Dynamic, {kinematic_count} Kinematic, {static_count} Static"
    ));

    // Event summary
    if let Some(events) = w.resource::<Events>() {
        let damage_count = events.read::<euca_gameplay::DamageEvent>().count();
        let death_count = events.read::<euca_gameplay::DeathEvent>().count();
        let spawn_count = events.read::<euca_gameplay::RuleSpawnEvent>().count();
        if damage_count + death_count + spawn_count > 0 {
            info.push(format!(
                "Events: {damage_count} DamageEvent, {death_count} DeathEvent, {spawn_count} RuleSpawnEvent"
            ));
        }
    }

    serde_json::json!({
        "warnings": warnings,
        "warning_count": warnings.len(),
        "info": info,
        "entity_count": w.entity_count(),
    })
}

/// GET /events — show pending events this frame
pub async fn events_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let result = world.with_world(|w| {
        let events = match w.resource::<Events>() {
            Some(e) => e,
            None => return serde_json::json!({"error": "No Events resource"}),
        };

        let damage: Vec<_> = events
            .read::<euca_gameplay::DamageEvent>()
            .map(|e| {
                serde_json::json!({
                    "type": "DamageEvent",
                    "target": e.target.index(),
                    "amount": e.amount,
                    "source": e.source.map(|s| s.index()),
                })
            })
            .collect();

        let deaths: Vec<_> = events
            .read::<euca_gameplay::DeathEvent>()
            .map(|e| {
                serde_json::json!({
                    "type": "DeathEvent",
                    "entity": e.entity.index(),
                    "killer": e.killer.map(|k| k.index()),
                })
            })
            .collect();

        let spawns: Vec<_> = events
            .read::<euca_gameplay::RuleSpawnEvent>()
            .map(|e| {
                serde_json::json!({
                    "type": "RuleSpawnEvent",
                    "entity": e.entity.index(),
                    "mesh": e.mesh,
                    "color": e.color,
                })
            })
            .collect();

        let mut all = Vec::new();
        all.extend(damage);
        all.extend(deaths);
        all.extend(spawns);

        serde_json::json!({
            "events": all,
            "count": all.len(),
        })
    });
    Json(result)
}
