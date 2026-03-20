use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;

use euca_ecs::{Entity, Query};
use euca_math::Vec3;
use euca_physics::Velocity;
use euca_render::{MaterialRef, MeshRenderer};
use euca_scene::{GlobalTransform, LocalTransform};

use crate::state::{Owner, SharedWorld};

use super::{
    ComponentPatch, DefaultAssets, DespawnRequest, MessageResponse, ObserveResponse,
    RichEntityData, SpawnRequest, SpawnResponse, StatusResponse, apply_collider,
    apply_physics_body, apply_velocity, find_entity, read_entity_data,
};

/// GET / — engine status
pub async fn status(State(world): State<SharedWorld>) -> Json<StatusResponse> {
    let resp = world.with_world(|w| StatusResponse {
        engine: "Euca Engine",
        version: env!("CARGO_PKG_VERSION"),
        entity_count: w.entity_count(),
        archetype_count: w.archetype_count(),
        tick: w.current_tick(),
    });
    Json(resp)
}

/// POST /observe — query full world state
pub async fn observe(State(world): State<SharedWorld>) -> Json<ObserveResponse> {
    let resp = world.with_world(|w| {
        let entities: Vec<RichEntityData> = {
            let query = Query::<Entity>::new(w);
            query.iter().map(|e| read_entity_data(w, e)).collect()
        };
        ObserveResponse {
            tick: w.current_tick(),
            entity_count: w.entity_count(),
            entities,
        }
    });
    Json(resp)
}

/// GET /entities/:id — query single entity
pub async fn get_entity(
    State(world): State<SharedWorld>,
    Path(id): Path<u32>,
) -> Result<Json<RichEntityData>, StatusCode> {
    let result = world.with_world(|w| find_entity(w, id).map(|e| read_entity_data(w, e)));
    match result {
        Some(data) => Ok(Json(data)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// POST /entities/:id/components — add/update components on an entity
pub async fn patch_entity(
    State(world): State<SharedWorld>,
    Path(id): Path<u32>,
    Json(patch): Json<ComponentPatch>,
) -> Result<Json<MessageResponse>, StatusCode> {
    world.with(|w, _| {
        let entity = match find_entity(w, id) {
            Some(e) => e,
            None => return Err(StatusCode::NOT_FOUND),
        };

        // Ownership check
        if let Some(owner) = w.get::<Owner>(entity) {
            match patch.agent_id {
                Some(aid) if aid == owner.0 => {}
                _ => {
                    return Ok(Json(MessageResponse {
                        ok: false,
                        message: Some("Permission denied: entity owned by another agent".into()),
                    }));
                }
            }
        }

        if let Some(t) = &patch.transform
            && let Some(lt) = w.get_mut::<LocalTransform>(entity)
        {
            if let Some(pos) = t.position {
                lt.0.translation = Vec3::new(pos[0], pos[1], pos[2]);
            }
            if let Some(scl) = t.scale {
                lt.0.scale = Vec3::new(scl[0], scl[1], scl[2]);
            }
            if let Some(rot) = t.rotation {
                lt.0.rotation = euca_math::Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]);
            }
        }
        if let Some(color) = &patch.color
            && let Some(assets) = w.resource::<DefaultAssets>().cloned()
            && let Some(mat) = assets.material(color)
        {
            if w.get::<MaterialRef>(entity).is_some() {
                if let Some(mr) = w.get_mut::<MaterialRef>(entity) {
                    mr.handle = mat;
                }
            } else {
                w.insert(entity, MaterialRef { handle: mat });
            }
        }
        if let Some(v) = &patch.velocity {
            apply_velocity(w, entity, v);
        }
        if let Some(c) = &patch.collider {
            apply_collider(w, entity, c);
        }
        if let Some(pb) = &patch.physics_body {
            apply_physics_body(w, entity, pb);
        }

        Ok(Json(MessageResponse {
            ok: true,
            message: None,
        }))
    })
}

/// POST /spawn — create entity with optional components
pub async fn spawn(
    State(world): State<SharedWorld>,
    Json(req): Json<SpawnRequest>,
) -> (StatusCode, Json<SpawnResponse>) {
    let resp = world.with(|w, _| {
        let pos = req.position.unwrap_or([0.0, 0.0, 0.0]);
        let scl = req.scale.unwrap_or([1.0, 1.0, 1.0]);
        let mut transform =
            euca_math::Transform::from_translation(Vec3::new(pos[0], pos[1], pos[2]));
        transform.scale = Vec3::new(scl[0], scl[1], scl[2]);

        let entity = w.spawn(LocalTransform(transform));
        w.insert(entity, GlobalTransform::default());

        if let Some(agent_id) = req.agent_id {
            w.insert(entity, Owner(agent_id));
        }

        if let Some(assets) = w.resource::<DefaultAssets>().cloned()
            && let Some(mesh_name) = &req.mesh
            && let Some(mesh) = assets.mesh(mesh_name)
        {
            w.insert(entity, MeshRenderer { mesh });
            let mat = req
                .color
                .as_deref()
                .and_then(|c| assets.material(c))
                .unwrap_or(assets.default_material);
            w.insert(entity, MaterialRef { handle: mat });
        }

        if let Some(v) = &req.velocity {
            apply_velocity(w, entity, v);
        }
        if let Some(c) = &req.collider {
            apply_collider(w, entity, c);
        }
        if let Some(pb) = &req.physics_body {
            apply_physics_body(w, entity, pb);
            if pb == "Dynamic" && w.get::<Velocity>(entity).is_none() {
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
        }
        if let Some(ref waypoints) = req.ai_patrol {
            let wps: Vec<euca_math::Vec3> = waypoints
                .iter()
                .map(|w| euca_math::Vec3::new(w[0], w[1], w[2]))
                .collect();
            let speed = req.combat_speed.unwrap_or(3.0);
            w.insert(entity, euca_gameplay::AiGoal::patrol(wps, speed));
        }

        // Economy + leveling + role
        if let Some(g) = req.gold {
            w.insert(entity, euca_gameplay::Gold(g));
            // Heroes with gold also get Level(1)
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

        SpawnResponse {
            entity_id: entity.index(),
            entity_generation: entity.generation(),
        }
    });
    (StatusCode::CREATED, Json(resp))
}

/// POST /despawn — remove an entity
pub async fn despawn(
    State(world): State<SharedWorld>,
    Json(req): Json<DespawnRequest>,
) -> Json<MessageResponse> {
    let resp = world.with(|w, _| {
        let entity = Entity::from_raw(req.entity_id, req.entity_generation);

        if let Some(owner) = w.get::<Owner>(entity) {
            match req.agent_id {
                Some(aid) if aid == owner.0 => {}
                _ => {
                    return MessageResponse {
                        ok: false,
                        message: Some("Permission denied: entity owned by another agent".into()),
                    };
                }
            }
        }

        if w.despawn(entity) {
            MessageResponse {
                ok: true,
                message: None,
            }
        } else {
            MessageResponse {
                ok: false,
                message: Some("Entity not found or already despawned".into()),
            }
        }
    });
    Json(resp)
}

/// POST /reset — reset the world (preserves Persistent entities like ground/lights)
pub async fn reset(State(world): State<SharedWorld>) -> Json<MessageResponse> {
    let resp = world.with(|w, _| {
        let entities: Vec<Entity> = {
            let query = Query::<Entity>::new(w);
            query
                .iter()
                .filter(|e| w.get::<crate::state::Persistent>(*e).is_none())
                .collect()
        };
        let count = entities.len();
        for entity in entities {
            w.despawn(entity);
        }
        MessageResponse {
            ok: true,
            message: Some(format!(
                "Reset: despawned {count} entities. Tick: {}",
                w.current_tick()
            )),
        }
    });
    Json(resp)
}

/// POST /entity/damage — apply damage to an entity
pub async fn entity_damage(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let amount = req.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

    world.with(|w, _| {
        if let Some(entity) = find_entity(w, entity_id)
            && let Some(events) = w.resource_mut::<euca_ecs::Events>()
        {
            events.send(euca_gameplay::DamageEvent {
                target: entity,
                amount,
                source: None,
            });
        }
    });

    Json(MessageResponse {
        ok: true,
        message: Some(format!("Applied {amount} damage to entity {entity_id}")),
    })
}

/// POST /entity/heal — heal an entity
pub async fn entity_heal(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let amount = req.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

    let ok = world.with(|w, _| {
        if let Some(entity) = find_entity(w, entity_id) {
            euca_gameplay::health::heal(w, entity, amount);
            true
        } else {
            false
        }
    });

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("Healed entity {entity_id} by {amount}")
        } else {
            format!("Entity {entity_id} not found")
        }),
    })
}

/// GET /schema — dynamic schema: all component types and actions
pub async fn schema() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "components": {
            "LocalTransform": {
                "fields": {"position": "[f32; 3]", "rotation": "[f32; 4] (xyzw)", "scale": "[f32; 3]"}
            },
            "GlobalTransform": {
                "fields": {"position": "[f32; 3]", "rotation": "[f32; 4] (xyzw)", "scale": "[f32; 3]"},
                "note": "Read-only. Computed from LocalTransform hierarchy."
            },
            "Velocity": {
                "fields": {"linear": "[f32; 3]", "angular": "[f32; 3]"}
            },
            "PhysicsBody": {
                "fields": {"body_type": "Dynamic | Static | Kinematic"}
            },
            "Collider": {
                "variants": {
                    "Aabb": {"hx": "f32", "hy": "f32", "hz": "f32"},
                    "Sphere": {"radius": "f32"}
                }
            }
        },
        "endpoints": {
            "GET /": "Engine status",
            "POST /observe": "Full world state (all entities with all components)",
            "GET /entities/:id": "Single entity with all components",
            "POST /entities/:id/components": "Add/update components on entity",
            "POST /spawn": "Create entity with optional components (position, scale, velocity, collider, physics_body)",
            "POST /despawn": "Remove entity by id + generation",
            "POST /step": "Advance simulation N ticks",
            "POST /reset": "Despawn all entities",
            "GET /schema": "This endpoint",
            "POST /entity/damage": "Apply damage to entity",
            "POST /entity/heal": "Heal entity",
            "POST /game/create": "Create match with mode and config",
            "GET /game/state": "Get match state and scores",
            "POST /trigger/create": "Create trigger zone",
            "POST /projectile/spawn": "Spawn projectile",
            "POST /ai/set": "Set AI behavior on entity"
        }
    }))
}
