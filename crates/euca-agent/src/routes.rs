use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use euca_ecs::{Entity, Query};
use euca_scene::{GlobalTransform, LocalTransform};

use crate::state::SharedWorld;

// ── Response types ──

#[derive(Serialize)]
pub struct StatusResponse {
    pub engine: &'static str,
    pub version: &'static str,
    pub entity_count: u32,
    pub archetype_count: usize,
    pub tick: u64,
}

#[derive(Serialize)]
pub struct EntityData {
    pub id: u32,
    pub generation: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<[f32; 3]>,
}

#[derive(Serialize)]
pub struct ObserveResponse {
    pub tick: u64,
    pub entity_count: u32,
    pub entities: Vec<EntityData>,
}

#[derive(Deserialize)]
pub struct StepRequest {
    #[serde(default = "default_ticks")]
    pub ticks: u64,
}

fn default_ticks() -> u64 {
    1
}

#[derive(Serialize)]
pub struct StepResponse {
    pub ticks_advanced: u64,
    pub new_tick: u64,
    pub entity_count: u32,
}

#[derive(Deserialize)]
pub struct SpawnRequest {
    #[serde(default)]
    pub position: Option<[f32; 3]>,
}

#[derive(Serialize)]
pub struct SpawnResponse {
    pub entity_id: u32,
    pub entity_generation: u32,
}

#[derive(Deserialize)]
pub struct DespawnRequest {
    pub entity_id: u32,
    pub entity_generation: u32,
}

#[derive(Serialize)]
pub struct MessageResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

// ── Route handlers ──

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

/// POST /observe — query world state
pub async fn observe(State(world): State<SharedWorld>) -> Json<ObserveResponse> {
    let resp = world.with_world(|w| {
        let mut entities = Vec::new();

        // Query all entities with GlobalTransform (positioned in the world)
        let query = Query::<(Entity, &GlobalTransform)>::new(w);
        for (entity, gt) in query.iter() {
            entities.push(EntityData {
                id: entity.index(),
                generation: entity.generation(),
                position: Some([gt.0.translation.x, gt.0.translation.y, gt.0.translation.z]),
            });
        }

        // Also include entities without transforms
        let query_bare = Query::<Entity, euca_ecs::Without<GlobalTransform>>::new(w);
        for entity in query_bare.iter() {
            entities.push(EntityData {
                id: entity.index(),
                generation: entity.generation(),
                position: None,
            });
        }

        ObserveResponse {
            tick: w.current_tick(),
            entity_count: w.entity_count(),
            entities,
        }
    });
    Json(resp)
}

/// POST /step — advance simulation
pub async fn step(
    State(world): State<SharedWorld>,
    Json(req): Json<StepRequest>,
) -> Json<StepResponse> {
    let resp = world.with(|w, schedule| {
        let ticks = req.ticks.min(10000); // Cap to prevent abuse
        for _ in 0..ticks {
            schedule.run(w);
        }
        StepResponse {
            ticks_advanced: ticks,
            new_tick: w.current_tick(),
            entity_count: w.entity_count(),
        }
    });
    Json(resp)
}

/// POST /spawn — create a new entity
pub async fn spawn(
    State(world): State<SharedWorld>,
    Json(req): Json<SpawnRequest>,
) -> (StatusCode, Json<SpawnResponse>) {
    let resp = world.with(|w, _| {
        let pos = req.position.unwrap_or([0.0, 0.0, 0.0]);
        let transform =
            euca_math::Transform::from_translation(euca_math::Vec3::new(pos[0], pos[1], pos[2]));
        let entity = w.spawn(LocalTransform(transform));
        w.insert(entity, GlobalTransform::default());
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

/// POST /reset — reset the world (despawn all entities)
pub async fn reset(State(world): State<SharedWorld>) -> Json<MessageResponse> {
    let resp = world.with(|w, _| {
        // Collect all entities, then despawn them
        let entities: Vec<Entity> = {
            let query = Query::<Entity>::new(w);
            query.iter().collect()
        };
        for entity in entities {
            w.despawn(entity);
        }
        MessageResponse {
            ok: true,
            message: Some(format!("World reset. Tick: {}", w.current_tick())),
        }
    });
    Json(resp)
}

/// GET /schema — list available component types and actions
pub async fn schema() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "components": [
            {
                "name": "LocalTransform",
                "fields": {"translation": "[f32; 3]", "rotation": "[f32; 4]", "scale": "[f32; 3]"}
            },
            {
                "name": "GlobalTransform",
                "fields": {"translation": "[f32; 3]", "rotation": "[f32; 4]", "scale": "[f32; 3]"}
            }
        ],
        "actions": [
            {"name": "spawn", "params": {"position": "[f32; 3] (optional)"}},
            {"name": "despawn", "params": {"entity_id": "u32", "entity_generation": "u32"}},
            {"name": "step", "params": {"ticks": "u64 (default: 1)"}},
            {"name": "reset", "params": {}}
        ]
    }))
}
