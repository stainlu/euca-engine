use axum::Json;
use axum::extract::State;

use euca_math::Vec3;
use euca_scene::{GlobalTransform, LocalTransform};

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

/// POST /particle/spawn — create a particle emitter entity
pub async fn particle_create(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let position = req.get("position").and_then(|v| v.as_array()).map(|a| {
        Vec3::new(
            a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        )
    });
    let rate = req.get("rate").and_then(|v| v.as_f64()).unwrap_or(50.0) as f32;
    let lifetime = req.get("lifetime").and_then(|v| v.as_f64()).unwrap_or(2.0) as f32;
    let max = req.get("max").and_then(|v| v.as_u64()).unwrap_or(1000) as u32;

    let config = euca_particle::EmitterConfig {
        rate,
        particle_lifetime: lifetime,
        max_particles: max,
        ..Default::default()
    };

    let entity_id = world.with(|w, _| {
        let pos = position.unwrap_or(Vec3::ZERO);
        let transform = euca_math::Transform::from_translation(pos);
        let entity = w.spawn(euca_particle::ParticleEmitter::new(config));
        w.insert(entity, LocalTransform(transform));
        w.insert(entity, GlobalTransform::default());
        entity.index()
    });

    Json(serde_json::json!({
        "ok": true,
        "entity_id": entity_id,
    }))
}

/// POST /particle/stop — deactivate a particle emitter
pub async fn particle_stop(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let ok = world.with(|w, _| {
        if let Some(entity) = find_entity(w, entity_id)
            && let Some(emitter) = w.get_mut::<euca_particle::ParticleEmitter>(entity)
        {
            emitter.active = false;
            true
        } else {
            false
        }
    });

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("Stopped particle emitter {entity_id}")
        } else {
            format!("Entity {entity_id} not found or not an emitter")
        }),
    })
}

/// GET /particle/list — list active particle emitters
pub async fn particle_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let emitters = world.with_world(|w| {
        let query = euca_ecs::Query::<(euca_ecs::Entity, &euca_particle::ParticleEmitter)>::new(w);
        query
            .iter()
            .map(|(e, em)| {
                serde_json::json!({
                    "entity_id": e.index(),
                    "active": em.active,
                    "particle_count": em.particles.len(),
                    "rate": em.config.rate,
                })
            })
            .collect::<Vec<_>>()
    });

    Json(serde_json::json!({"emitters": emitters, "count": emitters.len()}))
}
