use axum::Json;
use axum::extract::State;

use euca_math::Vec3;
use euca_scene::{GlobalTransform, LocalTransform};

use crate::state::SharedWorld;

use super::MessageResponse;

/// POST /audio/play — load and play a sound file
pub async fn audio_play(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let path = req
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let volume = req.get("volume").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let looping = req.get("loop").and_then(|v| v.as_bool()).unwrap_or(false);
    let position = req.get("position").and_then(|v| v.as_array()).map(|a| {
        Vec3::new(
            a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        )
    });
    let spatial = position.is_some();
    let max_distance = req
        .get("max_distance")
        .and_then(|v| v.as_f64())
        .unwrap_or(50.0) as f32;

    if path.is_empty() {
        return Json(serde_json::json!({"ok": false, "error": "Missing 'path' to audio file"}));
    }

    // Load clip
    let clip_result = world.with(|w, _| {
        let engine = w.resource_mut::<euca_audio::AudioEngine>();
        match engine {
            Some(eng) => eng.load(&path).map_err(|e| e.to_string()),
            None => Err("AudioEngine not initialized".to_string()),
        }
    });

    let clip = match clip_result {
        Ok(c) => c,
        Err(e) => {
            return Json(serde_json::json!({"ok": false, "error": e}));
        }
    };

    // Spawn audio source entity
    let entity_id = world.with(|w, _| {
        let mut src = if spatial {
            euca_audio::AudioSource::spatial(clip, max_distance)
        } else {
            euca_audio::AudioSource::global(clip)
        };
        src = src.with_volume(volume).with_looping(looping);

        let entity = w.spawn(src);

        if let Some(pos) = position {
            let transform = euca_math::Transform::from_translation(pos);
            w.insert(entity, LocalTransform(transform));
            w.insert(entity, GlobalTransform::default());
        }

        entity.index()
    });

    Json(serde_json::json!({
        "ok": true,
        "entity_id": entity_id,
        "clip": clip.0,
        "spatial": spatial,
    }))
}

/// POST /audio/stop — stop an audio source by entity ID
pub async fn audio_stop(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let ok = world.with(|w, _| {
        if let Some(entity) = super::find_entity(w, entity_id) {
            if let Some(src) = w.get_mut::<euca_audio::AudioSource>(entity) {
                src.playing = false;
            }
            true
        } else {
            false
        }
    });

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("Stopped audio on entity {entity_id}")
        } else {
            format!("Entity {entity_id} not found")
        }),
    })
}

/// GET /audio/list — list active audio sources
pub async fn audio_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let sources = world.with_world(|w| {
        let query = euca_ecs::Query::<(euca_ecs::Entity, &euca_audio::AudioSource)>::new(w);
        query
            .iter()
            .map(|(e, src)| {
                serde_json::json!({
                    "entity_id": e.index(),
                    "clip": src.clip.0,
                    "volume": src.volume,
                    "spatial": src.spatial,
                    "playing": src.playing,
                    "looping": src.looping,
                })
            })
            .collect::<Vec<_>>()
    });

    Json(serde_json::json!({"sources": sources, "count": sources.len()}))
}
