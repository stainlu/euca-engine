use axum::Json;
use axum::extract::State;

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

/// POST /animation/load — load a glTF file with animations
pub async fn animation_load(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let path = req
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if path.is_empty() {
        return Json(serde_json::json!({"ok": false, "error": "Missing 'path' to glTF file"}));
    }

    let result = euca_asset::load_gltf(&path);
    let scene = match result {
        Ok(s) => s,
        Err(e) => {
            return Json(serde_json::json!({"ok": false, "error": e}));
        }
    };

    let clip_names: Vec<String> = scene.animations.iter().map(|c| c.name.clone()).collect();
    let has_skeleton = scene.skeleton.is_some();
    let mesh_count = scene.meshes.len();

    // Store animations and skeleton in the library
    let stored = world.with(|w, _| {
        let lib = match w.resource_mut::<euca_asset::AnimationLibrary>() {
            Some(lib) => lib,
            None => return false,
        };

        if let Some(skeleton) = scene.skeleton {
            lib.add_skeleton(skeleton);
        }
        for clip in scene.animations {
            lib.add_clip(clip);
        }
        true
    });

    if !stored {
        return Json(serde_json::json!({
            "ok": false,
            "error": "AnimationLibrary resource not initialized",
        }));
    }

    Json(serde_json::json!({
        "ok": true,
        "meshes": mesh_count,
        "skeleton": has_skeleton,
        "animations": clip_names,
    }))
}

/// POST /animation/play — start animation on an entity
pub async fn animation_play(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let clip_index = req.get("clip").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let speed = req.get("speed").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let looping = req.get("loop").and_then(|v| v.as_bool()).unwrap_or(true);

    let ok = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return false,
        };

        let mut animator = euca_asset::SkeletalAnimator::new(clip_index);
        animator.speed = speed;
        animator.looping = looping;
        w.insert(entity, animator);
        true
    });

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("Playing animation clip {clip_index} on entity {entity_id}")
        } else {
            format!("Entity {entity_id} not found")
        }),
    })
}

/// POST /animation/stop — stop animation on an entity
pub async fn animation_stop(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let ok = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return false,
        };

        if let Some(animator) = w.get_mut::<euca_asset::SkeletalAnimator>(entity) {
            animator.playing = false;
        }
        true
    });

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("Stopped animation on entity {entity_id}")
        } else {
            format!("Entity {entity_id} not found")
        }),
    })
}

/// GET /animation/list — list loaded animation clips
pub async fn animation_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let clips = world.with_world(|w| {
        w.resource::<euca_asset::AnimationLibrary>()
            .map(|lib| {
                lib.clips
                    .iter()
                    .enumerate()
                    .map(|(i, clip)| {
                        serde_json::json!({
                            "index": i,
                            "name": clip.name,
                            "duration": clip.duration,
                            "channels": clip.channels.len(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });

    Json(serde_json::json!({"clips": clips, "count": clips.len()}))
}
