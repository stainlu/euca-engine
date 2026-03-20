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

// ── State Machine ──

#[derive(serde::Deserialize)]
pub struct StateMachineRequest {
    pub entity_id: u32,
    #[serde(default)]
    pub initial_state: usize,
    #[serde(default)]
    pub states: Vec<StateDef>,
}

#[derive(serde::Deserialize)]
pub struct StateDef {
    pub name: String,
    pub clip: usize,
    #[serde(default = "default_true")]
    pub looping: bool,
    #[serde(default = "default_speed")]
    pub speed: f32,
}

fn default_true() -> bool {
    true
}
fn default_speed() -> f32 {
    1.0
}

/// POST /animation/state-machine
pub async fn animation_state_machine(
    State(world): State<SharedWorld>,
    Json(req): Json<StateMachineRequest>,
) -> Json<MessageResponse> {
    let ok = world.with(|w, _| {
        let entity = match find_entity(w, req.entity_id) {
            Some(e) => e,
            None => return false,
        };
        let mut sm = euca_animation::AnimStateMachine::new(req.initial_state);
        for sd in &req.states {
            let idx = sm.add_state(&sd.name, sd.clip);
            if let Some(s) = sm.state_mut(idx) {
                s.speed = sd.speed;
                s.looping = sd.looping;
            }
        }
        w.insert(entity, sm);
        true
    });
    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("State machine set on entity {}", req.entity_id)
        } else {
            format!("Entity {} not found", req.entity_id)
        }),
    })
}

// ── Montage ──

#[derive(serde::Deserialize)]
pub struct MontageRequest {
    pub entity_id: u32,
    pub clip: usize,
    pub clip_duration: f32,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default = "default_blend")]
    pub blend_in: f32,
    #[serde(default = "default_blend")]
    pub blend_out: f32,
    #[serde(default)]
    pub bone_mask: Option<Vec<usize>>,
}

fn default_blend() -> f32 {
    0.1
}

/// POST /animation/montage
pub async fn animation_montage(
    State(world): State<SharedWorld>,
    Json(req): Json<MontageRequest>,
) -> Json<MessageResponse> {
    let ok = world.with(|w, _| {
        let entity = match find_entity(w, req.entity_id) {
            Some(e) => e,
            None => return false,
        };
        let montage = euca_animation::AnimationMontage {
            clip_index: req.clip,
            speed: req.speed,
            blend_in: req.blend_in,
            blend_out: req.blend_out,
            bone_mask: req.bone_mask,
        };
        if w.get::<euca_animation::MontagePlayer>(entity).is_some() {
            if let Some(player) = w.get_mut::<euca_animation::MontagePlayer>(entity) {
                player.play(montage, req.clip_duration);
            }
        } else {
            let mut player = euca_animation::MontagePlayer::new();
            player.play(montage, req.clip_duration);
            w.insert(entity, player);
        }
        true
    });
    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!(
                "Montage (clip {}) triggered on entity {}",
                req.clip, req.entity_id
            )
        } else {
            format!("Entity {} not found", req.entity_id)
        }),
    })
}
