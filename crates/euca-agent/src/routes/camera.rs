use axum::Json;
use axum::extract::State;

use euca_math::Vec3;
use euca_scene::GlobalTransform;

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

/// GET /camera — get current camera state
pub async fn camera_get(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let data = world.with_world(|w| {
        w.resource::<euca_render::Camera>().map(|cam| {
            serde_json::json!({
                "eye": [cam.eye.x, cam.eye.y, cam.eye.z],
                "target": [cam.target.x, cam.target.y, cam.target.z],
                "fov_y": cam.fov_y,
                "orthographic": cam.orthographic,
                "ortho_size": cam.ortho_size,
            })
        })
    });
    Json(data.unwrap_or(serde_json::json!({"error": "No camera"})))
}

/// POST /camera — set camera position and target
pub async fn camera_set(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    world.with(|w, _| {
        if let Some(cam) = w.resource_mut::<euca_render::Camera>() {
            if let Some(eye) = req.get("eye").and_then(|v| v.as_array())
                && eye.len() == 3
            {
                cam.eye = Vec3::new(
                    eye[0].as_f64().unwrap_or(0.0) as f32,
                    eye[1].as_f64().unwrap_or(0.0) as f32,
                    eye[2].as_f64().unwrap_or(0.0) as f32,
                );
            }
            if let Some(target) = req.get("target").and_then(|v| v.as_array())
                && target.len() == 3
            {
                cam.target = Vec3::new(
                    target[0].as_f64().unwrap_or(0.0) as f32,
                    target[1].as_f64().unwrap_or(0.0) as f32,
                    target[2].as_f64().unwrap_or(0.0) as f32,
                );
            }
        }
    });
    // Set CameraOverride so editor doesn't override with mouse orbit
    world.with_world(|w| {
        if let Some(co) = w.resource::<crate::control::CameraOverride>() {
            co.set();
        }
    });

    Json(MessageResponse {
        ok: true,
        message: Some("Camera updated".into()),
    })
}

/// POST /camera/view — apply a named view preset (top, front, right, etc.)
pub async fn camera_view(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let view = req
        .get("view")
        .and_then(|v| v.as_str())
        .unwrap_or("perspective");

    let ok = world.with(|w, _| {
        if let Some(cam) = w.resource_mut::<euca_render::Camera>() {
            cam.apply_preset(view)
        } else {
            false
        }
    });

    if ok {
        world.with_world(|w| {
            if let Some(co) = w.resource::<crate::control::CameraOverride>() {
                co.set();
            }
        });
    }

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("Camera set to {view} view")
        } else {
            format!("Unknown view: {view}. Use: top, front, back, right, left, perspective")
        }),
    })
}

/// POST /camera/focus — focus camera on a specific entity
pub async fn camera_focus(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req
        .get("entity_id")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let entity_id = match entity_id {
        Some(id) => id,
        None => {
            return Json(MessageResponse {
                ok: false,
                message: Some("Missing entity_id".into()),
            });
        }
    };

    let result = world.with(|w, _| {
        let entity = find_entity(w, entity_id)?;
        let pos = w
            .get::<GlobalTransform>(entity)
            .map(|gt| gt.0.translation)?;
        let cam = w.resource_mut::<euca_render::Camera>()?;
        cam.target = pos;
        let offset = cam.eye - cam.target;
        let dist = offset.length().clamp(5.0, 20.0);
        let dir = if offset.length() > 0.001 {
            offset.normalize()
        } else {
            Vec3::new(0.6, 0.5, 0.6).normalize()
        };
        cam.eye = pos + dir * dist;
        cam.orthographic = false;
        Some(pos)
    });

    if let Some(pos) = result {
        world.with_world(|w| {
            if let Some(co) = w.resource::<crate::control::CameraOverride>() {
                co.set();
            }
        });
        Json(MessageResponse {
            ok: true,
            message: Some(format!(
                "Focused on entity {} at ({:.1}, {:.1}, {:.1})",
                entity_id, pos.x, pos.y, pos.z
            )),
        })
    } else {
        Json(MessageResponse {
            ok: false,
            message: Some(format!("Entity {entity_id} not found")),
        })
    }
}
