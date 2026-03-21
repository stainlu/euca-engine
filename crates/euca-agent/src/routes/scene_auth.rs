use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use euca_ecs::{Entity, Query};
use euca_math::Vec3;
use euca_scene::{GlobalTransform, LocalTransform};

use crate::state::SharedWorld;

use super::{MessageResponse, RichEntityData, apply_physics_body, read_entity_data};

/// POST /scene/save — save current world state as JSON
pub async fn scene_save(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let path = req
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("scene.json")
        .to_string();

    let scene_data = world.with_world(|w| {
        let entities: Vec<RichEntityData> = {
            let query = Query::<Entity>::new(w);
            query.iter().map(|e| read_entity_data(w, e)).collect()
        };
        serde_json::json!({
            "version": 1,
            "tick": w.current_tick(),
            "entities": entities,
        })
    });

    match std::fs::write(
        &path,
        serde_json::to_string_pretty(&scene_data).expect("scene data serialization failed"),
    ) {
        Ok(()) => Json(MessageResponse {
            ok: true,
            message: Some(format!("Scene saved to {path}")),
        }),
        Err(e) => Json(MessageResponse {
            ok: false,
            message: Some(format!("Save failed: {e}")),
        }),
    }
}

/// POST /scene/load — load scene from JSON file
pub async fn scene_load(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let path = req
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("scene.json")
        .to_string();

    let data = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            return Json(MessageResponse {
                ok: false,
                message: Some(format!("Cannot read {path}: {e}")),
            });
        }
    };

    let scene: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => {
            return Json(MessageResponse {
                ok: false,
                message: Some(format!("Invalid JSON: {e}")),
            });
        }
    };

    let entities = scene["entities"].as_array();
    let count = entities.map(|e| e.len()).unwrap_or(0);

    world.with(|w, _| {
        let existing: Vec<Entity> = {
            let query = Query::<Entity>::new(w);
            query.iter().collect()
        };
        for entity in existing {
            w.despawn(entity);
        }

        if let Some(entities) = entities {
            for ent in entities {
                let pos = ent["transform"]["position"]
                    .as_array()
                    .map(|a| {
                        [
                            a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                            a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                            a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        ]
                    })
                    .unwrap_or([0.0, 0.0, 0.0]);

                let mut transform =
                    euca_math::Transform::from_translation(Vec3::new(pos[0], pos[1], pos[2]));

                if let Some(scl) = ent["transform"]["scale"].as_array() {
                    transform.scale = Vec3::new(
                        scl.first().and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                        scl.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                        scl.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                    );
                }

                let entity = w.spawn(LocalTransform(transform));
                w.insert(entity, GlobalTransform::default());

                if let Some(pb) = ent["physics_body"].as_str() {
                    apply_physics_body(w, entity, pb);
                }
            }
        }
    });

    Json(MessageResponse {
        ok: true,
        message: Some(format!("Loaded {count} entities from {path}")),
    })
}

/// POST /screenshot — capture 3D viewport as PNG
pub async fn screenshot(
    State(world): State<SharedWorld>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let rx = world.with_world(|w| {
        w.resource::<crate::control::ScreenshotChannel>()
            .map(|ch| ch.request())
    });

    let rx = match rx {
        Some(rx) => rx,
        None => {
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
    };

    match tokio::time::timeout(std::time::Duration::from_secs(2), rx).await {
        Ok(Ok(png_bytes)) => {
            let path = std::env::temp_dir().join(format!(
                "euca_screenshot_{}.png",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system clock before UNIX epoch")
                    .as_millis()
            ));
            if let Err(e) = std::fs::write(&path, &png_bytes) {
                log::error!("Failed to write screenshot: {e}");
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
            Ok(Json(serde_json::json!({
                "ok": true,
                "path": path.to_string_lossy(),
                "size_bytes": png_bytes.len(),
            })))
        }
        Ok(Err(_)) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        Err(_) => Err(StatusCode::GATEWAY_TIMEOUT),
    }
}

/// POST /auth/login — authenticate via nit Ed25519 signature
pub async fn auth_login(
    State(world): State<SharedWorld>,
    Json(payload): Json<crate::auth::LoginPayload>,
) -> Result<Json<crate::auth::LoginResponse>, (StatusCode, Json<crate::auth::AuthError>)> {
    let auth_store = world.with_world(|w| w.resource::<crate::auth::AuthStore>().cloned());

    let auth_store = match auth_store {
        Some(store) => store,
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(crate::auth::AuthError {
                    ok: false,
                    error: "Auth not configured".into(),
                }),
            ));
        }
    };

    match auth_store.login(&payload) {
        Ok(token) => Ok(Json(crate::auth::LoginResponse {
            ok: true,
            session_token: token,
            agent_id: payload.agent_id,
        })),
        Err(e) => Err((
            StatusCode::UNAUTHORIZED,
            Json(crate::auth::AuthError {
                ok: false,
                error: e,
            }),
        )),
    }
}

/// GET /auth/status — check current auth session
pub async fn auth_status(
    State(world): State<SharedWorld>,
    headers: axum::http::HeaderMap,
) -> Json<serde_json::Value> {
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    let auth_store = world.with_world(|w| w.resource::<crate::auth::AuthStore>().cloned());

    if let (Some(token), Some(store)) = (token, auth_store)
        && let Some(agent_id) = store.validate(token)
    {
        return Json(serde_json::json!({
            "authenticated": true,
            "agent_id": agent_id,
        }));
    }

    Json(serde_json::json!({
        "authenticated": false,
    }))
}
