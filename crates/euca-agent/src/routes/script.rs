//! Lua scripting endpoints — attach scripts to entities, list scripted entities.

use axum::Json;
use axum::extract::State;
use euca_ecs::{Entity, Query};
use serde::Deserialize;

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

/// Request body for `POST /script/load`.
#[derive(Deserialize)]
pub struct ScriptLoadRequest {
    /// Entity ID to attach the script to.
    pub entity_id: u32,
    /// Path to the `.lua` script file.
    pub path: String,
}

/// POST /script/load — attach a Lua script to an entity
pub async fn script_load(
    State(world): State<SharedWorld>,
    Json(req): Json<ScriptLoadRequest>,
) -> Json<MessageResponse> {
    let result = world.with(|w, _schedule| {
        let entity = match find_entity(w, req.entity_id) {
            Some(e) => e,
            None => {
                return MessageResponse {
                    ok: false,
                    message: Some(format!("Entity {} not found", req.entity_id)),
                };
            }
        };

        let path = std::path::Path::new(&req.path);

        // Load the script file into the engine.
        let engine = w.resource_mut::<euca_script::ScriptEngine>();
        let Some(engine) = engine else {
            return MessageResponse {
                ok: false,
                message: Some("No ScriptEngine resource in world".into()),
            };
        };

        if let Err(e) = engine.load_file(path) {
            return MessageResponse {
                ok: false,
                message: Some(format!("Failed to load script: {e}")),
            };
        }

        // Derive the script name from the file name (matches ScriptEngine convention).
        let script_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.lua");

        // Attach the ScriptComponent to the entity.
        let component = euca_script::ScriptComponent::new(script_name);
        w.insert(entity, component);

        MessageResponse {
            ok: true,
            message: Some(format!(
                "Script '{}' attached to entity {}",
                script_name, req.entity_id
            )),
        }
    });
    Json(result)
}

/// GET /script/list — list entities with scripts attached
pub async fn script_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let result = world.with_world(|w| {
        let scripted: Vec<serde_json::Value> = {
            let query = Query::<(Entity, &euca_script::ScriptComponent)>::new(w);
            query
                .iter()
                .map(|(entity, sc)| {
                    serde_json::json!({
                        "entity_id": entity.index(),
                        "script_name": sc.script_name,
                        "update_fn": sc.update_fn,
                        "enabled": sc.enabled,
                    })
                })
                .collect()
        };

        serde_json::json!({
            "count": scripted.len(),
            "entities": scripted,
        })
    });
    Json(result)
}
