use axum::Json;
use axum::extract::State;

use crate::state::SharedWorld;

use super::MessageResponse;

/// POST /input/bind — bind a key to an action
pub async fn input_bind(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let key_str = req
        .get("key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let action = req
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if key_str.is_empty() || action.is_empty() {
        return Json(MessageResponse {
            ok: false,
            message: Some("Missing 'key' or 'action'".into()),
        });
    }

    let key = euca_input::InputKey::Key(key_str.clone());

    world.with(|w, _| {
        if let Some(map) = w.resource_mut::<euca_input::ActionMap>() {
            map.bind(key, &action);
        }
    });

    Json(MessageResponse {
        ok: true,
        message: Some(format!("Bound '{key_str}' to '{action}'")),
    })
}

/// POST /input/unbind — remove a key binding
pub async fn input_unbind(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let key_str = req
        .get("key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if key_str.is_empty() {
        return Json(MessageResponse {
            ok: false,
            message: Some("Missing 'key'".into()),
        });
    }

    let key = euca_input::InputKey::Key(key_str.clone());

    let removed = world.with(|w, _| {
        w.resource_mut::<euca_input::ActionMap>()
            .and_then(|map| map.unbind(&key))
    });

    Json(MessageResponse {
        ok: removed.is_some(),
        message: Some(if removed.is_some() {
            format!("Unbound '{key_str}'")
        } else {
            format!("No binding for '{key_str}'")
        }),
    })
}

/// GET /input/list — list all key bindings
pub async fn input_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let bindings = world.with_world(|w| {
        w.resource::<euca_input::ActionMap>()
            .map(|map| {
                map.bindings()
                    .iter()
                    .map(|(key, action)| {
                        let key_str = match key {
                            euca_input::InputKey::Key(k) => k.clone(),
                            euca_input::InputKey::MouseLeft => "MouseLeft".into(),
                            euca_input::InputKey::MouseRight => "MouseRight".into(),
                            euca_input::InputKey::MouseMiddle => "MouseMiddle".into(),
                            euca_input::InputKey::GamepadButton(id) => {
                                format!("GamepadButton({id})")
                            }
                            euca_input::InputKey::GamepadAxis(id, axis) => {
                                format!("GamepadAxis({id}, {axis:?})")
                            }
                        };
                        serde_json::json!({"key": key_str, "action": action})
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });

    Json(serde_json::json!({"bindings": bindings, "count": bindings.len()}))
}

/// POST /input/context/push — push an input context
pub async fn input_context_push(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let ctx_str = req
        .get("context")
        .and_then(|v| v.as_str())
        .unwrap_or("gameplay");

    let ctx = match ctx_str {
        "menu" => euca_input::InputContext::Menu,
        "editor" => euca_input::InputContext::Editor,
        _ => euca_input::InputContext::Gameplay,
    };

    world.with(|w, _| {
        if let Some(stack) = w.resource_mut::<euca_input::InputContextStack>() {
            stack.push(ctx);
        }
    });

    Json(MessageResponse {
        ok: true,
        message: Some(format!("Pushed context '{ctx_str}'")),
    })
}

/// POST /input/context/pop — pop the top input context
pub async fn input_context_pop(State(world): State<SharedWorld>) -> Json<MessageResponse> {
    let popped = world.with(|w, _| {
        w.resource_mut::<euca_input::InputContextStack>()
            .and_then(|stack| stack.pop())
    });

    Json(MessageResponse {
        ok: popped.is_some(),
        message: Some(if let Some(ctx) = popped {
            format!("Popped context: {ctx:?}")
        } else {
            "Cannot pop last context".into()
        }),
    })
}
