use axum::Json;
use axum::extract::State;

use crate::state::SharedWorld;

use super::MessageResponse;

/// POST /ui/text — add text to HUD
pub async fn ui_text(
    State(world): State<SharedWorld>,
    Json(req): Json<crate::hud::HudElement>,
) -> Json<MessageResponse> {
    world.with(|w, _| {
        if let Some(canvas) = w.resource_mut::<crate::hud::HudCanvas>() {
            canvas.add(req.clone());
        }
    });
    Json(MessageResponse {
        ok: true,
        message: None,
    })
}

/// POST /ui/bar — add a bar to HUD
pub async fn ui_bar(
    State(world): State<SharedWorld>,
    Json(req): Json<crate::hud::HudElement>,
) -> Json<MessageResponse> {
    world.with(|w, _| {
        if let Some(canvas) = w.resource_mut::<crate::hud::HudCanvas>() {
            canvas.add(req.clone());
        }
    });
    Json(MessageResponse {
        ok: true,
        message: None,
    })
}

/// POST /ui/clear — remove all HUD elements
pub async fn ui_clear(State(world): State<SharedWorld>) -> Json<MessageResponse> {
    world.with(|w, _| {
        if let Some(canvas) = w.resource_mut::<crate::hud::HudCanvas>() {
            canvas.clear();
        }
    });
    Json(MessageResponse {
        ok: true,
        message: Some("HUD cleared".into()),
    })
}

/// GET /ui/list — list current HUD elements
pub async fn ui_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let elements = world.with_world(|w| {
        w.resource::<crate::hud::HudCanvas>()
            .map(|c| {
                c.elements
                    .iter()
                    .map(|e| serde_json::to_value(e).unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });
    Json(serde_json::json!({"elements": elements, "count": elements.len()}))
}
