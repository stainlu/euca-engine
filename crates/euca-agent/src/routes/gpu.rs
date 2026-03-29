use axum::Json;
use axum::extract::State;

use crate::state::SharedWorld;

use super::GpuInfo;

/// GET /engine/gpu -- report GPU backend and capability information.
pub async fn engine_gpu(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let info = world.with_world(|w| w.resource::<GpuInfo>().cloned());
    match info {
        Some(gpu) => Json(serde_json::to_value(gpu).unwrap_or_else(
            |e| serde_json::json!({"error": format!("serialization failed: {e}")}),
        )),
        None => Json(serde_json::json!({"error": "GPU info not available (headless mode?)"})),
    }
}
