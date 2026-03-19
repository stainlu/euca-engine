use axum::Json;
use axum::extract::State;

use crate::state::SharedWorld;

use super::{MessageResponse, StepRequest, StepResponse};

/// POST /step — advance simulation
pub async fn step(
    State(world): State<SharedWorld>,
    Json(req): Json<StepRequest>,
) -> Json<StepResponse> {
    let resp = world.with(|w, schedule| {
        let ticks = req.ticks.min(10000);
        for _ in 0..ticks {
            schedule.run(w);
        }
        StepResponse {
            ticks_advanced: ticks,
            new_tick: w.current_tick(),
            entity_count: w.entity_count(),
        }
    });
    Json(resp)
}

/// POST /play — start simulation
pub async fn play(State(world): State<SharedWorld>) -> Json<MessageResponse> {
    world.with_world(|w| {
        if let Some(ctrl) = w.resource::<crate::control::EngineControl>() {
            ctrl.set_playing(true);
        }
    });
    Json(MessageResponse {
        ok: true,
        message: Some("Simulation playing".into()),
    })
}

/// POST /pause — pause simulation
pub async fn pause(State(world): State<SharedWorld>) -> Json<MessageResponse> {
    world.with_world(|w| {
        if let Some(ctrl) = w.resource::<crate::control::EngineControl>() {
            ctrl.set_playing(false);
        }
    });
    Json(MessageResponse {
        ok: true,
        message: Some("Simulation paused".into()),
    })
}
