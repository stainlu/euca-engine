//! Profiler endpoint — exposes per-frame timing data.

use axum::Json;
use axum::extract::State;
use euca_core::Profiler;

use crate::state::SharedWorld;

/// GET /profile — return frame timing breakdown
pub async fn profile(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let result = world.with_world(|w| {
        let Some(profiler) = w.resource::<Profiler>() else {
            return serde_json::json!({"error": "No Profiler resource"});
        };

        let fps = profiler.fps();
        let frame_ms = profiler.avg_frame_time_ms();
        let sections: Vec<serde_json::Value> = profiler
            .frame_summary()
            .iter()
            .map(|(name, us, kind)| {
                serde_json::json!({
                    "name": name,
                    "us": (*us * 10.0).round() / 10.0,
                    "kind": format!("{kind:?}"),
                })
            })
            .collect();

        serde_json::json!({
            "fps": (fps * 10.0).round() / 10.0,
            "frame_ms": (frame_ms * 10.0).round() / 10.0,
            "sections": sections,
        })
    });
    Json(result)
}

#[cfg(test)]
mod tests {
    use euca_core::{Profiler, profiler_begin, profiler_end};

    #[test]
    fn profile_response_structure() {
        // Simulate a few profiled frames
        let mut profiler = Profiler::default();
        for _ in 0..3 {
            profiler_begin(&mut profiler, "physics");
            profiler_end(&mut profiler);
            profiler_begin(&mut profiler, "render_draw");
            profiler_end(&mut profiler);
            profiler.end_frame();
        }

        // Verify the data the route handler would read
        assert!(profiler.fps() > 0.0);
        assert!(profiler.avg_frame_time_ms() >= 0.0);
        // frame_summary is empty after end_frame (sections cleared),
        // but fps/avg are computed from archived history
        assert!(profiler.frame_summary().is_empty());
    }

    #[test]
    fn profile_sections_before_end_frame() {
        let mut profiler = Profiler::default();
        profiler_begin(&mut profiler, "physics");
        profiler_end(&mut profiler);
        profiler_begin(&mut profiler, "gameplay");
        profiler_end(&mut profiler);

        let summary = profiler.frame_summary();
        assert_eq!(summary.len(), 2);
        assert_eq!(summary[0].0, "physics");
        assert_eq!(summary[1].0, "gameplay");
    }
}
