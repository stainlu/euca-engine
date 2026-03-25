//! Simulation probe endpoint — advance N ticks then evaluate assertions.
//!
//! The probe runs the simulation forward and checks assertions, combining
//! step + evaluate into a single atomic operation. Currently does NOT
//! restore world state after probing (World::clone not yet available).

use axum::Json;
use axum::extract::State;

use euca_gameplay::assertions;

use crate::state::SharedWorld;

/// POST /probe — advance simulation and evaluate assertions
///
/// Body: {
///   "ticks": 300,              // how many ticks to advance (0 = evaluate at current state)
///   "assertions": ["hero-alive", "towers-attack"],  // optional: only evaluate named assertions
///   "snapshot_before": true,   // optional: capture a snapshot before advancing
///   "snapshot_after": true,    // optional: capture a snapshot after advancing
/// }
pub async fn probe(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let ticks = req
        .get("ticks")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let filter_names: Option<Vec<String>> = req
        .get("assertions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        });
    let snapshot_before = req
        .get("snapshot_before")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let snapshot_after = req
        .get("snapshot_after")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let result = world.with(|w, schedule| {
        // Optionally capture before-snapshot
        let before = if snapshot_before {
            Some(super::snapshot::capture_snapshot_from_world(w, "probe_before".into()))
        } else {
            None
        };

        // Advance simulation
        let start_tick = w.current_tick();
        for _ in 0..ticks {
            schedule.run(w);
            w.tick();
        }
        let end_tick = w.current_tick();

        // Evaluate assertions
        let report = assertions::evaluate_assertions(w);

        // Filter results if specific assertion names were requested
        let results: Vec<_> = if let Some(names) = &filter_names {
            report
                .results
                .iter()
                .filter(|r| names.iter().any(|n| r.name == *n))
                .cloned()
                .collect()
        } else {
            report.results.clone()
        };

        let passed = results.iter().filter(|r| r.passed).count();
        let failed = results.iter().filter(|r| !r.passed).count();

        // Optionally capture after-snapshot
        let after = if snapshot_after {
            Some(super::snapshot::capture_snapshot_from_world(w, "probe_after".into()))
        } else {
            None
        };

        serde_json::json!({
            "ok": true,
            "ticks_advanced": ticks,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "total_assertions": results.len(),
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0,
            "results": results.iter().map(|r| serde_json::json!({
                "name": r.name,
                "severity": r.severity,
                "passed": r.passed,
                "message": r.message,
            })).collect::<Vec<_>>(),
            "before_snapshot": before,
            "after_snapshot": after,
        })
    });

    Json(result)
}
