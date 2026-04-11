//! Fork endpoints — counterfactual simulation for agent-driven reasoning.
//!
//! A fork is a deep copy of the main world that evolves independently.
//! Agents use forks to answer "what if?" questions: spawn a fork, apply
//! an intervention, advance the simulation, observe the outcome, compare
//! to the main world, then drop the fork. The main world is never
//! touched.
//!
//! Every fork shares the same [`Schedule`] as the main world, so
//! stepping a fork runs the same systems (physics, combat, AI, rules)
//! that would run on main. Forks are stored in [`SharedWorld`] keyed by
//! agent-chosen string ids.
//!
//! Endpoints:
//! - `POST   /fork`                    create a new fork
//! - `GET    /fork/list`               list active fork ids
//! - `DELETE /fork/{id}`               delete a fork
//! - `POST   /fork/{id}/step`          advance the fork by N ticks
//! - `POST   /fork/{id}/probe`         advance + evaluate assertions
//! - `GET    /fork/{id}/observe`       read the fork's entities
//!
//! All endpoints return JSON. 404 is signalled via `{"ok": false,
//! "error": "fork '...' not found"}` rather than an HTTP status code —
//! the existing routes in this crate prefer stable `200 + ok:false`
//! responses so agents can parse failures without handling transport-
//! level errors.

use axum::Json;
use axum::extract::{Path, State};
use euca_ecs::{Entity, ForkError, Query};
use euca_gameplay::assertions;
use serde::Deserialize;

use super::RichEntityData;
use crate::state::SharedWorld;

// ── Request / response helpers ──────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateForkRequest {
    pub fork_id: String,
}

#[derive(Deserialize)]
pub struct ForkStepRequest {
    #[serde(default = "default_ticks")]
    pub ticks: u64,
}

fn default_ticks() -> u64 {
    1
}

/// Body accepted by `/fork/{id}/probe`. Matches the main `/probe`
/// endpoint's fields so agents can use the same shape.
#[derive(Deserialize)]
pub struct ForkProbeRequest {
    #[serde(default)]
    pub ticks: u64,
    #[serde(default)]
    pub assertions: Option<Vec<String>>,
    #[serde(default)]
    pub snapshot_before: bool,
    #[serde(default)]
    pub snapshot_after: bool,
}

fn not_found_response(fork_id: &str) -> serde_json::Value {
    serde_json::json!({
        "ok": false,
        "error": format!("fork '{fork_id}' not found"),
    })
}

// ── Handlers ────────────────────────────────────────────────────────────────

/// POST /fork — create a new fork by deep-cloning the main world.
pub async fn create_fork(
    State(shared): State<SharedWorld>,
    Json(req): Json<CreateForkRequest>,
) -> Json<serde_json::Value> {
    let parent_tick = shared.with(|w, _| w.current_tick());
    match shared.fork(req.fork_id.clone()) {
        Ok(()) => Json(serde_json::json!({
            "ok": true,
            "fork_id": req.fork_id,
            "parent_tick": parent_tick,
        })),
        Err(ForkError::AlreadyExists(id)) => Json(serde_json::json!({
            "ok": false,
            "error": format!("fork '{id}' already exists"),
        })),
        Err(other) => Json(serde_json::json!({
            "ok": false,
            "error": other.to_string(),
        })),
    }
}

/// GET /fork/list — list all active fork ids.
pub async fn list_forks(State(shared): State<SharedWorld>) -> Json<serde_json::Value> {
    let mut forks = shared.list_forks();
    forks.sort();
    let count = forks.len();
    Json(serde_json::json!({
        "ok": true,
        "forks": forks,
        "count": count,
    }))
}

/// DELETE /fork/{id} — drop a fork.
pub async fn delete_fork(
    State(shared): State<SharedWorld>,
    Path(fork_id): Path<String>,
) -> Json<serde_json::Value> {
    if shared.delete_fork(&fork_id) {
        Json(serde_json::json!({
            "ok": true,
            "deleted": fork_id,
        }))
    } else {
        Json(not_found_response(&fork_id))
    }
}

/// POST /fork/{id}/step — advance the fork by N ticks.
pub async fn fork_step(
    State(shared): State<SharedWorld>,
    Path(fork_id): Path<String>,
    Json(req): Json<ForkStepRequest>,
) -> Json<serde_json::Value> {
    let ticks = req.ticks.min(10_000);
    let result = shared.with_fork(&fork_id, |w, schedule| {
        let start_tick = w.current_tick();
        for _ in 0..ticks {
            schedule.run(w);
            w.tick();
        }
        let end_tick = w.current_tick();
        serde_json::json!({
            "ok": true,
            "fork_id": fork_id,
            "ticks_advanced": ticks,
            "start_tick": start_tick,
            "new_tick": end_tick,
            "entity_count": w.entity_count(),
        })
    });
    match result {
        Some(json) => Json(json),
        None => Json(not_found_response(&fork_id)),
    }
}

/// POST /fork/{id}/probe — advance the fork and evaluate assertions in
/// a single atomic call. Mirrors the shape of `/probe` but operates on
/// the named fork instead of the main world.
pub async fn fork_probe(
    State(shared): State<SharedWorld>,
    Path(fork_id): Path<String>,
    Json(req): Json<ForkProbeRequest>,
) -> Json<serde_json::Value> {
    let filter_names = req.assertions;
    let snapshot_before = req.snapshot_before;
    let snapshot_after = req.snapshot_after;
    let ticks = req.ticks;

    let result = shared.with_fork(&fork_id, |w, schedule| {
        let before = if snapshot_before {
            Some(super::snapshot::capture_snapshot_from_world(
                w,
                format!("{fork_id}_before"),
            ))
        } else {
            None
        };

        let start_tick = w.current_tick();
        for _ in 0..ticks {
            schedule.run(w);
            w.tick();
        }
        let end_tick = w.current_tick();

        let report = assertions::evaluate_assertions(w);

        let results: Vec<_> = if let Some(names) = &filter_names {
            report
                .results
                .iter()
                .filter(|r| names.contains(&r.name))
                .cloned()
                .collect()
        } else {
            report.results.clone()
        };

        let passed = results.iter().filter(|r| r.passed).count();
        let failed = results.iter().filter(|r| !r.passed).count();

        let after = if snapshot_after {
            Some(super::snapshot::capture_snapshot_from_world(
                w,
                format!("{fork_id}_after"),
            ))
        } else {
            None
        };

        serde_json::json!({
            "ok": true,
            "fork_id": fork_id,
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

    match result {
        Some(json) => Json(json),
        None => Json(not_found_response(&fork_id)),
    }
}

/// GET /fork/{id}/observe — read the fork's entities. Mirrors `/observe`
/// but scoped to the named fork.
pub async fn fork_observe(
    State(shared): State<SharedWorld>,
    Path(fork_id): Path<String>,
) -> Json<serde_json::Value> {
    let result = shared.with_fork_ref(&fork_id, |w| {
        let entities: Vec<Entity> = {
            let query = Query::<Entity>::new(w);
            query.iter().collect()
        };
        let rich: Vec<RichEntityData> =
            entities.iter().map(|&e| super::read_entity_data(w, e)).collect();
        serde_json::json!({
            "ok": true,
            "fork_id": fork_id,
            "tick": w.current_tick(),
            "entity_count": w.entity_count(),
            "entities": rich,
        })
    });
    match result {
        Some(json) => Json(json),
        None => Json(not_found_response(&fork_id)),
    }
}
