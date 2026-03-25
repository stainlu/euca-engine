//! Assertion endpoints — define and evaluate testable expectations.

use axum::Json;
use axum::extract::{Path, State};

use euca_ecs::{Entity, Query};
use euca_gameplay::assertions::{self, Assertion, AssertCondition, Severity};

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

/// POST /assert/create — define a new assertion
///
/// Body: { "name": "hero-exists", "condition": {...}, "severity": "error" }
pub async fn assert_create(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let name = match req.get("name").and_then(|v| v.as_str()) {
        Some(n) => n.to_string(),
        None => return Json(serde_json::json!({"ok": false, "error": "Missing 'name' field"})),
    };

    // Parse condition — accept either structured JSON or shorthand string.
    let condition = if let Some(cond_obj) = req.get("condition") {
        if let Some(cond_str) = cond_obj.as_str() {
            // Shorthand: "entity-exists", "all-teams-have-spawns", etc.
            parse_condition_from_shorthand(cond_str, &req)
        } else {
            // Full JSON condition
            match serde_json::from_value::<AssertCondition>(cond_obj.clone()) {
                Ok(c) => Some(c),
                Err(e) => {
                    return Json(serde_json::json!({
                        "ok": false,
                        "error": format!("Invalid condition: {e}")
                    }));
                }
            }
        }
    } else {
        None
    };

    let condition = match condition {
        Some(c) => c,
        None => {
            return Json(serde_json::json!({
                "ok": false,
                "error": "Missing or invalid 'condition' field"
            }));
        }
    };

    let severity = req
        .get("severity")
        .and_then(|v| v.as_str())
        .map(|s| match s {
            "error" => Severity::Error,
            "warning" => Severity::Warning,
            "info" => Severity::Info,
            _ => Severity::Error,
        })
        .unwrap_or(Severity::Error);

    let entity_id = world.with(|w, _| {
        let entity = w.spawn(Assertion {
            name: name.clone(),
            condition,
            severity,
            last_result: None,
        });
        entity.index()
    });

    Json(serde_json::json!({
        "ok": true,
        "entity_id": entity_id,
        "name": name,
    }))
}

/// POST /assert/evaluate — run all assertions and return results
pub async fn assert_evaluate(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let report = world.with(|w, _| assertions::evaluate_assertions(w));

    Json(serde_json::json!({
        "ok": true,
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "results": report.results.iter().map(|r| serde_json::json!({
            "entity_id": r.entity_id,
            "name": r.name,
            "severity": r.severity,
            "passed": r.passed,
            "message": r.message,
        })).collect::<Vec<_>>(),
    }))
}

/// GET /assert/results — get last evaluation results without re-evaluating
pub async fn assert_results(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let results = world.with_world(|w| {
        let query = Query::<(Entity, &Assertion)>::new(w);
        query
            .iter()
            .map(|(e, a)| {
                serde_json::json!({
                    "entity_id": e.index(),
                    "name": a.name,
                    "severity": a.severity,
                    "last_result": a.last_result.as_ref().map(|r| serde_json::json!({
                        "passed": r.passed,
                        "message": r.message,
                        "tick": r.tick,
                    })),
                })
            })
            .collect::<Vec<_>>()
    });

    Json(serde_json::json!({
        "ok": true,
        "assertions": results,
        "count": results.len(),
    }))
}

/// GET /assert/list — list all registered assertions
pub async fn assert_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let items = world.with_world(|w| {
        let query = Query::<(Entity, &Assertion)>::new(w);
        query
            .iter()
            .map(|(e, a)| {
                serde_json::json!({
                    "entity_id": e.index(),
                    "name": a.name,
                    "severity": a.severity,
                    "has_result": a.last_result.is_some(),
                    "last_passed": a.last_result.as_ref().map(|r| r.passed),
                })
            })
            .collect::<Vec<_>>()
    });

    Json(serde_json::json!({
        "ok": true,
        "assertions": items,
        "count": items.len(),
    }))
}

/// DELETE /assert/{id} — remove an assertion
pub async fn assert_delete(
    State(world): State<SharedWorld>,
    Path(id): Path<u32>,
) -> Json<MessageResponse> {
    let ok = world.with(|w, _| {
        if let Some(entity) = find_entity(w, id) {
            if w.get::<Assertion>(entity).is_some() {
                w.despawn(entity);
                return true;
            }
        }
        false
    });

    Json(MessageResponse {
        ok,
        message: if ok {
            Some(format!("Assertion {id} deleted"))
        } else {
            Some(format!("Assertion {id} not found"))
        },
    })
}

// ── Shorthand parsing ──

fn parse_condition_from_shorthand(
    shorthand: &str,
    req: &serde_json::Value,
) -> Option<AssertCondition> {
    let filter = req
        .get("filter")
        .and_then(|v| v.as_str())
        .and_then(assertions::parse_entity_filter)
        .unwrap_or(assertions::EntityFilter::Any);

    match shorthand {
        "entity-exists" | "exists" => {
            Some(AssertCondition::EntityExists { filter })
        }
        "entity-count" | "count" => {
            let min = req.get("min").and_then(|v| v.as_u64()).map(|v| v as u32);
            let max = req.get("max").and_then(|v| v.as_u64()).map(|v| v as u32);
            Some(AssertCondition::EntityCount { filter, min, max })
        }
        "all-teams-have-spawns" | "spawns" => Some(AssertCondition::AllTeamsHaveSpawnPoints),
        "no-overlap" | "overlap" => {
            let min_distance = req
                .get("min_distance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32;
            Some(AssertCondition::NoOverlap { filter, min_distance })
        }
        "none-dead" | "alive" => Some(AssertCondition::NoneAreDead { filter }),
        "no-zero-health" => Some(AssertCondition::NoZeroHealthAlive),
        "renderable" | "all-renderable" => Some(AssertCondition::AllRenderableHaveTransform),
        "game-phase" | "phase" => {
            let phase = req.get("phase").and_then(|v| v.as_str())?.to_string();
            Some(AssertCondition::GamePhase { phase })
        }
        "entity-budget" | "budget" => {
            let max = req.get("max").and_then(|v| v.as_u64())? as u32;
            Some(AssertCondition::EntityBudget { max })
        }
        "field-check" | "field" => {
            let field = req.get("field").and_then(|v| v.as_str())?.to_string();
            let op_str = req.get("op").and_then(|v| v.as_str()).unwrap_or("greater");
            let op = match op_str {
                ">" | "greater" | "gt" => assertions::CompareOp::Greater,
                ">=" | "greater_equal" | "gte" => assertions::CompareOp::GreaterEqual,
                "<" | "less" | "lt" => assertions::CompareOp::Less,
                "<=" | "less_equal" | "lte" => assertions::CompareOp::LessEqual,
                "==" | "equal" | "eq" => assertions::CompareOp::Equal,
                "!=" | "not_equal" | "ne" => assertions::CompareOp::NotEqual,
                _ => return None,
            };
            let value = req.get("value").and_then(|v| v.as_f64())?;
            Some(AssertCondition::FieldCheck { filter, field, op, value })
        }
        _ => None,
    }
}
