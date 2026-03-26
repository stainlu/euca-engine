//! Game manifest endpoints — sprint contracts in engine-native form.
//!
//! A manifest describes WHAT the game IS: its genre, features, and completion
//! criteria. Each feature links to assertion names that define "done."
//! When an agent reads the manifest, it knows: this is a MOBA, these features
//! exist, these assertions define done, these pass, these fail.

use axum::Json;
use axum::extract::State;
use euca_ecs::{Entity, Query, World};
use serde::{Deserialize, Serialize};

use crate::state::SharedWorld;

/// Status of a game feature.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeatureStatus {
    Planned,
    InProgress,
    Complete,
    Verified,
}

/// A single game feature with linked assertions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Feature {
    pub name: String,
    pub description: String,
    pub status: FeatureStatus,
    /// Names of assertions that define "done" for this feature.
    pub assertions: Vec<String>,
}

/// The game manifest — a structured description of what the game is and
/// what "complete" means for each feature.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GameManifest {
    pub name: String,
    pub genre: String,
    pub description: String,
    pub features: Vec<Feature>,
}

/// Evaluate a manifest against the current assertion state.
fn evaluate_manifest(manifest: &GameManifest, w: &World) -> serde_json::Value {
    let query = Query::<(Entity, &euca_gameplay::assertions::Assertion)>::new(w);
    let assertion_map: std::collections::HashMap<String, bool> = query
        .iter()
        .filter_map(|(_, a)| a.last_result.as_ref().map(|r| (a.name.clone(), r.passed)))
        .collect();

    let features: Vec<serde_json::Value> = manifest
        .features
        .iter()
        .map(|f| {
            let assertion_results: Vec<serde_json::Value> = f
                .assertions
                .iter()
                .map(|name| {
                    let passed = assertion_map.get(name);
                    serde_json::json!({
                        "name": name,
                        "status": match passed {
                            Some(true) => "pass",
                            Some(false) => "fail",
                            None => "not_evaluated",
                        },
                    })
                })
                .collect();

            let all_pass = f
                .assertions
                .iter()
                .all(|name| assertion_map.get(name).copied().unwrap_or(false));
            let any_evaluated = f
                .assertions
                .iter()
                .any(|name| assertion_map.contains_key(name));

            let effective_status = if all_pass && any_evaluated && !f.assertions.is_empty() {
                "verified"
            } else {
                match f.status {
                    FeatureStatus::Planned => "planned",
                    FeatureStatus::InProgress => "in_progress",
                    FeatureStatus::Complete => "complete",
                    FeatureStatus::Verified => "verified",
                }
            };

            serde_json::json!({
                "name": f.name,
                "description": f.description,
                "status": effective_status,
                "assertions": assertion_results,
                "all_assertions_pass": all_pass,
            })
        })
        .collect();

    let total = manifest.features.len();
    let verified = features
        .iter()
        .filter(|f| f["all_assertions_pass"].as_bool().unwrap_or(false))
        .count();

    serde_json::json!({
        "name": manifest.name,
        "genre": manifest.genre,
        "description": manifest.description,
        "features": features,
        "total_features": total,
        "verified_features": verified,
        "completion_pct": if total > 0 { (verified as f64 / total as f64 * 100.0) as u32 } else { 0 },
    })
}

// ── HTTP Handlers ──

/// POST /manifest — set or update the game manifest
pub async fn manifest_set(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let manifest: GameManifest = match serde_json::from_value(req) {
        Ok(m) => m,
        Err(e) => {
            return Json(serde_json::json!({
                "ok": false,
                "error": format!("Invalid manifest: {e}"),
            }));
        }
    };

    let feature_count = manifest.features.len();
    world.with(|w, _| {
        w.insert_resource(manifest);
    });

    Json(serde_json::json!({
        "ok": true,
        "message": format!("Manifest set with {feature_count} features"),
    }))
}

/// GET /manifest — read the manifest with assertion evaluation inlined
pub async fn manifest_get(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let result = world.with_world(|w| match w.resource::<GameManifest>() {
        Some(manifest) => {
            let evaluated = evaluate_manifest(manifest, w);
            serde_json::json!({
                "ok": true,
                "manifest": evaluated,
            })
        }
        None => {
            serde_json::json!({
                "ok": false,
                "error": "No manifest set. Use POST /manifest first.",
            })
        }
    });

    Json(result)
}

/// POST /manifest/feature — update a single feature's status
pub async fn manifest_feature_update(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let name = match req.get("name").and_then(|v| v.as_str()) {
        Some(n) => n.to_string(),
        None => {
            return Json(serde_json::json!({
                "ok": false,
                "error": "Missing 'name' field",
            }));
        }
    };

    let status = req
        .get("status")
        .and_then(|v| v.as_str())
        .and_then(|s| match s {
            "planned" => Some(FeatureStatus::Planned),
            "in_progress" => Some(FeatureStatus::InProgress),
            "complete" => Some(FeatureStatus::Complete),
            "verified" => Some(FeatureStatus::Verified),
            _ => None,
        });

    let result = world.with(|w, _| {
        if let Some(manifest) = w.resource_mut::<GameManifest>() {
            if let Some(feature) = manifest.features.iter_mut().find(|f| f.name == name) {
                if let Some(s) = status {
                    feature.status = s;
                }
                // Add new assertions if provided
                if let Some(assertions) = req.get("assertions").and_then(|v| v.as_array()) {
                    for a in assertions {
                        if let Some(name) = a.as_str() {
                            if !feature.assertions.contains(&name.to_string()) {
                                feature.assertions.push(name.to_string());
                            }
                        }
                    }
                }
                serde_json::json!({"ok": true, "message": format!("Feature '{name}' updated")})
            } else {
                serde_json::json!({"ok": false, "error": format!("Feature '{name}' not found in manifest")})
            }
        } else {
            serde_json::json!({"ok": false, "error": "No manifest set"})
        }
    });

    Json(result)
}
