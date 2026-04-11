//! HTTP endpoints for AI-powered 3D asset generation.
//!
//! Delegates to [`euca_asset::GenerationService`] which manages providers
//! (Tripo, Meshy, Rodin, Hunyuan3D), task lifecycle, downloads, and caching.
//! The service is stored as a World resource.
//!
//! Provider methods use `reqwest::blocking::Client` internally, which cannot
//! run on tokio worker threads (it creates a nested runtime). All handlers
//! that trigger provider I/O use `tokio::task::spawn_blocking` to run the
//! blocking work on a dedicated thread pool.

use axum::Json;
use axum::extract::{Path, State};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use euca_asset::GenerationService;
use euca_asset::ai_gen::{GenerationRequest, GenerationStatus, Quality};

use crate::state::SharedWorld;

/// Shared handle to [`GenerationService`]. Stored as a world resource so
/// that it satisfies the `Clone` bound on resources; a [`World::clone`]
/// fork shares the same provider configuration and task registry.
type SharedService = Arc<Mutex<GenerationService>>;

// ── Request / Response types ──

#[derive(Clone, Deserialize)]
pub struct AssetGenerateRequest {
    pub prompt: String,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default)]
    pub quality: Option<String>,
}

fn default_provider() -> String {
    "tripo".to_string()
}

#[derive(Serialize)]
pub struct AssetStatusResponse {
    pub task_id: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct AssetListEntry {
    pub task_id: String,
    pub prompt: String,
    pub provider: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
}

#[derive(Serialize)]
pub struct ProviderInfo {
    pub name: String,
    pub available: bool,
}

#[derive(Serialize)]
pub struct ProvidersResponse {
    pub providers: Vec<ProviderInfo>,
}

// ── Helpers ──

fn parse_quality(s: Option<&str>) -> Quality {
    match s {
        Some("low") => Quality::Low,
        Some("high") => Quality::High,
        _ => Quality::Medium,
    }
}

fn status_string(status: &GenerationStatus) -> &'static str {
    match status {
        GenerationStatus::Pending { .. } => "pending",
        GenerationStatus::Complete { .. } => "complete",
        GenerationStatus::Failed { .. } => "failed",
    }
}

/// Ensure a `SharedService` resource exists in the World.
///
/// Must be called from a non-tokio thread (e.g. inside `spawn_blocking`)
/// because provider constructors create `reqwest::blocking::Client`.
fn ensure_service(w: &mut euca_ecs::World) -> SharedService {
    if let Some(existing) = w.resource::<SharedService>() {
        return existing.clone();
    }
    let service = GenerationService::new(PathBuf::from("assets/generated"));
    let shared: SharedService = Arc::new(Mutex::new(service));
    w.insert_resource(shared.clone());
    shared
}

// ── Handlers ──

/// POST /asset/generate -- start a new AI 3D generation task.
///
/// Runs on a blocking thread because `service.start()` may call
/// `provider.generate()` which uses `reqwest::blocking`.
pub async fn asset_generate(
    State(shared): State<SharedWorld>,
    Json(req): Json<AssetGenerateRequest>,
) -> Json<serde_json::Value> {
    let result = tokio::task::spawn_blocking(move || {
        let service_handle = shared.with(|w, _| ensure_service(w));
        let mut service = service_handle
            .lock()
            .expect("GenerationService mutex poisoned");

        let provider_name = req.provider.to_lowercase();
        let quality = parse_quality(req.quality.as_deref());
        let gen_request = GenerationRequest {
            prompt: Some(req.prompt.clone()),
            quality,
            ..Default::default()
        };

        service
            .start(&provider_name, &gen_request)
            .map_err(|e| e.to_string())
    })
    .await
    .unwrap_or_else(|e| Err(format!("task join error: {e}")));

    match result {
        Ok(task_id) => Json(serde_json::json!({
            "task_id": task_id,
            "status": "pending",
        })),
        Err(msg) => Json(serde_json::json!({
            "ok": false,
            "error": msg,
        })),
    }
}

/// GET /asset/status/{task_id} -- check the status of a generation task.
///
/// Polls the provider for updated status. Runs on a blocking thread because
/// `service.update()` calls `provider.poll()` which uses `reqwest::blocking`.
pub async fn asset_status(
    State(shared): State<SharedWorld>,
    Path(task_id): Path<String>,
) -> Json<serde_json::Value> {
    let result = tokio::task::spawn_blocking(move || {
        let service_handle = shared.with(|w, _| ensure_service(w));
        let mut service = service_handle
            .lock()
            .expect("GenerationService mutex poisoned");

        let status = match service.update(&task_id) {
            Ok(s) => s.clone(),
            Err(e) => return Err(e.to_string()),
        };

        let file_path = service
            .file_path(&task_id)
            .map(|p| p.to_string_lossy().into_owned());

        let resp = AssetStatusResponse {
            task_id: task_id.clone(),
            status: status_string(&status).to_string(),
            progress: match &status {
                GenerationStatus::Pending { progress } => Some(*progress),
                _ => None,
            },
            file_path,
            error: match &status {
                GenerationStatus::Failed { error } => Some(error.clone()),
                _ => None,
            },
        };
        Ok(resp)
    })
    .await
    .unwrap_or_else(|e| Err(format!("task join error: {e}")));

    match result {
        Ok(resp) => Json(serde_json::json!(resp)),
        Err(msg) => Json(serde_json::json!({
            "ok": false,
            "error": msg,
        })),
    }
}

/// GET /asset/generated -- list all generation tasks.
///
/// Read-only — no provider I/O, safe to run on tokio thread.
/// Still uses spawn_blocking for ensure_service consistency.
pub async fn asset_generated(State(shared): State<SharedWorld>) -> Json<serde_json::Value> {
    let entries = tokio::task::spawn_blocking(move || {
        let service_handle = shared.with(|w, _| ensure_service(w));
        let service = service_handle
            .lock()
            .expect("GenerationService mutex poisoned");

        let mut entries: Vec<AssetListEntry> = service
            .list_tasks()
            .into_iter()
            .map(|(task_id, prompt, status)| {
                let file_path = service
                    .file_path(task_id)
                    .map(|p| p.to_string_lossy().into_owned());
                AssetListEntry {
                    task_id: task_id.to_string(),
                    prompt: prompt.to_string(),
                    provider: String::new(),
                    status: status_string(status).to_string(),
                    file_path,
                }
            })
            .collect();

        entries.sort_by(|a, b| a.task_id.cmp(&b.task_id));
        entries
    })
    .await
    .unwrap_or_default();

    Json(serde_json::json!(entries))
}

/// GET /asset/providers -- list available AI generation providers.
pub async fn asset_providers(State(shared): State<SharedWorld>) -> Json<serde_json::Value> {
    let resp = tokio::task::spawn_blocking(move || {
        let service_handle = shared.with(|w, _| ensure_service(w));
        let service = service_handle
            .lock()
            .expect("GenerationService mutex poisoned");

        let all_providers = ["tripo", "meshy", "rodin", "hunyuan"];
        let available = service.available_providers();

        let providers: Vec<ProviderInfo> = all_providers
            .iter()
            .map(|name| ProviderInfo {
                name: name.to_string(),
                available: available.contains(name),
            })
            .collect();

        ProvidersResponse { providers }
    })
    .await
    .unwrap_or_else(|_| ProvidersResponse {
        providers: Vec::new(),
    });

    Json(serde_json::json!(resp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_deserializes_with_defaults() {
        let json = r#"{"prompt": "medieval sword"}"#;
        let req: AssetGenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "medieval sword");
        assert_eq!(req.provider, "tripo");
        assert!(req.quality.is_none());
    }

    #[test]
    fn request_deserializes_all_fields() {
        let json = r#"{"prompt": "dragon", "provider": "meshy", "quality": "high"}"#;
        let req: AssetGenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "dragon");
        assert_eq!(req.provider, "meshy");
        assert_eq!(req.quality.as_deref(), Some("high"));
    }

    #[test]
    fn generate_response_json_format() {
        let json = serde_json::json!({
            "task_id": "gen_1",
            "status": "pending",
        });
        assert_eq!(json["task_id"], "gen_1");
        assert_eq!(json["status"], "pending");
    }

    #[test]
    fn status_response_serializes_pending() {
        let resp = AssetStatusResponse {
            task_id: "gen_1".to_string(),
            status: "pending".to_string(),
            progress: Some(0.5),
            file_path: None,
            error: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["task_id"], "gen_1");
        assert_eq!(json["progress"], 0.5);
        assert!(json.get("file_path").is_none());
        assert!(json.get("error").is_none());
    }

    #[test]
    fn status_response_serializes_complete() {
        let resp = AssetStatusResponse {
            task_id: "gen_1".to_string(),
            status: "complete".to_string(),
            progress: None,
            file_path: Some("assets/generated/gen_1.glb".to_string()),
            error: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["status"], "complete");
        assert_eq!(json["file_path"], "assets/generated/gen_1.glb");
        assert!(json.get("progress").is_none());
    }

    #[test]
    fn status_response_serializes_failed() {
        let resp = AssetStatusResponse {
            task_id: "gen_2".to_string(),
            status: "failed".to_string(),
            progress: None,
            file_path: None,
            error: Some("Provider error: rate limit".to_string()),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["status"], "failed");
        assert_eq!(json["error"], "Provider error: rate limit");
    }

    #[test]
    fn providers_response_serializes() {
        let resp = ProvidersResponse {
            providers: vec![
                ProviderInfo {
                    name: "tripo".to_string(),
                    available: true,
                },
                ProviderInfo {
                    name: "meshy".to_string(),
                    available: false,
                },
            ],
        };
        let json = serde_json::to_value(&resp).unwrap();
        let providers = json["providers"].as_array().unwrap();
        assert_eq!(providers.len(), 2);
        assert_eq!(providers[0]["name"], "tripo");
        assert!(providers[0]["available"].as_bool().unwrap());
        assert_eq!(providers[1]["name"], "meshy");
        assert!(!providers[1]["available"].as_bool().unwrap());
    }

    #[test]
    fn list_entry_serializes() {
        let entry = AssetListEntry {
            task_id: "gen_1".to_string(),
            prompt: "medieval sword".to_string(),
            provider: "tripo".to_string(),
            status: "complete".to_string(),
            file_path: Some("assets/generated/gen_1.glb".to_string()),
        };
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["task_id"], "gen_1");
        assert_eq!(json["prompt"], "medieval sword");
        assert_eq!(json["provider"], "tripo");
        assert_eq!(json["file_path"], "assets/generated/gen_1.glb");
    }

    #[test]
    fn parse_quality_values() {
        assert!(matches!(parse_quality(None), Quality::Medium));
        assert!(matches!(parse_quality(Some("low")), Quality::Low));
        assert!(matches!(parse_quality(Some("medium")), Quality::Medium));
        assert!(matches!(parse_quality(Some("high")), Quality::High));
        assert!(matches!(parse_quality(Some("unknown")), Quality::Medium));
    }

    #[test]
    fn status_string_values() {
        assert_eq!(
            status_string(&GenerationStatus::Pending { progress: 0.0 }),
            "pending"
        );
        assert_eq!(
            status_string(&GenerationStatus::Complete {
                download_url: String::new()
            }),
            "complete"
        );
        assert_eq!(
            status_string(&GenerationStatus::Failed {
                error: "err".into()
            }),
            "failed"
        );
    }
}
