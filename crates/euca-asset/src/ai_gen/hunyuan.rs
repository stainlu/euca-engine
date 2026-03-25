//! [Hunyuan3D](https://github.com/Tencent/Hunyuan3D-2) provider for AI 3D model generation.
//!
//! Hunyuan3D is Tencent's open-source 3D generation model. Unlike other
//! providers, it can be **self-hosted** — the API client supports a
//! configurable base URL so teams can run generation on their own hardware.
//!
//! # Configuration
//!
//! | Environment Variable  | Required           | Description                            |
//! |-----------------------|--------------------|----------------------------------------|
//! | `HUNYUAN_API_KEY`     | Cloud only         | Bearer token for the cloud API         |
//! | `HUNYUAN_BASE_URL`    | No (has default)   | Override for self-hosted deployments   |
//!
//! # Self-hosted usage
//!
//! When running a local Hunyuan3D server (e.g. `http://localhost:8000`), set
//! `HUNYUAN_BASE_URL` to point at it. No API key is required in self-hosted
//! mode — the provider reports itself as available based solely on the base
//! URL being overridden.
//!
//! # Availability rules
//!
//! - API key present (any base URL) → available
//! - Custom base URL set (no API key) → available (self-hosted)
//! - Neither API key nor custom base URL → **not** available

use base64::Engine as _;
use serde::{Deserialize, Serialize};

use super::{AssetGenerator, GenError, GenerationId, GenerationRequest, GenerationStatus, Quality};

/// Default cloud-hosted API base URL (3D AI Studio).
const DEFAULT_BASE_URL: &str = "https://api.3daistudio.com";
const ENV_API_KEY: &str = "HUNYUAN_API_KEY";
const ENV_BASE_URL: &str = "HUNYUAN_BASE_URL";

// ---------------------------------------------------------------------------
// Request / response DTOs
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct GenerateBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<String>,
    #[serde(rename = "type")]
    request_type: &'static str,
    format: &'static str,
    resolution: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<String>,
}

#[derive(Deserialize)]
struct GenerateResponse {
    task_id: String,
}

#[derive(Deserialize)]
struct TaskStatusResponse {
    status: String,
    #[serde(default)]
    progress: f32,
    #[serde(default)]
    result_url: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// HunyuanGenerator
// ---------------------------------------------------------------------------

/// AI 3D model generator backed by the Hunyuan3D API.
///
/// Supports both cloud-hosted (via 3D AI Studio) and self-hosted deployments.
/// See the [module-level docs](self) for configuration details.
pub struct HunyuanGenerator {
    /// Bearer token for authenticated requests (optional for self-hosted).
    api_key: Option<String>,
    /// API root URL, without a trailing slash.
    base_url: String,
    /// Whether the base URL was explicitly overridden (i.e. self-hosted mode).
    is_self_hosted: bool,
    /// Reusable HTTP client.
    client: reqwest::blocking::Client,
}

impl HunyuanGenerator {
    /// Create a new generator, reading configuration from the environment.
    ///
    /// - `HUNYUAN_API_KEY`  — optional Bearer token
    /// - `HUNYUAN_BASE_URL` — optional base URL override (self-hosted)
    pub fn new() -> Self {
        let api_key = std::env::var(ENV_API_KEY).ok().filter(|s| !s.is_empty());

        let (base_url, is_self_hosted) = match std::env::var(ENV_BASE_URL) {
            Ok(url) if !url.is_empty() => (url.trim_end_matches('/').to_owned(), true),
            _ => (DEFAULT_BASE_URL.to_owned(), false),
        };

        Self {
            api_key,
            base_url,
            is_self_hosted,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create a generator with an explicit base URL and optional API key.
    ///
    /// A non-default base URL activates self-hosted mode, making the provider
    /// available even without an API key.
    pub fn with_base_url(base_url: &str, api_key: Option<String>) -> Self {
        let trimmed = base_url.trim_end_matches('/');
        Self {
            api_key: api_key.filter(|s| !s.is_empty()),
            is_self_hosted: trimmed != DEFAULT_BASE_URL,
            base_url: trimmed.to_owned(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Build the authorization header, if an API key is configured.
    fn auth_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(key) = &self.api_key
            && let Ok(val) = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
        {
            headers.insert(reqwest::header::AUTHORIZATION, val);
        }
        headers
    }
}

impl Default for HunyuanGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a [`Quality`] tier to the Hunyuan3D API `resolution` field.
fn resolution_param(quality: Quality) -> &'static str {
    match quality {
        Quality::Low => "low",
        Quality::Medium => "medium",
        Quality::High => "high",
    }
}

impl AssetGenerator for HunyuanGenerator {
    fn name(&self) -> &str {
        "hunyuan"
    }

    /// The provider is available when:
    /// - An API key is configured (cloud or self-hosted), **or**
    /// - A custom base URL is set (self-hosted mode, no key needed).
    fn is_available(&self) -> bool {
        self.api_key.is_some() || self.is_self_hosted
    }

    fn generate(&self, request: &GenerationRequest) -> Result<GenerationId, GenError> {
        if !self.is_available() {
            return Err(GenError::NoApiKey);
        }

        let (request_type, prompt, image) = if let Some(ref prompt) = request.prompt {
            ("text_to_3d", Some(prompt.clone()), None)
        } else if let Some(ref image_bytes) = request.image {
            let encoded = base64::engine::general_purpose::STANDARD.encode(image_bytes);
            ("image_to_3d", None, Some(encoded))
        } else {
            return Err(GenError::InvalidRequest(
                "request must contain a prompt or image".into(),
            ));
        };

        let body = GenerateBody {
            prompt,
            request_type,
            format: "glb",
            resolution: resolution_param(request.quality),
            image,
        };

        let url = format!("{}/api/generate", self.base_url);
        let resp = self
            .client
            .post(&url)
            .headers(self.auth_headers())
            .json(&body)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(GenError::ProviderError(format!(
                "Hunyuan3D submit returned {status}: {text}"
            )));
        }

        let gen_resp: GenerateResponse = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        Ok(GenerationId(gen_resp.task_id))
    }

    fn poll(&self, id: &GenerationId) -> Result<GenerationStatus, GenError> {
        let url = format!("{}/api/task/{}", self.base_url, id.0);

        let resp = self
            .client
            .get(&url)
            .headers(self.auth_headers())
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(GenError::ProviderError(format!(
                "Hunyuan3D poll returned {status}: {text}"
            )));
        }

        let parsed: TaskStatusResponse = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        match parsed.status.as_str() {
            "completed" => {
                let download_url = parsed.result_url.ok_or_else(|| {
                    GenError::ProviderError("completed task missing result_url".into())
                })?;
                Ok(GenerationStatus::Complete { download_url })
            }
            "failed" => {
                let error = parsed.error.unwrap_or_else(|| "unknown error".into());
                Ok(GenerationStatus::Failed { error })
            }
            // "processing" and any other value are treated as in-progress.
            _ => Ok(GenerationStatus::Pending {
                progress: parsed.progress,
            }),
        }
    }

    fn download(&self, url: &str) -> Result<Vec<u8>, GenError> {
        let bytes = self
            .client
            .get(url)
            .headers(self.auth_headers())
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?
            .bytes()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        Ok(bytes.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn without_env_vars_is_not_available() {
        // Use explicit constructor to avoid env-var coupling.
        let generator = HunyuanGenerator::with_base_url(DEFAULT_BASE_URL, None);
        assert!(!generator.is_available());
        assert_eq!(generator.name(), "hunyuan");
    }

    #[test]
    fn custom_base_url_makes_available_without_api_key() {
        let generator = HunyuanGenerator::with_base_url("http://localhost:8000", None);
        assert!(generator.is_available());
        assert_eq!(generator.base_url, "http://localhost:8000");
    }

    #[test]
    fn trailing_slash_stripped_from_base_url() {
        let generator = HunyuanGenerator::with_base_url("http://localhost:8000/", None);
        assert_eq!(generator.base_url, "http://localhost:8000");
    }

    #[test]
    fn cloud_url_with_key_is_available() {
        let generator = HunyuanGenerator::with_base_url(DEFAULT_BASE_URL, Some("sk-key".into()));
        assert!(generator.is_available());
    }

    #[test]
    fn empty_api_key_treated_as_none() {
        let generator = HunyuanGenerator::with_base_url("http://localhost:8000", Some("".into()));
        assert!(generator.is_available()); // self-hosted, no key needed
        assert!(generator.api_key.is_none());
    }

    #[test]
    fn self_hosted_with_key_is_available() {
        let generator =
            HunyuanGenerator::with_base_url("http://localhost:8000", Some("sk-test".into()));
        assert!(generator.is_available());
        assert_eq!(generator.api_key.as_deref(), Some("sk-test"));
        assert!(generator.is_self_hosted);
    }

    #[test]
    fn quality_mapping() {
        assert_eq!(resolution_param(Quality::Low), "low");
        assert_eq!(resolution_param(Quality::Medium), "medium");
        assert_eq!(resolution_param(Quality::High), "high");
    }

    #[test]
    fn generate_without_availability_returns_no_api_key() {
        let generator = HunyuanGenerator::with_base_url(DEFAULT_BASE_URL, None);
        let req = GenerationRequest {
            prompt: Some("a sword".into()),
            image: None,
            quality: Quality::Medium,
        };
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::NoApiKey));
    }

    #[test]
    fn generate_empty_request_returns_invalid() {
        let generator = HunyuanGenerator::with_base_url("http://localhost:8000", None);
        let req = GenerationRequest {
            prompt: None,
            image: None,
            quality: Quality::Medium,
        };
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::InvalidRequest(_)));
    }

    #[test]
    fn poll_response_completed() {
        let json: Value = serde_json::from_str(
            r#"{
                "status": "completed",
                "progress": 1.0,
                "result_url": "https://cdn.example.com/model.glb"
            }"#,
        )
        .unwrap();

        let parsed: TaskStatusResponse = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.status, "completed");
        assert_eq!(
            parsed.result_url.as_deref(),
            Some("https://cdn.example.com/model.glb")
        );
    }

    #[test]
    fn poll_response_processing() {
        let json: Value = serde_json::from_str(
            r#"{
                "status": "processing",
                "progress": 0.5
            }"#,
        )
        .unwrap();

        let parsed: TaskStatusResponse = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.status, "processing");
        assert!((parsed.progress - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn poll_response_failed() {
        let json: Value = serde_json::from_str(
            r#"{
                "status": "failed",
                "progress": 0.0,
                "error": "out of VRAM"
            }"#,
        )
        .unwrap();

        let parsed: TaskStatusResponse = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.status, "failed");
        assert_eq!(parsed.error.as_deref(), Some("out of VRAM"));
    }
}
