//! [Rodin (Hyper3D)](https://hyperhuman.deemos.com/) provider for AI 3D model generation.
//!
//! Supports both text-to-3D and image-to-3D via the Hyper3D Rodin API v2.
//! Set the `RODIN_API_KEY` environment variable to enable this provider.
//!
//! The Rodin workflow differs from other providers: it returns a
//! `subscription_key` for polling status and uses the task `uuid` for
//! downloading. Both are packed into the [`GenerationId`] as
//! `"{uuid}:{subscription_key}"`.

use super::{AssetGenerator, GenError, GenerationId, GenerationRequest, GenerationStatus, Quality};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://hyperhuman.deemos.com/api/v2";
const ENV_API_KEY: &str = "RODIN_API_KEY";

/// AI 3D model generator backed by the Hyper3D Rodin API.
pub struct RodinGenerator {
    api_key: Option<String>,
    client: reqwest::blocking::Client,
}

// ── API request / response types ──────────────────────────────────────────────

#[derive(Serialize)]
struct GenerateBody {
    prompt: String,
    condition_mode: &'static str,
    quality: &'static str,
    geometry_file_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct GenerateResponse {
    uuid: String,
    #[serde(default)]
    subscription_key: Option<String>,
}

#[derive(Serialize)]
struct StatusBody {
    subscription_key: String,
}

#[derive(Deserialize)]
struct StatusResponse {
    jobs: std::collections::HashMap<String, JobStatus>,
}

#[derive(Deserialize)]
struct JobStatus {
    status: String,
    #[serde(default)]
    progress: f32,
}

#[derive(Serialize)]
struct DownloadBody {
    task_uuid: String,
}

#[derive(Deserialize)]
struct DownloadResponse {
    #[serde(default)]
    list: Vec<DownloadItem>,
}

#[derive(Deserialize)]
struct DownloadItem {
    url: String,
}

// ── Implementation ────────────────────────────────────────────────────────────

impl RodinGenerator {
    /// Create a new generator, reading `RODIN_API_KEY` from the environment.
    pub fn new() -> Self {
        Self {
            api_key: std::env::var(ENV_API_KEY).ok().filter(|k| !k.is_empty()),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create a generator with an explicit API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        let key = api_key.into();
        Self {
            api_key: if key.is_empty() { None } else { Some(key) },
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Return the API key or [`GenError::NoApiKey`].
    fn require_key(&self) -> Result<&str, GenError> {
        self.api_key.as_deref().ok_or(GenError::NoApiKey)
    }

    /// Pack `uuid` and `subscription_key` into a single [`GenerationId`].
    fn pack_id(uuid: &str, subscription_key: &str) -> GenerationId {
        GenerationId(format!("{uuid}:{subscription_key}"))
    }

    /// Unpack a [`GenerationId`] into `(uuid, subscription_key)`.
    fn unpack_id(id: &GenerationId) -> Result<(&str, &str), GenError> {
        id.0.split_once(':')
            .ok_or_else(|| GenError::ProviderError(format!("invalid Rodin task ID: {}", id.0)))
    }
}

impl Default for RodinGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a [`Quality`] tier to the Rodin API quality string.
fn quality_param(quality: Quality) -> &'static str {
    match quality {
        Quality::Low => "sketch",
        Quality::Medium => "regular",
        Quality::High => "detail",
    }
}

impl AssetGenerator for RodinGenerator {
    fn name(&self) -> &str {
        "rodin"
    }

    fn is_available(&self) -> bool {
        self.api_key.is_some()
    }

    fn generate(&self, request: &GenerationRequest) -> Result<GenerationId, GenError> {
        let api_key = self.require_key()?;

        let prompt = request
            .prompt
            .as_deref()
            .ok_or_else(|| GenError::InvalidRequest("prompt is required for Rodin".into()))?;

        let images = request.image.as_ref().map(|bytes| {
            let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
            vec![format!("data:image/png;base64,{encoded}")]
        });

        let body = GenerateBody {
            prompt: prompt.to_owned(),
            condition_mode: "concat",
            quality: quality_param(request.quality),
            geometry_file_format: "glb",
            images,
        };

        let resp = self
            .client
            .post(format!("{BASE_URL}/rodin"))
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(GenError::ProviderError(format!(
                "Rodin submit returned {status}: {text}"
            )));
        }

        let gen_resp: GenerateResponse = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let subscription_key = gen_resp
            .subscription_key
            .ok_or_else(|| GenError::ProviderError("response missing subscription_key".into()))?;

        Ok(Self::pack_id(&gen_resp.uuid, &subscription_key))
    }

    fn poll(&self, id: &GenerationId) -> Result<GenerationStatus, GenError> {
        let api_key = self.require_key()?;
        let (uuid, subscription_key) = Self::unpack_id(id)?;

        let body = StatusBody {
            subscription_key: subscription_key.to_owned(),
        };

        let resp = self
            .client
            .post(format!("{BASE_URL}/status"))
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(GenError::ProviderError(format!(
                "Rodin status returned {status}: {text}"
            )));
        }

        let status_resp: StatusResponse = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        // Find the job matching our UUID, falling back to the first job.
        let job = status_resp
            .jobs
            .get(uuid)
            .or_else(|| status_resp.jobs.values().next())
            .ok_or_else(|| GenError::ProviderError("status response contains no jobs".into()))?;

        match job.status.as_str() {
            "Done" => {
                // Rodin requires a separate download call to get the URL.
                // We construct the download URL from the task UUID.
                let download_url = format!("{BASE_URL}/download?task_uuid={uuid}");
                Ok(GenerationStatus::Complete { download_url })
            }
            "Failed" => Ok(GenerationStatus::Failed {
                error: format!("generation failed for task {uuid}"),
            }),
            _ => Ok(GenerationStatus::Pending {
                progress: job.progress.min(99.0) / 100.0,
            }),
        }
    }

    fn download(&self, url: &str) -> Result<Vec<u8>, GenError> {
        let api_key = self.require_key()?;

        // The download URL we constructed in `poll` encodes the task UUID as a
        // query parameter. We need to POST to the download endpoint with the
        // task_uuid in the body rather than using the URL directly.
        if let Some(task_uuid) = url.strip_prefix(&format!("{BASE_URL}/download?task_uuid=")) {
            let body = DownloadBody {
                task_uuid: task_uuid.to_owned(),
            };

            let resp = self
                .client
                .post(format!("{BASE_URL}/download"))
                .bearer_auth(api_key)
                .json(&body)
                .send()
                .map_err(|e| GenError::HttpError(e.to_string()))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().unwrap_or_default();
                return Err(GenError::ProviderError(format!(
                    "Rodin download returned {status}: {text}"
                )));
            }

            let dl: DownloadResponse = resp
                .json()
                .map_err(|e| GenError::HttpError(e.to_string()))?;

            let file_url = dl.list.first().map(|item| &item.url).ok_or_else(|| {
                GenError::ProviderError("download response contains no files".into())
            })?;

            // Fetch the actual GLB file from the CDN URL.
            let glb_bytes = self
                .client
                .get(file_url)
                .send()
                .map_err(|e| GenError::HttpError(e.to_string()))?
                .bytes()
                .map_err(|e| GenError::HttpError(e.to_string()))?;

            Ok(glb_bytes.to_vec())
        } else {
            // Direct URL download (e.g. a CDN link already resolved).
            let bytes = self
                .client
                .get(url)
                .send()
                .map_err(|e| GenError::HttpError(e.to_string()))?
                .bytes()
                .map_err(|e| GenError::HttpError(e.to_string()))?;

            Ok(bytes.to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_api_key_is_unavailable() {
        let generator = RodinGenerator::with_api_key("");
        assert!(!generator.is_available());
        assert_eq!(generator.name(), "rodin");
    }

    #[test]
    fn with_api_key_is_available() {
        let generator = RodinGenerator::with_api_key("test-key-12345");
        assert!(generator.is_available());
    }

    #[test]
    fn quality_mapping() {
        assert_eq!(quality_param(Quality::Low), "sketch");
        assert_eq!(quality_param(Quality::Medium), "regular");
        assert_eq!(quality_param(Quality::High), "detail");
    }

    #[test]
    fn id_pack_unpack_roundtrip() {
        let id = RodinGenerator::pack_id("abc-123", "sub-key-456");
        assert_eq!(id.0, "abc-123:sub-key-456");

        let (uuid, sub_key) = RodinGenerator::unpack_id(&id).unwrap();
        assert_eq!(uuid, "abc-123");
        assert_eq!(sub_key, "sub-key-456");
    }

    #[test]
    fn unpack_invalid_id() {
        let bad_id = GenerationId("no-colon-here".into());
        assert!(RodinGenerator::unpack_id(&bad_id).is_err());
    }

    #[test]
    fn generate_without_key_returns_no_api_key() {
        let generator = RodinGenerator::with_api_key("");
        let req = GenerationRequest {
            prompt: Some("a sword".into()),
            ..Default::default()
        };
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::NoApiKey));
    }

    #[test]
    fn generate_without_prompt_returns_invalid_request() {
        let generator = RodinGenerator::with_api_key("test-key");
        let req = GenerationRequest::default();
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::InvalidRequest(_)));
    }
}
