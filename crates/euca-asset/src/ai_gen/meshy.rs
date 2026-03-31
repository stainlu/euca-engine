//! [Meshy](https://www.meshy.ai/) provider for AI 3D model generation.
//!
//! Supports text-to-3D and image-to-3D via the Meshy v2 API.
//! Set the `MESHY_API_KEY` environment variable to enable this provider.

use super::{AssetGenerator, GenError, GenerationId, GenerationRequest, GenerationStatus, Quality};
use serde::Serialize;
use serde_json::Value;

const BASE_URL: &str = "https://api.meshy.ai/openapi/v2";

/// AI 3D model generator backed by the Meshy API.
pub struct MeshyGenerator {
    api_key: Option<String>,
    client: reqwest::blocking::Client,
}

impl MeshyGenerator {
    /// Create a new generator, reading `MESHY_API_KEY` from the environment.
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("MESHY_API_KEY")
                .ok()
                .filter(|k| !k.is_empty()),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create a generator with an explicit API key (or `None` for unavailable).
    ///
    /// Useful for dependency injection and testing without touching environment
    /// variables.
    pub fn with_api_key(api_key: Option<String>) -> Self {
        Self {
            api_key: api_key.filter(|k| !k.is_empty()),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Return the API key or a [`GenError::NoApiKey`] error.
    fn require_key(&self) -> Result<&str, GenError> {
        self.api_key.as_deref().ok_or(GenError::NoApiKey)
    }
}

impl Default for MeshyGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a [`Quality`] tier to the Meshy API `mode` and `should_remesh` parameters.
fn quality_params(quality: Quality) -> (&'static str, bool) {
    match quality {
        Quality::Low => ("preview", false),
        Quality::Medium => ("preview", true),
        Quality::High => ("refine", true),
    }
}

/// JSON body for the Meshy text-to-3D endpoint.
#[derive(Serialize)]
struct TextTo3DBody {
    mode: &'static str,
    prompt: String,
    art_style: &'static str,
    should_remesh: bool,
}

impl AssetGenerator for MeshyGenerator {
    fn name(&self) -> &str {
        "meshy"
    }

    fn is_available(&self) -> bool {
        self.api_key.is_some()
    }

    fn generate(&self, request: &GenerationRequest) -> Result<GenerationId, GenError> {
        let api_key = self.require_key()?;

        if let Some(ref prompt) = request.prompt {
            // Text-to-3D
            let (mode, should_remesh) = quality_params(request.quality);
            let body = TextTo3DBody {
                mode,
                prompt: prompt.clone(),
                art_style: "realistic",
                should_remesh,
            };

            let resp = self
                .client
                .post(format!("{BASE_URL}/text-to-3d"))
                .bearer_auth(api_key)
                .json(&body)
                .send()
                .map_err(|e| GenError::HttpError(e.to_string()))?;

            let json: Value = resp
                .json()
                .map_err(|e| GenError::HttpError(e.to_string()))?;

            // Meshy returns `{"result": "task_id"}` on success.
            let task_id = json["result"]
                .as_str()
                .ok_or_else(|| GenError::ProviderError("missing result in response".into()))?
                .to_owned();

            Ok(GenerationId(task_id))
        } else if let Some(ref image_bytes) = request.image {
            // Image-to-3D via multipart upload.
            let part = reqwest::blocking::multipart::Part::bytes(image_bytes.clone())
                .file_name("image.png")
                .mime_str("application/octet-stream")
                .map_err(|e| GenError::HttpError(e.to_string()))?;

            let form = reqwest::blocking::multipart::Form::new().part("image", part);

            let resp = self
                .client
                .post(format!("{BASE_URL}/image-to-3d"))
                .bearer_auth(api_key)
                .multipart(form)
                .send()
                .map_err(|e| GenError::HttpError(e.to_string()))?;

            let json: Value = resp
                .json()
                .map_err(|e| GenError::HttpError(e.to_string()))?;

            let task_id = json["result"]
                .as_str()
                .ok_or_else(|| GenError::ProviderError("missing result in response".into()))?
                .to_owned();

            Ok(GenerationId(task_id))
        } else {
            Err(GenError::InvalidRequest(
                "request must contain a prompt or image".into(),
            ))
        }
    }

    fn poll(&self, id: &GenerationId) -> Result<GenerationStatus, GenError> {
        let api_key = self.require_key()?;

        let resp = self
            .client
            .get(format!("{BASE_URL}/text-to-3d/{}", id.0))
            .bearer_auth(api_key)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let json: Value = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let status = json["status"].as_str().unwrap_or("unknown");

        match status {
            "SUCCEEDED" => {
                let download_url = json["model_urls"]["glb"]
                    .as_str()
                    .ok_or_else(|| {
                        GenError::ProviderError("missing GLB URL in completed task".into())
                    })?
                    .to_owned();
                Ok(GenerationStatus::Complete { download_url })
            }
            "FAILED" | "EXPIRED" => {
                let error = json["message"]
                    .as_str()
                    .unwrap_or("generation failed")
                    .to_owned();
                Ok(GenerationStatus::Failed { error })
            }
            // "IN_PROGRESS", "PENDING", etc.
            _ => {
                let progress = json["progress"].as_f64().unwrap_or(0.0) as f32;
                Ok(GenerationStatus::Pending { progress })
            }
        }
    }

    fn download(&self, url: &str) -> Result<Vec<u8>, GenError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_api_key_is_unavailable() {
        let generator = MeshyGenerator::with_api_key(None);
        assert!(!generator.is_available());
        assert_eq!(generator.name(), "meshy");
    }

    #[test]
    fn with_api_key_is_available() {
        let generator = MeshyGenerator::with_api_key(Some("test-key-123".to_string()));
        assert!(generator.is_available());
        assert_eq!(generator.name(), "meshy");
    }

    #[test]
    fn empty_api_key_is_unavailable() {
        let generator = MeshyGenerator::with_api_key(Some(String::new()));
        assert!(!generator.is_available());
    }

    #[test]
    fn generate_without_key_returns_no_api_key() {
        let generator = MeshyGenerator::with_api_key(None);
        let req = GenerationRequest {
            prompt: Some("a chair".into()),
            ..Default::default()
        };
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::NoApiKey));
    }

    #[test]
    fn generate_empty_request_returns_invalid() {
        let generator = MeshyGenerator::with_api_key(Some("key".into()));
        let req = GenerationRequest::default();
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::InvalidRequest(_)));
    }

    #[test]
    fn quality_params_low() {
        let (mode, remesh) = quality_params(Quality::Low);
        assert_eq!(mode, "preview");
        assert!(!remesh);
    }

    #[test]
    fn quality_params_medium() {
        let (mode, remesh) = quality_params(Quality::Medium);
        assert_eq!(mode, "preview");
        assert!(remesh);
    }

    #[test]
    fn quality_params_high() {
        let (mode, remesh) = quality_params(Quality::High);
        assert_eq!(mode, "refine");
        assert!(remesh);
    }

    #[test]
    fn poll_response_succeeded() {
        let json: Value = serde_json::from_str(
            r#"{
                "id": "task-abc",
                "status": "SUCCEEDED",
                "progress": 100,
                "model_urls": { "glb": "https://cdn.meshy.ai/model.glb" }
            }"#,
        )
        .unwrap();

        let status = json["status"].as_str().unwrap();
        assert_eq!(status, "SUCCEEDED");
        let glb = json["model_urls"]["glb"].as_str().unwrap();
        assert_eq!(glb, "https://cdn.meshy.ai/model.glb");
    }

    #[test]
    fn poll_response_in_progress() {
        let json: Value = serde_json::from_str(
            r#"{
                "id": "task-abc",
                "status": "IN_PROGRESS",
                "progress": 42,
                "model_urls": null
            }"#,
        )
        .unwrap();

        let status = json["status"].as_str().unwrap();
        assert_eq!(status, "IN_PROGRESS");
        let progress = json["progress"].as_f64().unwrap();
        assert!((progress - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn poll_response_failed() {
        let json: Value = serde_json::from_str(
            r#"{
                "id": "task-xyz",
                "status": "FAILED",
                "progress": 0,
                "model_urls": null
            }"#,
        )
        .unwrap();

        let status = json["status"].as_str().unwrap();
        assert_eq!(status, "FAILED");
    }
}
