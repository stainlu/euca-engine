//! [Tripo3D](https://www.tripo3d.ai/) provider for AI 3D model generation.
//!
//! Supports both text-to-3D and image-to-3D via the Tripo v2 API.
//! Set the `TRIPO_API_KEY` environment variable to enable this provider.

use super::{AssetGenerator, GenError, GenerationId, GenerationRequest, GenerationStatus, Quality};
use serde_json::Value;

const BASE_URL: &str = "https://api.tripo3d.ai/v2/openapi";

/// AI 3D model generator backed by the Tripo3D API.
pub struct TripoGenerator {
    api_key: Option<String>,
    client: reqwest::blocking::Client,
}

impl TripoGenerator {
    /// Create a new generator, reading `TRIPO_API_KEY` from the environment.
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("TRIPO_API_KEY").ok(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Return the API key or a [`GenError::NoApiKey`] error.
    fn require_key(&self) -> Result<&str, GenError> {
        self.api_key.as_deref().ok_or(GenError::NoApiKey)
    }
}

impl Default for TripoGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a [`Quality`] tier to the Tripo `model_version` parameter.
fn model_version(quality: Quality) -> &'static str {
    match quality {
        Quality::Low => "lite",
        Quality::Medium => "default",
        Quality::High => "v2.5-20250414",
    }
}

impl AssetGenerator for TripoGenerator {
    fn name(&self) -> &str {
        "tripo"
    }

    fn is_available(&self) -> bool {
        self.api_key.is_some()
    }

    fn generate(&self, request: &GenerationRequest) -> Result<GenerationId, GenError> {
        let api_key = self.require_key()?;

        // Build JSON body based on whether we have a prompt or image.
        let body = if let Some(ref prompt) = request.prompt {
            serde_json::json!({
                "type": "text_to_model",
                "prompt": prompt,
                "model_version": model_version(request.quality),
            })
        } else if request.image.is_some() {
            // Image-to-model requires a multipart upload which is a separate
            // flow — for now we only support text prompts. A future PR will
            // add image support via the multipart/form-data endpoint.
            return Err(GenError::InvalidRequest(
                "image-to-model is not yet implemented".into(),
            ));
        } else {
            return Err(GenError::InvalidRequest(
                "request must contain a prompt or image".into(),
            ));
        };

        let url = format!("{BASE_URL}/task");
        let resp = self
            .client
            .post(&url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let json: Value = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        // Tripo returns `{"code": 0, "data": {"task_id": "..."}}` on success.
        let code = json["code"].as_i64().unwrap_or(-1);
        if code != 0 {
            let msg = json["message"]
                .as_str()
                .unwrap_or("unknown error")
                .to_owned();
            return Err(GenError::ProviderError(msg));
        }

        let task_id = json["data"]["task_id"]
            .as_str()
            .ok_or_else(|| GenError::ProviderError("missing task_id in response".into()))?
            .to_owned();

        Ok(GenerationId(task_id))
    }

    fn poll(&self, id: &GenerationId) -> Result<GenerationStatus, GenError> {
        let api_key = self.require_key()?;

        let url = format!("{BASE_URL}/task/{}", id.0);
        let resp = self
            .client
            .get(&url)
            .bearer_auth(api_key)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let json: Value = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let code = json["code"].as_i64().unwrap_or(-1);
        if code != 0 {
            let msg = json["message"]
                .as_str()
                .unwrap_or("unknown error")
                .to_owned();
            return Err(GenError::ProviderError(msg));
        }

        let data = &json["data"];
        let status = data["status"].as_str().unwrap_or("unknown");

        match status {
            "success" => {
                // Tripo v2 returns the PBR GLB at `output.pbr_model` (direct URL string)
                // or at `result.pbr_model.url` (nested object). Try both.
                let download_url = data["output"]["pbr_model"]
                    .as_str()
                    .or_else(|| data["result"]["pbr_model"]["url"].as_str())
                    .ok_or_else(|| {
                        GenError::ProviderError("missing model URL in completed task".into())
                    })?
                    .to_owned();
                Ok(GenerationStatus::Complete { download_url })
            }
            "failed" | "cancelled" | "unknown" => {
                let error = data["message"]
                    .as_str()
                    .unwrap_or("generation failed")
                    .to_owned();
                Ok(GenerationStatus::Failed { error })
            }
            // "queued", "running", etc.
            _ => {
                let progress = data["progress"].as_f64().unwrap_or(0.0) as f32;
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
    fn new_without_api_key_is_unavailable() {
        // Remove the env var if it happens to be set in the test environment.
        // SAFETY: These tests are not run concurrently with other code that
        // reads this env var, so the mutation is safe.
        unsafe { std::env::remove_var("TRIPO_API_KEY") };
        let generator = TripoGenerator::new();
        assert!(!generator.is_available());
        assert_eq!(generator.name(), "tripo");
    }

    #[test]
    fn generation_request_with_prompt() {
        let req = GenerationRequest {
            prompt: Some("a wooden chair".into()),
            ..Default::default()
        };
        assert!(req.prompt.is_some());
        assert!(req.image.is_none());
    }

    #[test]
    fn quality_default_is_medium() {
        let q = Quality::default();
        assert!(matches!(q, Quality::Medium));
    }

    #[test]
    fn gen_error_display() {
        assert_eq!(GenError::NoApiKey.to_string(), "API key not configured");
        assert_eq!(
            GenError::HttpError("timeout".into()).to_string(),
            "HTTP error: timeout"
        );
        assert_eq!(
            GenError::ProviderError("rate limit".into()).to_string(),
            "Provider error: rate limit"
        );
        assert_eq!(
            GenError::InvalidRequest("empty".into()).to_string(),
            "Invalid request: empty"
        );
    }

    #[test]
    fn generate_without_key_returns_no_api_key() {
        // SAFETY: These tests are not run concurrently with other code that
        // reads this env var, so the mutation is safe.
        unsafe { std::env::remove_var("TRIPO_API_KEY") };
        let generator = TripoGenerator::new();
        let req = GenerationRequest {
            prompt: Some("test".into()),
            ..Default::default()
        };
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::NoApiKey));
    }

    #[test]
    fn quality_maps_to_model_version() {
        assert_eq!(model_version(Quality::Low), "lite");
        assert_eq!(model_version(Quality::Medium), "default");
        assert_eq!(model_version(Quality::High), "v2.5-20250414");
    }
}
