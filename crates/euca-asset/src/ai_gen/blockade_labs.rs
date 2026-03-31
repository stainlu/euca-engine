//! [Blockade Labs](https://www.blockadelabs.com/) provider for AI skybox
//! generation.
//!
//! Generates 360° panoramic skyboxes from text prompts. Set the
//! `BLOCKADE_API_KEY` environment variable to enable this provider.

use super::{AssetGenerator, GenError, GenerationId, GenerationRequest, GenerationStatus};
use serde_json::Value;

const BASE_URL: &str = "https://backend.blockadelabs.com/api/v1";

/// AI skybox generator backed by the Blockade Labs API.
pub struct BlockadeLabsGenerator {
    api_key: Option<String>,
    client: reqwest::blocking::Client,
}

impl BlockadeLabsGenerator {
    /// Create a new generator, reading `BLOCKADE_API_KEY` from the
    /// environment.
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("BLOCKADE_API_KEY").ok(),
            client: reqwest::blocking::Client::new(),
        }
    }

    fn require_key(&self) -> Result<&str, GenError> {
        self.api_key.as_deref().ok_or(GenError::NoApiKey)
    }
}

impl Default for BlockadeLabsGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetGenerator for BlockadeLabsGenerator {
    fn name(&self) -> &str {
        "blockade_labs"
    }

    fn is_available(&self) -> bool {
        self.api_key.is_some()
    }

    fn generate(&self, request: &GenerationRequest) -> Result<GenerationId, GenError> {
        let api_key = self.require_key()?;

        let prompt = request.prompt.as_deref().ok_or_else(|| {
            GenError::InvalidRequest("blockade_labs: prompt is required".into())
        })?;

        let body = serde_json::json!({
            "prompt": prompt,
        });

        let url = format!("{BASE_URL}/skybox");
        let resp = self
            .client
            .post(&url)
            .header("x-api-key", api_key)
            .json(&body)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let status = resp.status();
        let json: Value = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        if !status.is_success() {
            let msg = json["error"]
                .as_str()
                .or_else(|| json["message"].as_str())
                .unwrap_or("unknown error")
                .to_owned();
            return Err(GenError::ProviderError(format!(
                "blockade_labs ({status}): {msg}"
            )));
        }

        // The skybox endpoint returns an `id` for the generation request.
        let id = json["id"]
            .as_u64()
            .map(|n| n.to_string())
            .or_else(|| json["id"].as_str().map(|s| s.to_owned()))
            .ok_or_else(|| GenError::ProviderError("missing 'id' in response".into()))?;

        Ok(GenerationId(id))
    }

    fn poll(&self, id: &GenerationId) -> Result<GenerationStatus, GenError> {
        let api_key = self.require_key()?;

        let url = format!("{BASE_URL}/imagine/requests/{}", id.0);
        let resp = self
            .client
            .get(&url)
            .header("x-api-key", api_key)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let json: Value = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let request = &json["request"];
        let status = request["status"]
            .as_str()
            .unwrap_or("unknown");

        match status {
            "complete" => {
                let file_url = request["file_url"]
                    .as_str()
                    .ok_or_else(|| {
                        GenError::ProviderError("missing file_url in completed skybox".into())
                    })?
                    .to_owned();
                Ok(GenerationStatus::Complete {
                    download_url: file_url,
                })
            }
            "error" | "abort" => {
                let error = request["error_message"]
                    .as_str()
                    .unwrap_or("skybox generation failed")
                    .to_owned();
                Ok(GenerationStatus::Failed { error })
            }
            // "pending", "dispatched", "processing"
            _ => {
                let progress = request["progress"]
                    .as_f64()
                    .map(|p| p as f32 / 100.0)
                    .unwrap_or(0.0);
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
        unsafe { std::env::remove_var("BLOCKADE_API_KEY") };
        let generator =BlockadeLabsGenerator::new();
        assert!(!generator.is_available());
        assert_eq!(generator.name(), "blockade_labs");
    }

    #[test]
    fn generate_without_key_returns_no_api_key() {
        unsafe { std::env::remove_var("BLOCKADE_API_KEY") };
        let generator =BlockadeLabsGenerator::new();
        let req = GenerationRequest {
            prompt: Some("sunset over mountains".into()),
            ..Default::default()
        };
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::NoApiKey));
    }

    #[test]
    fn generate_without_prompt_returns_invalid() {
        let generator =BlockadeLabsGenerator {
            api_key: Some("fake-key".into()),
            client: reqwest::blocking::Client::new(),
        };
        let req = GenerationRequest::default();
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::InvalidRequest(_)));
    }
}
