//! [Scenario.gg](https://www.scenario.com/) provider for AI PBR texture
//! generation.
//!
//! Generates seamless tileable PBR texture sets (albedo, normal, roughness, AO)
//! from text prompts. Set the `SCENARIO_API_KEY` and `SCENARIO_API_SECRET`
//! environment variables to enable this provider.
//!
//! The Scenario API uses a 2-phase workflow:
//! 1. Generate albedo image via `txt2img-texture`
//! 2. Generate PBR maps from the albedo via `texture` endpoint
//!
//! For simplicity, this provider handles phase 1 only (albedo). PBR map
//! generation can be triggered separately or added as a follow-up step.

use super::{AssetGenerator, GenError, GenerationId, GenerationRequest, GenerationStatus};
use serde_json::Value;

const BASE_URL: &str = "https://api.cloud.scenario.com/v1";

/// AI texture generator backed by the Scenario.gg API.
///
/// Generates seamless tileable textures from text prompts. Best suited for
/// terrain textures, material surfaces, and game asset textures.
pub struct ScenarioGenerator {
    api_key: Option<String>,
    api_secret: Option<String>,
    client: reqwest::blocking::Client,
}

impl ScenarioGenerator {
    /// Create a new generator, reading `SCENARIO_API_KEY` and
    /// `SCENARIO_API_SECRET` from the environment.
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("SCENARIO_API_KEY").ok(),
            api_secret: std::env::var("SCENARIO_API_SECRET").ok(),
            client: reqwest::blocking::Client::new(),
        }
    }

    fn require_auth(&self) -> Result<String, GenError> {
        let key = self.api_key.as_deref().ok_or(GenError::NoApiKey)?;
        let secret = self.api_secret.as_deref().ok_or(GenError::NoApiKey)?;
        // Basic auth: base64(key:secret)
        use base64::Engine;
        let encoded =
            base64::engine::general_purpose::STANDARD.encode(format!("{key}:{secret}"));
        Ok(format!("Basic {encoded}"))
    }
}

impl Default for ScenarioGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetGenerator for ScenarioGenerator {
    fn name(&self) -> &str {
        "scenario"
    }

    fn is_available(&self) -> bool {
        self.api_key.is_some() && self.api_secret.is_some()
    }

    fn generate(&self, request: &GenerationRequest) -> Result<GenerationId, GenError> {
        let auth = self.require_auth()?;

        let prompt = request.prompt.as_deref().ok_or_else(|| {
            GenError::InvalidRequest("scenario: prompt is required".into())
        })?;

        let (width, height) = request.dimensions.unwrap_or((1024, 1024));

        let body = serde_json::json!({
            "prompt": prompt,
            "width": width,
            "height": height,
            "imageType": "texture",
            "numOutputs": 1,
            "numInferenceSteps": 30,
        });

        let url = format!("{BASE_URL}/generate/txt2img-texture");
        let resp = self
            .client
            .post(&url)
            .header("Authorization", &auth)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let status = resp.status();
        let json: Value = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        if !status.is_success() {
            let msg = json["error"]["message"]
                .as_str()
                .or_else(|| json["message"].as_str())
                .unwrap_or("unknown error")
                .to_owned();
            return Err(GenError::ProviderError(format!(
                "scenario ({status}): {msg}"
            )));
        }

        let job_id = json["jobId"]
            .as_str()
            .or_else(|| json["job"]["jobId"].as_str())
            .ok_or_else(|| GenError::ProviderError("missing jobId in response".into()))?
            .to_owned();

        Ok(GenerationId(job_id))
    }

    fn poll(&self, id: &GenerationId) -> Result<GenerationStatus, GenError> {
        let auth = self.require_auth()?;

        let url = format!("{BASE_URL}/jobs/{}", id.0);
        let resp = self
            .client
            .get(&url)
            .header("Authorization", &auth)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let json: Value = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let job = if json.get("job").is_some() {
            &json["job"]
        } else {
            &json
        };

        let status = job["status"].as_str().unwrap_or("unknown");

        match status {
            "success" | "completed" => {
                // Extract the generated image URL from the assets array.
                let download_url = job["assets"]
                    .as_array()
                    .and_then(|a| a.first())
                    .and_then(|a| a["url"].as_str())
                    .or_else(|| job["url"].as_str())
                    .ok_or_else(|| {
                        GenError::ProviderError("no asset URL in completed job".into())
                    })?
                    .to_owned();
                Ok(GenerationStatus::Complete { download_url })
            }
            "failed" | "error" => {
                let error = job["error"]
                    .as_str()
                    .unwrap_or("texture generation failed")
                    .to_owned();
                Ok(GenerationStatus::Failed { error })
            }
            _ => {
                let progress = job["progress"]
                    .as_f64()
                    .map(|p| p as f32)
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
    fn new_without_keys_is_unavailable() {
        unsafe {
            std::env::remove_var("SCENARIO_API_KEY");
            std::env::remove_var("SCENARIO_API_SECRET");
        }
        let generator = ScenarioGenerator::new();
        assert!(!generator.is_available());
        assert_eq!(generator.name(), "scenario");
    }

    #[test]
    fn generate_without_key_returns_no_api_key() {
        unsafe {
            std::env::remove_var("SCENARIO_API_KEY");
            std::env::remove_var("SCENARIO_API_SECRET");
        }
        let generator = ScenarioGenerator::new();
        let req = GenerationRequest {
            prompt: Some("seamless grass texture".into()),
            ..Default::default()
        };
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::NoApiKey));
    }

    #[test]
    fn generate_without_prompt_returns_invalid() {
        let generator = ScenarioGenerator {
            api_key: Some("key".into()),
            api_secret: Some("secret".into()),
            client: reqwest::blocking::Client::new(),
        };
        let req = GenerationRequest::default();
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::InvalidRequest(_)));
    }
}
