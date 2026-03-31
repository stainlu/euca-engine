//! [World Labs](https://www.worldlabs.ai/) Marble provider for AI 3D scene
//! generation.
//!
//! Generates room/scene-scale 3D worlds as GLB meshes from text prompts. Set
//! the `WORLDLABS_API_KEY` environment variable to enable this provider.

use super::{AssetGenerator, GenError, GenerationId, GenerationRequest, GenerationStatus};
use serde_json::Value;

const BASE_URL: &str = "https://api.worldlabs.ai/marble/v1";

/// AI 3D scene generator backed by the World Labs Marble API.
///
/// Produces GLB meshes (up to 600k triangles with UV textures) from text
/// prompts. Best suited for room-scale interiors and architectural scenes.
pub struct WorldLabsGenerator {
    api_key: Option<String>,
    client: reqwest::blocking::Client,
}

impl WorldLabsGenerator {
    /// Create a new generator, reading `WORLDLABS_API_KEY` from the
    /// environment.
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("WORLDLABS_API_KEY").ok(),
            client: reqwest::blocking::Client::new(),
        }
    }

    fn require_key(&self) -> Result<&str, GenError> {
        self.api_key.as_deref().ok_or(GenError::NoApiKey)
    }
}

impl Default for WorldLabsGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetGenerator for WorldLabsGenerator {
    fn name(&self) -> &str {
        "world_labs"
    }

    fn is_available(&self) -> bool {
        self.api_key.is_some()
    }

    fn generate(&self, request: &GenerationRequest) -> Result<GenerationId, GenError> {
        let api_key = self.require_key()?;

        let prompt = request.prompt.as_deref().ok_or_else(|| {
            GenError::InvalidRequest("world_labs: prompt is required".into())
        })?;

        let body = serde_json::json!({
            "prompt": prompt,
        });

        let url = format!("{BASE_URL}/worlds:generate");
        let resp = self
            .client
            .post(&url)
            .bearer_auth(api_key)
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
                "world_labs ({status}): {msg}"
            )));
        }

        // The response contains a world ID for polling.
        let world_id = json["world_id"]
            .as_str()
            .or_else(|| json["id"].as_str())
            .ok_or_else(|| GenError::ProviderError("missing world_id in response".into()))?
            .to_owned();

        Ok(GenerationId(world_id))
    }

    fn poll(&self, id: &GenerationId) -> Result<GenerationStatus, GenError> {
        let api_key = self.require_key()?;

        let url = format!("{BASE_URL}/worlds/{}", id.0);
        let resp = self
            .client
            .get(&url)
            .bearer_auth(api_key)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let json: Value = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let status = json["status"].as_str().unwrap_or("unknown");

        match status {
            "completed" | "complete" | "succeeded" => {
                // Get the mesh export URL. Marble provides mesh export at a
                // separate endpoint or inline.
                let mesh_url = json["exports"]["mesh"]["url"]
                    .as_str()
                    .or_else(|| json["mesh_url"].as_str())
                    .or_else(|| json["output"]["mesh_url"].as_str())
                    .unwrap_or_else(|| {
                        // If no direct mesh URL, construct the export endpoint.
                        // The caller will need to trigger mesh export separately.
                        ""
                    });

                if mesh_url.is_empty() {
                    // Build the mesh export URL from the world ID.
                    let export_url =
                        format!("{BASE_URL}/worlds/{}/exports/mesh", id.0);
                    Ok(GenerationStatus::Complete {
                        download_url: export_url,
                    })
                } else {
                    Ok(GenerationStatus::Complete {
                        download_url: mesh_url.to_owned(),
                    })
                }
            }
            "failed" | "error" => {
                let error = json["error"]["message"]
                    .as_str()
                    .or_else(|| json["message"].as_str())
                    .unwrap_or("world generation failed")
                    .to_owned();
                Ok(GenerationStatus::Failed { error })
            }
            // "pending", "generating", "processing"
            _ => {
                let progress = json["progress"]
                    .as_f64()
                    .map(|p| p as f32)
                    .unwrap_or(0.0);
                Ok(GenerationStatus::Pending { progress })
            }
        }
    }

    fn download(&self, url: &str) -> Result<Vec<u8>, GenError> {
        let api_key = self.require_key()?;

        let bytes = self
            .client
            .get(url)
            .bearer_auth(api_key)
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
        unsafe { std::env::remove_var("WORLDLABS_API_KEY") };
        let generator =WorldLabsGenerator::new();
        assert!(!generator.is_available());
        assert_eq!(generator.name(), "world_labs");
    }

    #[test]
    fn generate_without_key_returns_no_api_key() {
        unsafe { std::env::remove_var("WORLDLABS_API_KEY") };
        let generator =WorldLabsGenerator::new();
        let req = GenerationRequest {
            prompt: Some("a medieval tavern interior".into()),
            ..Default::default()
        };
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::NoApiKey));
    }

    #[test]
    fn generate_without_prompt_returns_invalid() {
        let generator =WorldLabsGenerator {
            api_key: Some("fake-key".into()),
            client: reqwest::blocking::Client::new(),
        };
        let req = GenerationRequest::default();
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::InvalidRequest(_)));
    }
}
