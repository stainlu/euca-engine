//! [Stability AI](https://stability.ai/) provider for heightmap and texture
//! generation via the Stable Image API.
//!
//! Generates grayscale heightmap PNGs from text prompts. Set the
//! `STABILITY_API_KEY` environment variable to enable this provider.

use super::{AssetGenerator, GenError, GenerationId, GenerationRequest, GenerationStatus};
use serde_json::Value;

const BASE_URL: &str = "https://api.stability.ai/v2beta/stable-image/generate/ultra";

/// AI image generator backed by the Stability AI API.
///
/// Used primarily for generating grayscale terrain heightmaps from text
/// prompts, but can also produce texture images.
pub struct StabilityGenerator {
    api_key: Option<String>,
    client: reqwest::blocking::Client,
}

impl StabilityGenerator {
    /// Create a new generator, reading `STABILITY_API_KEY` from the
    /// environment.
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("STABILITY_API_KEY").ok(),
            client: reqwest::blocking::Client::new(),
        }
    }

    fn require_key(&self) -> Result<&str, GenError> {
        self.api_key.as_deref().ok_or(GenError::NoApiKey)
    }
}

impl Default for StabilityGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetGenerator for StabilityGenerator {
    fn name(&self) -> &str {
        "stability"
    }

    fn is_available(&self) -> bool {
        self.api_key.is_some()
    }

    fn generate(&self, request: &GenerationRequest) -> Result<GenerationId, GenError> {
        let api_key = self.require_key()?;

        let prompt = request.prompt.as_deref().ok_or_else(|| {
            GenError::InvalidRequest("stability: prompt is required".into())
        })?;

        // Stability's image generation API uses multipart form data.
        // The response can be JSON (with base64 image) or raw bytes depending
        // on the `accept` header.
        let form = reqwest::blocking::multipart::Form::new()
            .text("prompt", prompt.to_owned())
            .text("output_format", "png")
            .text("aspect_ratio", "1:1");

        let resp = self
            .client
            .post(BASE_URL)
            .bearer_auth(api_key)
            .header("Accept", "application/json")
            .multipart(form)
            .send()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        let status = resp.status();
        let json: Value = resp
            .json()
            .map_err(|e| GenError::HttpError(e.to_string()))?;

        if !status.is_success() {
            let msg = json["message"]
                .as_str()
                .or_else(|| json["name"].as_str())
                .unwrap_or("unknown error")
                .to_owned();
            return Err(GenError::ProviderError(format!("stability ({status}): {msg}")));
        }

        // The ultra endpoint returns a generation ID for async retrieval, or
        // the image inline as base64. We treat the base64 image as a
        // "completed on first call" pattern — store the base64 data in the
        // GenerationId so `poll` can extract it.
        let image_b64 = json["image"]
            .as_str()
            .ok_or_else(|| GenError::ProviderError("missing 'image' field in response".into()))?;

        // Prefix with "b64:" so poll knows this is inline data, not a task ID.
        Ok(GenerationId(format!("b64:{image_b64}")))
    }

    fn poll(&self, id: &GenerationId) -> Result<GenerationStatus, GenError> {
        // Stability returns images inline — generation is always "complete"
        // by the time we have an ID.
        if id.0.starts_with("b64:") {
            // The "download URL" is the base64 data itself (prefixed).
            Ok(GenerationStatus::Complete {
                download_url: id.0.clone(),
            })
        } else {
            Err(GenError::ProviderError(
                "unexpected generation ID format".into(),
            ))
        }
    }

    fn download(&self, url: &str) -> Result<Vec<u8>, GenError> {
        if let Some(b64_data) = url.strip_prefix("b64:") {
            // Decode the base64 image data that was returned inline.
            use base64::Engine;
            base64::engine::general_purpose::STANDARD
                .decode(b64_data)
                .map_err(|e| GenError::HttpError(format!("base64 decode error: {e}")))
        } else {
            // Regular URL download.
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
    fn new_without_api_key_is_unavailable() {
        unsafe { std::env::remove_var("STABILITY_API_KEY") };
        let generator =StabilityGenerator::new();
        assert!(!generator.is_available());
        assert_eq!(generator.name(), "stability");
    }

    #[test]
    fn generate_without_key_returns_no_api_key() {
        unsafe { std::env::remove_var("STABILITY_API_KEY") };
        let generator =StabilityGenerator::new();
        let req = GenerationRequest {
            prompt: Some("terrain heightmap".into()),
            ..Default::default()
        };
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::NoApiKey));
    }

    #[test]
    fn generate_without_prompt_returns_invalid() {
        // Manually set a fake key so we get past the key check.
        let generator =StabilityGenerator {
            api_key: Some("fake-key".into()),
            client: reqwest::blocking::Client::new(),
        };
        let req = GenerationRequest::default();
        let err = generator.generate(&req).unwrap_err();
        assert!(matches!(err, GenError::InvalidRequest(_)));
    }

    #[test]
    fn poll_b64_returns_complete() {
        let generator =StabilityGenerator::new();
        let id = GenerationId("b64:AAAA".into());
        let status = generator.poll(&id).unwrap();
        assert!(matches!(status, GenerationStatus::Complete { .. }));
    }

    #[test]
    fn download_b64_decodes_correctly() {
        use base64::Engine;
        let data = b"hello world";
        let encoded = base64::engine::general_purpose::STANDARD.encode(data);
        let url = format!("b64:{encoded}");

        let generator =StabilityGenerator::new();
        let result = generator.download(&url).unwrap();
        assert_eq!(result, data);
    }
}
