//! AI-powered 3D asset generation.
//!
//! This module defines the [`AssetGenerator`] trait that all AI generation
//! providers implement, along with shared request/response types. Concrete
//! providers live in sub-modules (e.g. [`tripo`]).
//!
//! All provider methods are **synchronous and blocking** — callers are
//! responsible for scheduling them on a background thread or task pool.

pub mod tripo;
// Other providers will add their modules here.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a generation task.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct GenerationId(pub String);

/// What to generate.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    /// Text prompt describing the 3D model (text-to-3D).
    pub prompt: Option<String>,
    /// Image bytes for image-to-3D generation.
    pub image: Option<Vec<u8>>,
    /// Quality level.
    pub quality: Quality,
}

/// Generation quality tier.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum Quality {
    Low,
    #[default]
    Medium,
    High,
}

/// Status of an ongoing generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GenerationStatus {
    Pending { progress: f32 },
    Complete { download_url: String },
    Failed { error: String },
}

/// Errors from generation providers.
#[derive(Clone, Debug)]
pub enum GenError {
    /// Missing API key.
    NoApiKey,
    /// HTTP request failed.
    HttpError(String),
    /// Provider returned an error.
    ProviderError(String),
    /// Invalid request.
    InvalidRequest(String),
}

impl fmt::Display for GenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenError::NoApiKey => write!(f, "API key not configured"),
            GenError::HttpError(e) => write!(f, "HTTP error: {e}"),
            GenError::ProviderError(e) => write!(f, "Provider error: {e}"),
            GenError::InvalidRequest(e) => write!(f, "Invalid request: {e}"),
        }
    }
}

impl std::error::Error for GenError {}

/// Trait that all AI 3D generation providers implement.
///
/// All methods are synchronous and blocking — they use `reqwest::blocking`
/// internally. The caller (e.g. a future `GenerationService`) manages async
/// scheduling.
pub trait AssetGenerator: Send + Sync {
    /// Provider name (e.g., `"tripo"`, `"meshy"`).
    fn name(&self) -> &str;

    /// Whether this provider has a configured API key.
    fn is_available(&self) -> bool;

    /// Start a generation task. Returns a provider-specific task ID.
    fn generate(&self, request: &GenerationRequest) -> Result<GenerationId, GenError>;

    /// Check the status of a generation task.
    fn poll(&self, id: &GenerationId) -> Result<GenerationStatus, GenError>;

    /// Download the generated GLB model bytes.
    fn download(&self, url: &str) -> Result<Vec<u8>, GenError>;
}
