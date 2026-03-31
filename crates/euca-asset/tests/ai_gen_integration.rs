//! Integration tests for AI asset generation.
//!
//! Uses a [`MockGenerator`] that simulates the generate-poll-download lifecycle
//! without hitting any real API. This validates the full [`GenerationService`]
//! flow: task creation, status transitions, disk caching, and error handling.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use euca_asset::ai_gen::service::GenerationService;
use euca_asset::ai_gen::{
    AssetGenerator, GenError, GenerationId, GenerationRequest, GenerationStatus, Quality,
};

// ---------------------------------------------------------------------------
// MockGenerator
// ---------------------------------------------------------------------------

/// A fake provider that returns pre-scripted poll responses and produces
/// a minimal valid GLB binary on download.
struct MockGenerator {
    responses: Vec<GenerationStatus>,
    call_count: AtomicUsize,
}

impl MockGenerator {
    fn new(responses: Vec<GenerationStatus>) -> Self {
        Self {
            responses,
            call_count: AtomicUsize::new(0),
        }
    }
}

impl AssetGenerator for MockGenerator {
    fn name(&self) -> &str {
        "mock"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn generate(&self, _request: &GenerationRequest) -> Result<GenerationId, GenError> {
        Ok(GenerationId("mock_task_1".into()))
    }

    fn poll(&self, _id: &GenerationId) -> Result<GenerationStatus, GenError> {
        let idx = self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.responses[idx.min(self.responses.len() - 1)].clone())
    }

    fn download(&self, _url: &str) -> Result<Vec<u8>, GenError> {
        // Minimal valid GLB (glTF binary) header + empty JSON chunk.
        let json = b"{}";
        let json_padded_len = (json.len() + 3) & !3; // align to 4 bytes
        let total_len = 12 + 8 + json_padded_len; // header + chunk-header + data

        let mut glb = Vec::with_capacity(total_len);
        glb.extend_from_slice(b"glTF"); // magic
        glb.extend_from_slice(&2u32.to_le_bytes()); // version
        glb.extend_from_slice(&(total_len as u32).to_le_bytes()); // total length
        // JSON chunk
        glb.extend_from_slice(&(json_padded_len as u32).to_le_bytes()); // chunk length
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // chunk type: JSON
        glb.extend_from_slice(json);
        // Pad JSON chunk to 4-byte alignment with spaces (per glTF spec).
        for _ in json.len()..json_padded_len {
            glb.push(b' ');
        }

        Ok(glb)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("euca_integration_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn mock_request(prompt: &str) -> GenerationRequest {
    GenerationRequest {
        prompt: Some(prompt.into()),
        ..Default::default()
    }
}

fn service_with_mock(responses: Vec<GenerationStatus>, dir: PathBuf) -> GenerationService {
    let mut providers: HashMap<String, Box<dyn AssetGenerator>> = HashMap::new();
    providers.insert("mock".into(), Box::new(MockGenerator::new(responses)));
    GenerationService::with_providers(providers, dir)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full lifecycle: start generation, poll through pending, poll to complete,
/// verify the GLB file is saved to disk with a valid glTF binary header.
#[test]
fn mock_generate_poll_download_flow() {
    let dir = temp_dir("gen_poll_download");
    let mut svc = service_with_mock(
        vec![
            GenerationStatus::Pending { progress: 0.5 },
            GenerationStatus::Complete {
                download_url: "https://example.com/model.glb".into(),
            },
        ],
        dir.clone(),
    );

    // Start generation.
    let task_id = svc
        .start("mock", &mock_request("a treasure chest"))
        .unwrap();

    // First poll: still pending.
    let status = svc.update(&task_id).unwrap().clone();
    assert!(
        matches!(status, GenerationStatus::Pending { progress } if (progress - 0.5).abs() < f32::EPSILON),
        "expected Pending with 0.5 progress, got {status:?}"
    );

    // Second poll: complete — triggers auto-download.
    let status = svc.update(&task_id).unwrap().clone();
    assert!(
        matches!(status, GenerationStatus::Complete { .. }),
        "expected Complete, got {status:?}"
    );

    // Verify the GLB file was saved to disk.
    let path = svc
        .file_path(&task_id)
        .expect("file_path should be set after completion");
    assert!(path.exists(), "GLB file should exist at {}", path.display());

    // Verify the file starts with a valid glTF binary magic.
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.len() >= 12, "GLB too short");
    assert_eq!(&bytes[0..4], b"glTF", "GLB magic mismatch");
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        2,
        "GLB version should be 2"
    );
}

/// Generating with the same prompt twice should return a cached result on the
/// second call without invoking the provider's generate method again.
#[test]
fn cached_result_skips_generation() {
    let dir = temp_dir("cached_result");

    // First service: generate and complete.
    let mut svc = service_with_mock(
        vec![GenerationStatus::Complete {
            download_url: "https://example.com/model.glb".into(),
        }],
        dir.clone(),
    );
    let req = mock_request("a wooden shield");
    let id1 = svc.start("mock", &req).unwrap();
    // Poll to trigger download and disk-write.
    let _ = svc.update(&id1).unwrap();

    // Second service: same output dir — the cache file should already exist.
    let mut svc2 = service_with_mock(
        vec![
            // This response should never be reached if caching works.
            GenerationStatus::Failed {
                error: "should not be called".into(),
            },
        ],
        dir,
    );
    let id2 = svc2.start("mock", &req).unwrap();

    // The task should be Complete immediately (cache hit), without any poll.
    let status = svc2.status(&id2).unwrap();
    assert!(
        matches!(status, GenerationStatus::Complete { .. }),
        "expected cache hit (Complete), got {status:?}"
    );
}

/// Requesting generation from a nonexistent provider returns an error.
#[test]
fn unknown_provider_returns_error() {
    let dir = temp_dir("unknown_provider");
    let mut svc = GenerationService::with_providers(HashMap::new(), dir);
    let req = mock_request("anything");
    let err = svc.start("nonexistent", &req).unwrap_err();
    assert!(
        matches!(err, GenError::InvalidRequest(_)),
        "expected InvalidRequest, got {err:?}"
    );
}

/// The default `GenerationService::new` registers all real providers.
#[test]
fn service_lists_providers() {
    let svc = GenerationService::new(PathBuf::from("/tmp/euca_provider_list_test"));
    let registered = svc.registered_providers();
    assert_eq!(registered.len(), 7);
    assert!(registered.contains(&"tripo"));
    assert!(registered.contains(&"meshy"));
    assert!(registered.contains(&"rodin"));
    assert!(registered.contains(&"hunyuan"));
    assert!(registered.contains(&"stability"));
    assert!(registered.contains(&"blockade_labs"));
    assert!(registered.contains(&"world_labs"));
}
