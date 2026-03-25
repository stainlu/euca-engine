//! Generation service that manages async AI asset generation across providers.
//!
//! [`GenerationService`] wraps the individual [`AssetGenerator`] providers and
//! provides a unified interface for starting generation tasks, polling their
//! status, auto-downloading completed GLBs, and caching results on disk.
//!
//! # Caching
//!
//! Before submitting a request to a provider, the service checks if a GLB
//! already exists on disk for the same provider + prompt combination. If so,
//! it returns immediately with a `Complete` status and the cached path.
//!
//! # ECS integration
//!
//! The module also defines [`PendingAsset`] (a component for entities waiting
//! on generation) and [`AssetGeneratedEvent`] (emitted when a task completes).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use euca_ecs::Entity;

use super::hunyuan::HunyuanGenerator;
use super::meshy::MeshyGenerator;
use super::rodin::RodinGenerator;
use super::tripo::TripoGenerator;
use super::{AssetGenerator, GenError, GenerationId, GenerationRequest, GenerationStatus};

// ---------------------------------------------------------------------------
// TaskEntry
// ---------------------------------------------------------------------------

/// Internal bookkeeping for a generation task.
struct TaskEntry {
    /// Which provider is handling this task (e.g. `"tripo"`).
    provider: String,
    /// The provider-level task identifier.
    provider_id: GenerationId,
    /// The original prompt (used for display / events).
    prompt: String,
    /// Current generation status.
    status: GenerationStatus,
    /// Path to the downloaded GLB file, populated on completion.
    file_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// GenerationService
// ---------------------------------------------------------------------------

/// Manages AI 3D asset generation across multiple providers.
///
/// Provides task lifecycle management (start, poll, download), prompt-based
/// disk caching, and unique service-level task IDs.
pub struct GenerationService {
    /// Registered providers, keyed by name.
    providers: HashMap<String, Box<dyn AssetGenerator>>,
    /// Active generation tasks, keyed by service-level task ID.
    tasks: HashMap<String, TaskEntry>,
    /// Directory where generated GLB files are written.
    output_dir: PathBuf,
    /// Monotonic counter for generating unique task IDs.
    next_id: u64,
}

impl GenerationService {
    /// Create a service with the four default providers, each reading its API
    /// key from the environment.
    pub fn new(output_dir: PathBuf) -> Self {
        let mut providers: HashMap<String, Box<dyn AssetGenerator>> = HashMap::new();
        providers.insert("tripo".into(), Box::new(TripoGenerator::new()));
        providers.insert("meshy".into(), Box::new(MeshyGenerator::new()));
        providers.insert("rodin".into(), Box::new(RodinGenerator::new()));
        providers.insert("hunyuan".into(), Box::new(HunyuanGenerator::new()));
        Self {
            providers,
            tasks: HashMap::new(),
            output_dir,
            next_id: 0,
        }
    }

    /// Create a service with explicitly-provided generators (useful for tests).
    #[cfg(test)]
    fn with_providers(
        providers: HashMap<String, Box<dyn AssetGenerator>>,
        output_dir: PathBuf,
    ) -> Self {
        Self {
            providers,
            tasks: HashMap::new(),
            output_dir,
            next_id: 0,
        }
    }

    /// List the names of providers that have a valid API key configured.
    pub fn available_providers(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .providers
            .iter()
            .filter(|(_, generator)| generator.is_available())
            .map(|(name, _)| name.as_str())
            .collect();
        names.sort_unstable();
        names
    }

    /// Start a generation task on `provider`.
    ///
    /// Returns a service-level task ID (e.g. `"gen_0"`). If a cached GLB
    /// already exists for this provider + prompt, the task is created with
    /// `Complete` status immediately.
    pub fn start(
        &mut self,
        provider: &str,
        request: &GenerationRequest,
    ) -> Result<String, GenError> {
        let generator = self
            .providers
            .get(provider)
            .ok_or_else(|| GenError::InvalidRequest(format!("unknown provider: {provider}")))?;

        let prompt = request.prompt.clone().unwrap_or_default();

        // Check disk cache before hitting the network.
        let cache_path = self.cache_path(provider, &prompt);
        if cache_path.exists() {
            let task_id = self.alloc_task_id();
            log::info!(
                "[GenerationService] cache hit for \"{prompt}\" via {provider}: {}",
                cache_path.display()
            );
            self.tasks.insert(
                task_id.clone(),
                TaskEntry {
                    provider: provider.to_owned(),
                    provider_id: GenerationId(String::new()),
                    prompt,
                    status: GenerationStatus::Complete {
                        download_url: String::new(),
                    },
                    file_path: Some(cache_path),
                },
            );
            return Ok(task_id);
        }

        let provider_id = generator.generate(request)?;
        let task_id = self.alloc_task_id();

        self.tasks.insert(
            task_id.clone(),
            TaskEntry {
                provider: provider.to_owned(),
                provider_id,
                prompt,
                status: GenerationStatus::Pending { progress: 0.0 },
                file_path: None,
            },
        );

        log::info!("[GenerationService] started task {task_id} on {provider}");
        Ok(task_id)
    }

    /// Poll a single task and update its status.
    ///
    /// When a task transitions to `Complete`, the generated GLB is
    /// auto-downloaded and saved to `output_dir/{task_id}.glb`.
    pub fn update(&mut self, task_id: &str) -> Result<&GenerationStatus, GenError> {
        // Pull the entry out so we can borrow `self` for the provider lookup.
        let mut entry = self
            .tasks
            .remove(task_id)
            .ok_or_else(|| GenError::InvalidRequest(format!("unknown task: {task_id}")))?;

        // Only poll tasks that are still pending.
        if matches!(entry.status, GenerationStatus::Pending { .. }) {
            let generator = self.providers.get(&entry.provider).ok_or_else(|| {
                GenError::InvalidRequest(format!("provider {} not found", entry.provider))
            })?;

            let new_status = generator.poll(&entry.provider_id)?;

            // If the provider reports completion, download the GLB.
            if let GenerationStatus::Complete { ref download_url } = new_status {
                match self.download_and_save(&entry.provider, task_id, &entry.prompt, download_url)
                {
                    Ok(path) => {
                        entry.file_path = Some(path);
                    }
                    Err(e) => {
                        log::error!("[GenerationService] download failed for task {task_id}: {e}");
                        entry.status = GenerationStatus::Failed {
                            error: format!("download failed: {e}"),
                        };
                        self.tasks.insert(task_id.to_owned(), entry);
                        return Ok(&self.tasks[task_id].status);
                    }
                }
            }

            entry.status = new_status;
        }

        self.tasks.insert(task_id.to_owned(), entry);
        Ok(&self.tasks[task_id].status)
    }

    /// Poll **all** active (pending) tasks. Returns a snapshot of each task's
    /// current status.
    ///
    /// Call this periodically (e.g. once per frame or on a timer) to drive
    /// generation progress.
    pub fn update_all(&mut self) -> Vec<(String, GenerationStatus)> {
        let pending_ids: Vec<String> = self
            .tasks
            .iter()
            .filter(|(_, e)| matches!(e.status, GenerationStatus::Pending { .. }))
            .map(|(id, _)| id.clone())
            .collect();

        let mut results = Vec::with_capacity(pending_ids.len());
        for id in pending_ids {
            match self.update(&id) {
                Ok(status) => results.push((id, status.clone())),
                Err(e) => {
                    log::error!("[GenerationService] update failed for task {id}: {e}");
                    results.push((
                        id,
                        GenerationStatus::Failed {
                            error: e.to_string(),
                        },
                    ));
                }
            }
        }
        results
    }

    /// Get the current status of a task without polling the provider.
    pub fn status(&self, task_id: &str) -> Option<&GenerationStatus> {
        self.tasks.get(task_id).map(|e| &e.status)
    }

    /// Get the file path for a completed task, if available.
    pub fn file_path(&self, task_id: &str) -> Option<&Path> {
        self.tasks.get(task_id).and_then(|e| e.file_path.as_deref())
    }

    /// List all tasks as `(task_id, prompt, status)` triples.
    pub fn list_tasks(&self) -> Vec<(&str, &str, &GenerationStatus)> {
        self.tasks
            .iter()
            .map(|(id, e)| (id.as_str(), e.prompt.as_str(), &e.status))
            .collect()
    }

    // -- private helpers ----------------------------------------------------

    /// Allocate a unique task ID.
    fn alloc_task_id(&mut self) -> String {
        let id = format!("gen_{}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Deterministic cache path for a `(provider, prompt)` pair.
    ///
    /// The filename is derived from the provider name and a sanitized,
    /// truncated prompt: `{provider}_{sanitized_prompt}.glb`.
    fn cache_path(&self, provider: &str, prompt: &str) -> PathBuf {
        let sanitized: String = prompt
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .take(50)
            .collect();
        self.output_dir.join(format!("{provider}_{sanitized}.glb"))
    }

    /// Download GLB bytes from the provider and write them to disk.
    ///
    /// The file is saved both at the task-id path (for retrieval by callers)
    /// and at the cache path (for future prompt-based cache hits). When both
    /// paths differ, a second copy is written so that re-generating the same
    /// prompt short-circuits.
    fn download_and_save(
        &self,
        provider_name: &str,
        task_id: &str,
        prompt: &str,
        download_url: &str,
    ) -> Result<PathBuf, GenError> {
        let generator = self.providers.get(provider_name).ok_or_else(|| {
            GenError::InvalidRequest(format!("provider {provider_name} not found"))
        })?;

        let bytes = generator.download(download_url)?;

        // Ensure the output directory exists.
        std::fs::create_dir_all(&self.output_dir).map_err(|e| {
            GenError::HttpError(format!(
                "failed to create output dir {}: {e}",
                self.output_dir.display()
            ))
        })?;

        // Primary path: task-id based.
        let task_path = self.output_dir.join(format!("{task_id}.glb"));
        std::fs::write(&task_path, &bytes).map_err(|e| {
            GenError::HttpError(format!("failed to write {}: {e}", task_path.display()))
        })?;

        // Cache path: prompt-based (enables cache hits on future identical prompts).
        let cache = self.cache_path(provider_name, prompt);
        if cache != task_path {
            let _ = std::fs::write(&cache, &bytes);
        }

        log::info!(
            "[GenerationService] saved {} bytes to {}",
            bytes.len(),
            task_path.display()
        );

        Ok(task_path)
    }
}

// ---------------------------------------------------------------------------
// ECS integration types
// ---------------------------------------------------------------------------

/// Component attached to entities waiting for an AI-generated asset.
///
/// When the associated task completes, systems can remove this component and
/// attach the loaded mesh instead.
#[derive(Clone, Debug)]
pub struct PendingAsset {
    /// Service-level task ID (from [`GenerationService::start`]).
    pub task_id: String,
}

/// Event emitted when a generation task completes successfully.
#[derive(Clone, Debug)]
pub struct AssetGeneratedEvent {
    /// Service-level task ID.
    pub task_id: String,
    /// The original text prompt.
    pub prompt: String,
    /// Path to the downloaded GLB file on disk.
    pub file_path: PathBuf,
    /// Optional entity that was waiting on this asset.
    pub entity: Option<Entity>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A dummy generator that never contacts a real API.
    struct FakeGenerator {
        name: &'static str,
        available: bool,
    }

    impl AssetGenerator for FakeGenerator {
        fn name(&self) -> &str {
            self.name
        }
        fn is_available(&self) -> bool {
            self.available
        }
        fn generate(&self, _request: &GenerationRequest) -> Result<GenerationId, GenError> {
            if !self.available {
                return Err(GenError::NoApiKey);
            }
            Ok(GenerationId("fake-provider-id".into()))
        }
        fn poll(&self, _id: &GenerationId) -> Result<GenerationStatus, GenError> {
            Ok(GenerationStatus::Pending { progress: 0.5 })
        }
        fn download(&self, _url: &str) -> Result<Vec<u8>, GenError> {
            Ok(b"fake-glb-data".to_vec())
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("euca_service_test_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn service_with_fake(name: &'static str, available: bool, dir: PathBuf) -> GenerationService {
        let mut providers: HashMap<String, Box<dyn AssetGenerator>> = HashMap::new();
        providers.insert(name.into(), Box::new(FakeGenerator { name, available }));
        GenerationService::with_providers(providers, dir)
    }

    // -- constructor & provider listing ------------------------------------

    #[test]
    fn new_creates_four_providers() {
        let svc = GenerationService::new(PathBuf::from("/tmp/euca_test"));
        assert_eq!(svc.providers.len(), 4);
        assert!(svc.providers.contains_key("tripo"));
        assert!(svc.providers.contains_key("meshy"));
        assert!(svc.providers.contains_key("rodin"));
        assert!(svc.providers.contains_key("hunyuan"));
    }

    #[test]
    fn available_providers_returns_only_configured() {
        let dir = temp_dir("available");
        let mut providers: HashMap<String, Box<dyn AssetGenerator>> = HashMap::new();
        providers.insert(
            "yes".into(),
            Box::new(FakeGenerator {
                name: "yes",
                available: true,
            }),
        );
        providers.insert(
            "no".into(),
            Box::new(FakeGenerator {
                name: "no",
                available: false,
            }),
        );
        let svc = GenerationService::with_providers(providers, dir);
        let avail = svc.available_providers();
        assert_eq!(avail, vec!["yes"]);
    }

    // -- start -------------------------------------------------------------

    #[test]
    fn start_unknown_provider_returns_error() {
        let dir = temp_dir("unknown_provider");
        let mut svc = GenerationService::with_providers(HashMap::new(), dir);
        let req = GenerationRequest {
            prompt: Some("a chair".into()),
            image: None,
            quality: super::super::Quality::Medium,
        };
        let err = svc.start("nonexistent", &req).unwrap_err();
        assert!(matches!(err, GenError::InvalidRequest(_)));
    }

    #[test]
    fn start_creates_task_with_pending_status() {
        let dir = temp_dir("start_pending");
        let mut svc = service_with_fake("fake", true, dir);
        let req = GenerationRequest {
            prompt: Some("a robot".into()),
            image: None,
            quality: super::super::Quality::Medium,
        };
        let id = svc.start("fake", &req).unwrap();
        assert_eq!(id, "gen_0");
        let status = svc.status(&id).unwrap();
        assert!(matches!(status, GenerationStatus::Pending { .. }));
    }

    // -- status / list_tasks -----------------------------------------------

    #[test]
    fn status_unknown_task_returns_none() {
        let dir = temp_dir("status_none");
        let svc = GenerationService::with_providers(HashMap::new(), dir);
        assert!(svc.status("gen_999").is_none());
    }

    #[test]
    fn list_tasks_shows_all_tasks() {
        let dir = temp_dir("list_tasks");
        let mut svc = service_with_fake("fake", true, dir);
        let req = GenerationRequest {
            prompt: Some("sword".into()),
            image: None,
            quality: super::super::Quality::Low,
        };
        svc.start("fake", &req).unwrap();
        svc.start("fake", &req).unwrap();
        assert_eq!(svc.list_tasks().len(), 2);
    }

    // -- cache hit ---------------------------------------------------------

    #[test]
    fn start_returns_cached_when_file_exists() {
        let dir = temp_dir("cache_hit");

        // Pre-populate the cache file.
        let cache_file = dir.join("fake_a_wooden_chair.glb");
        let mut f = std::fs::File::create(&cache_file).unwrap();
        f.write_all(b"cached-glb").unwrap();

        let mut svc = service_with_fake("fake", true, dir);
        let req = GenerationRequest {
            prompt: Some("a wooden chair".into()),
            image: None,
            quality: super::super::Quality::Medium,
        };
        let id = svc.start("fake", &req).unwrap();

        // Status should be Complete immediately.
        let status = svc.status(&id).unwrap();
        assert!(matches!(status, GenerationStatus::Complete { .. }));

        // File path should point at the cached file.
        let path = svc.file_path(&id).unwrap();
        assert_eq!(path, cache_file);
    }

    // -- ECS types ---------------------------------------------------------

    #[test]
    fn pending_asset_roundtrip() {
        let pa = PendingAsset {
            task_id: "gen_42".into(),
        };
        assert_eq!(pa.task_id, "gen_42");
        let pa2 = pa.clone();
        assert_eq!(pa2.task_id, pa.task_id);
    }

    #[test]
    fn asset_generated_event_roundtrip() {
        let evt = AssetGeneratedEvent {
            task_id: "gen_0".into(),
            prompt: "a sword".into(),
            file_path: PathBuf::from("/tmp/gen_0.glb"),
            entity: None,
        };
        assert_eq!(evt.task_id, "gen_0");
        assert!(evt.entity.is_none());
        let evt2 = evt.clone();
        assert_eq!(evt2.prompt, "a sword");
    }

    // -- alloc_task_id monotonicity ----------------------------------------

    #[test]
    fn task_ids_are_monotonically_increasing() {
        let dir = temp_dir("monotonic_ids");
        let mut svc = service_with_fake("fake", true, dir);
        let req = GenerationRequest {
            prompt: Some("test".into()),
            image: None,
            quality: super::super::Quality::Medium,
        };
        let id0 = svc.start("fake", &req).unwrap();
        let id1 = svc.start("fake", &req).unwrap();
        assert_eq!(id0, "gen_0");
        assert_eq!(id1, "gen_1");
    }
}
