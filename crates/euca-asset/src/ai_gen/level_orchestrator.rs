//! Level orchestrator — coordinates multiple AI providers to generate a
//! complete level from a text prompt.
//!
//! The orchestrator is a thin convenience layer on top of
//! [`GenerationService`]. It:
//!
//! 1. Calls an LLM to produce a level layout (entity positions, terrain zones,
//!    sub-prompts for heightmap / skybox / props).
//! 2. Kicks off parallel generation tasks via the service.
//! 3. Provides a single poll method that checks all outstanding tasks.
//! 4. Assembles the results into a [`LevelData`] + associated asset file paths.
//!
//! The orchestrator is **optional** — artists can always provide their own
//! PNGs, GLBs, and JSON directly. This module exists for the "generate a
//! complete level from a single text prompt" workflow.

use std::collections::HashMap;
use std::path::PathBuf;

use euca_terrain::level_data::{CameraConfig, EntityPlacement, LevelData, NavConfig, SurfaceType};

use super::service::GenerationService;
use super::{GenError, GenerationKind, GenerationRequest, GenerationStatus, Quality};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for level generation.
#[derive(Clone, Debug)]
pub struct LevelConfig {
    /// Grid columns.
    pub width: u32,
    /// Grid rows.
    pub height: u32,
    /// World-space distance between cells.
    pub cell_size: f32,
    /// Maximum terrain height.
    pub max_height: f32,
    /// Whether to generate a skybox.
    pub generate_skybox: bool,
    /// Whether to generate 3D props for entities.
    pub generate_props: bool,
    /// Quality tier for all generation tasks.
    pub quality: Quality,
    /// Preferred provider for heightmap generation.
    pub heightmap_provider: String,
    /// Preferred provider for skybox generation.
    pub skybox_provider: String,
    /// Preferred provider for 3D prop generation.
    pub prop_provider: String,
}

impl Default for LevelConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            cell_size: 1.0,
            max_height: 50.0,
            generate_skybox: true,
            generate_props: true,
            quality: Quality::Medium,
            heightmap_provider: "stability".into(),
            skybox_provider: "blockade_labs".into(),
            prop_provider: "tripo".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Layout — intermediate representation from LLM
// ---------------------------------------------------------------------------

/// The layout produced by the LLM layout generation step.
///
/// This is the structured output that the LLM produces from a free-form text
/// prompt. It describes WHAT the level should contain — the orchestrator then
/// dispatches generation tasks for each asset.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LevelLayout {
    /// Sub-prompt for heightmap generation (describes terrain shape).
    pub terrain_prompt: String,
    /// Sub-prompt for skybox generation (describes sky / environment).
    pub skybox_prompt: String,
    /// Surface type grid (row-major). If empty, defaults to all Grass.
    #[serde(default)]
    pub surface_grid: Vec<String>,
    /// Entities to place with their positions and types.
    pub entities: Vec<LayoutEntity>,
    /// Camera configuration.
    #[serde(default)]
    pub camera: Option<CameraConfig>,
}

/// An entity in the layout, possibly requiring a generated 3D model.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LayoutEntity {
    /// Position in world coordinates.
    pub position: [f32; 3],
    /// Entity type identifier.
    pub entity_type: String,
    /// If set, a text prompt to generate a 3D model for this entity.
    #[serde(default)]
    pub model_prompt: Option<String>,
    /// Arbitrary properties.
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Task tracking
// ---------------------------------------------------------------------------

/// Tracks all generation tasks for one level.
#[derive(Clone, Debug)]
pub struct LevelGenerationProgress {
    /// Task ID for the heightmap generation (if started).
    pub heightmap_task: Option<String>,
    /// Task ID for the skybox generation (if started).
    pub skybox_task: Option<String>,
    /// Task IDs for 3D prop generation, keyed by entity index.
    pub prop_tasks: HashMap<usize, String>,
    /// The layout that was used to start generation.
    pub layout: LevelLayout,
    /// The config used.
    pub config: LevelConfig,
}

impl LevelGenerationProgress {
    /// Check if all tasks have completed (or failed).
    pub fn is_done(&self, service: &GenerationService) -> bool {
        let all_tasks = self.all_task_ids();
        all_tasks.iter().all(|id| {
            matches!(
                service.status(id),
                Some(GenerationStatus::Complete { .. } | GenerationStatus::Failed { .. }) | None
            )
        })
    }

    /// Collect all task IDs.
    fn all_task_ids(&self) -> Vec<&str> {
        let mut ids = Vec::new();
        if let Some(ref id) = self.heightmap_task {
            ids.push(id.as_str());
        }
        if let Some(ref id) = self.skybox_task {
            ids.push(id.as_str());
        }
        for id in self.prop_tasks.values() {
            ids.push(id.as_str());
        }
        ids
    }
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Coordinates multi-provider level generation.
///
/// The orchestrator does NOT own a `GenerationService` — it borrows one. This
/// keeps ownership simple (the service lives in the ECS world resource or
/// wherever the caller manages it).
pub struct LevelOrchestrator;

impl LevelOrchestrator {
    /// Start generating a level from a pre-built layout.
    ///
    /// This kicks off all generation tasks (heightmap, skybox, props) in
    /// parallel via the service. Returns a progress tracker that the caller
    /// can poll.
    pub fn start(
        service: &mut GenerationService,
        layout: LevelLayout,
        config: &LevelConfig,
    ) -> Result<LevelGenerationProgress, GenError> {
        // -- Heightmap --
        let heightmap_task = {
            let req = GenerationRequest {
                prompt: Some(layout.terrain_prompt.clone()),
                quality: config.quality,
                kind: GenerationKind::Heightmap,
                dimensions: Some((config.width, config.height)),
                ..Default::default()
            };
            Some(service.start(&config.heightmap_provider, &req)?)
        };

        // -- Skybox --
        let skybox_task = if config.generate_skybox {
            let req = GenerationRequest {
                prompt: Some(layout.skybox_prompt.clone()),
                quality: config.quality,
                kind: GenerationKind::Skybox,
                ..Default::default()
            };
            Some(service.start(&config.skybox_provider, &req)?)
        } else {
            None
        };

        // -- Props --
        let mut prop_tasks = HashMap::new();
        if config.generate_props {
            for (idx, entity) in layout.entities.iter().enumerate() {
                if let Some(ref prompt) = entity.model_prompt {
                    let req = GenerationRequest {
                        prompt: Some(prompt.clone()),
                        quality: config.quality,
                        kind: GenerationKind::Model3D,
                        ..Default::default()
                    };
                    let task_id = service.start(&config.prop_provider, &req)?;
                    prop_tasks.insert(idx, task_id);
                }
            }
        }

        Ok(LevelGenerationProgress {
            heightmap_task,
            skybox_task,
            prop_tasks,
            layout,
            config: config.clone(),
        })
    }

    /// Assemble completed generation results into a [`LevelData`].
    ///
    /// Call this after `progress.is_done(service)` returns `true`. Returns
    /// the level data and a map of asset file paths (skybox, prop meshes).
    pub fn assemble(
        service: &GenerationService,
        progress: &LevelGenerationProgress,
    ) -> Result<(LevelData, HashMap<String, PathBuf>), GenError> {
        let config = &progress.config;
        let layout = &progress.layout;

        // Build the base LevelData.
        let mut level = LevelData::new(config.width, config.height, config.cell_size);
        level.max_height = config.max_height;
        level.interpolate_height = true;

        // Parse surface grid if provided.
        if !layout.surface_grid.is_empty() {
            let count = level.cell_count();
            for (i, name) in layout.surface_grid.iter().enumerate().take(count) {
                level.surface[i] = parse_surface_type(name);
            }
        }

        // Set camera if provided.
        if let Some(ref cam) = layout.camera {
            level.camera = cam.clone();
        }

        // Set nav config.
        level.nav_config = NavConfig::default();

        // Add entities.
        for entity in &layout.entities {
            level.entities.push(EntityPlacement {
                position: euca_math::Vec3::new(
                    entity.position[0],
                    entity.position[1],
                    entity.position[2],
                ),
                rotation: euca_math::Quat::IDENTITY,
                scale: euca_math::Vec3::ONE,
                entity_type: entity.entity_type.clone(),
                properties: entity.properties.clone(),
            });
        }

        // Collect asset paths.
        let mut assets: HashMap<String, PathBuf> = HashMap::new();

        if let Some(ref task_id) = progress.heightmap_task
            && let Some(path) = service.file_path(task_id)
        {
            assets.insert("heightmap".into(), path.to_path_buf());
        }

        if let Some(ref task_id) = progress.skybox_task
            && let Some(path) = service.file_path(task_id)
        {
            assets.insert("skybox".into(), path.to_path_buf());
        }

        for (idx, task_id) in &progress.prop_tasks {
            if let Some(path) = service.file_path(task_id) {
                let key = format!("prop_{idx}");
                assets.insert(key, path.to_path_buf());
            }
        }

        Ok((level, assets))
    }
}

/// Parse a surface type name string into a [`SurfaceType`].
fn parse_surface_type(name: &str) -> SurfaceType {
    match name.to_lowercase().as_str() {
        "grass" => SurfaceType::Grass,
        "dirt" => SurfaceType::Dirt,
        "stone" | "rock" => SurfaceType::Stone,
        "water" => SurfaceType::Water,
        "sand" => SurfaceType::Sand,
        "snow" => SurfaceType::Snow,
        "mud" => SurfaceType::Mud,
        "road" => SurfaceType::Road,
        "cliff" => SurfaceType::Cliff,
        "void" => SurfaceType::Void,
        _ => SurfaceType::Grass,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = LevelConfig::default();
        assert_eq!(config.width, 64);
        assert_eq!(config.height, 64);
        assert_eq!(config.heightmap_provider, "stability");
        assert_eq!(config.skybox_provider, "blockade_labs");
        assert_eq!(config.prop_provider, "tripo");
    }

    #[test]
    fn parse_surface_types() {
        assert_eq!(parse_surface_type("grass"), SurfaceType::Grass);
        assert_eq!(parse_surface_type("DIRT"), SurfaceType::Dirt);
        assert_eq!(parse_surface_type("Rock"), SurfaceType::Stone);
        assert_eq!(parse_surface_type("water"), SurfaceType::Water);
        assert_eq!(parse_surface_type("unknown"), SurfaceType::Grass);
    }

    #[test]
    fn layout_entity_serde() {
        let entity = LayoutEntity {
            position: [1.0, 2.0, 3.0],
            entity_type: "tree_oak".into(),
            model_prompt: Some("a large oak tree".into()),
            properties: HashMap::from([("health".into(), "100".into())]),
        };
        let json = serde_json::to_string(&entity).unwrap();
        let restored: LayoutEntity = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.entity_type, "tree_oak");
        assert_eq!(restored.model_prompt, Some("a large oak tree".into()));
    }

    #[test]
    fn layout_serde_roundtrip() {
        let layout = LevelLayout {
            terrain_prompt: "mountainous terrain".into(),
            skybox_prompt: "sunset sky".into(),
            surface_grid: vec!["grass".into(), "water".into()],
            entities: vec![LayoutEntity {
                position: [0.0, 0.0, 0.0],
                entity_type: "spawn_point".into(),
                model_prompt: None,
                properties: HashMap::new(),
            }],
            camera: None,
        };
        let json = serde_json::to_string(&layout).unwrap();
        let restored: LevelLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.terrain_prompt, "mountainous terrain");
        assert_eq!(restored.entities.len(), 1);
    }

    #[test]
    fn assemble_produces_level_data() {
        // Create a mock service and progress with no actual tasks.
        let service =
            GenerationService::with_providers(HashMap::new(), PathBuf::from("/tmp/euca_orch_test"));

        let layout = LevelLayout {
            terrain_prompt: "flat grass".into(),
            skybox_prompt: "clear sky".into(),
            surface_grid: vec![],
            entities: vec![
                LayoutEntity {
                    position: [5.0, 0.0, 5.0],
                    entity_type: "tree".into(),
                    model_prompt: None,
                    properties: HashMap::new(),
                },
                LayoutEntity {
                    position: [10.0, 0.0, 10.0],
                    entity_type: "rock".into(),
                    model_prompt: None,
                    properties: HashMap::new(),
                },
            ],
            camera: None,
        };

        let config = LevelConfig {
            width: 32,
            height: 32,
            cell_size: 2.0,
            ..Default::default()
        };

        let progress = LevelGenerationProgress {
            heightmap_task: None,
            skybox_task: None,
            prop_tasks: HashMap::new(),
            layout,
            config,
        };

        let (level, assets) = LevelOrchestrator::assemble(&service, &progress).unwrap();
        assert_eq!(level.width, 32);
        assert_eq!(level.height, 32);
        assert!((level.cell_size - 2.0).abs() < f32::EPSILON);
        assert_eq!(level.entities.len(), 2);
        assert_eq!(level.entities[0].entity_type, "tree");
        assert!(assets.is_empty());
    }
}
