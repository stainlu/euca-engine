//! World streaming / chunk loading for the Euca engine.
//!
//! Divides the world into a 2D grid of chunks. As the camera moves, chunks
//! within `load_radius` are loaded and those beyond `unload_radius` are
//! despawned. This keeps memory usage bounded for large open worlds while
//! providing seamless exploration.
//!
//! # Architecture
//!
//! - **[`StreamingConfig`]** — resource controlling chunk dimensions and radii.
//! - **[`StreamingState`]** — resource tracking which chunks are loaded / pending.
//! - **[`WorldChunk`]** — per-chunk bookkeeping: the entities it owns.
//! - **[`ChunkData`]** — serializable payload describing a chunk's content.
//! - **[`ChunkLoader`]** — trait for pluggable chunk I/O backends.
//! - **[`streaming_update_system`]** — the system that drives load / unload.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use serde::{Deserialize, Serialize};

use crate::{GlobalTransform, LocalTransform};

// ── Chunk coordinate helpers ────────────────────────────────────────────────

/// Compute the chunk coordinate for a world-space position given `chunk_size`.
#[inline]
pub fn world_to_chunk(position: Vec3, chunk_size: f32) -> (i32, i32) {
    (
        (position.x / chunk_size).floor() as i32,
        (position.z / chunk_size).floor() as i32,
    )
}

/// Compute all chunk coordinates within `radius` chunks of a center chunk.
///
/// `radius` is measured in chunk units (not world units). A radius of 3 means
/// all chunks whose Chebyshev distance from `center` is <= 3.
pub fn chunks_in_radius(center: (i32, i32), radius: i32) -> Vec<(i32, i32)> {
    let mut result = Vec::with_capacity(((2 * radius + 1) * (2 * radius + 1)) as usize);
    for x in (center.0 - radius)..=(center.0 + radius) {
        for z in (center.1 - radius)..=(center.1 + radius) {
            result.push((x, z));
        }
    }
    result
}

// ── Core types ──────────────────────────────────────────────────────────────

/// Marker resource: the position used to drive streaming decisions.
///
/// If no explicit `CameraPosition` is inserted, the system falls back to the
/// first entity with a `GlobalTransform` in the world, or `Vec3::ZERO`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraPosition(pub Vec3);

/// Bookkeeping for a single loaded world chunk.
#[derive(Clone, Debug)]
pub struct WorldChunk {
    /// Grid coordinate of this chunk (x, z).
    pub chunk_id: (i32, i32),
    /// Entities spawned into the world for this chunk.
    pub entities: Vec<Entity>,
    /// Whether the chunk is currently loaded (entities present in the world).
    pub loaded: bool,
}

/// Configuration resource controlling streaming behaviour.
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    /// Side length of a single chunk in world units.
    pub chunk_size: f32,
    /// Chunks within this many chunk-units of the camera are loaded.
    pub load_radius: i32,
    /// Chunks beyond this many chunk-units of the camera are unloaded.
    pub unload_radius: i32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64.0,
            load_radius: 3,
            unload_radius: 5,
        }
    }
}

/// Tracks which chunks are loaded, pending load, and pending unload.
#[derive(Clone, Debug, Default)]
pub struct StreamingState {
    /// Currently loaded chunks, keyed by chunk coordinate.
    pub loaded: HashMap<(i32, i32), WorldChunk>,
    /// Chunks that should be loaded next frame.
    pub pending_load: HashSet<(i32, i32)>,
    /// Chunks that should be unloaded next frame.
    pub pending_unload: HashSet<(i32, i32)>,
}

impl StreamingState {
    /// Returns `true` if the given chunk is currently loaded.
    pub fn is_loaded(&self, chunk_id: (i32, i32)) -> bool {
        self.loaded.contains_key(&chunk_id)
    }

    /// The number of currently loaded chunks.
    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }
}

/// Serializable description of a chunk's contents, suitable for storage or
/// network transfer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkData {
    /// Which chunk this data belongs to.
    pub chunk_id: (i32, i32),
    /// Entity definitions: position + optional component payload.
    pub entities: Vec<ChunkEntityData>,
}

/// A single entity definition within a [`ChunkData`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkEntityData {
    /// Position relative to the chunk origin.
    pub position: Vec3,
    /// Optional human-readable label.
    pub name: Option<String>,
}

// ── ChunkLoader trait ───────────────────────────────────────────────────────

/// Pluggable backend for loading and unloading chunk data.
///
/// Implement this trait to back chunk streaming with a file system, network
/// source, procedural generator, or any other data source.
pub trait ChunkLoader: Send + Sync {
    /// Load the data for a chunk. Returns `None` if the chunk has no data
    /// (e.g. empty ocean tile).
    fn load(&self, chunk_id: (i32, i32)) -> Option<ChunkData>;

    /// Notification that a chunk has been unloaded. Implementations may use
    /// this to persist dirty state or release cached resources.
    fn unload(&self, chunk_id: (i32, i32));
}

/// A no-op chunk loader that always returns an empty chunk. Useful for tests
/// and as a default when no real loader is configured.
#[derive(Clone, Debug, Default)]
pub struct NullChunkLoader;

impl ChunkLoader for NullChunkLoader {
    fn load(&self, chunk_id: (i32, i32)) -> Option<ChunkData> {
        Some(ChunkData {
            chunk_id,
            entities: Vec::new(),
        })
    }

    fn unload(&self, _chunk_id: (i32, i32)) {}
}

// ── System ──────────────────────────────────────────────────────────────────

/// The main streaming system. Call once per frame.
///
/// 1. Reads the camera/player position (from `CameraPosition` resource, or
///    falls back to the first `GlobalTransform`).
/// 2. Computes which chunks should be loaded (within `load_radius`).
/// 3. Marks chunks beyond `unload_radius` for unloading.
/// 4. Processes pending loads: spawns entities from `ChunkLoader` into the world.
/// 5. Processes pending unloads: despawns chunk entities from the world.
pub fn streaming_update_system(world: &mut World) {
    // Ensure resources exist with defaults.
    if world.resource::<StreamingConfig>().is_none() {
        world.insert_resource(StreamingConfig::default());
    }
    if world.resource::<StreamingState>().is_none() {
        world.insert_resource(StreamingState::default());
    }

    // 1. Determine camera position.
    let camera_pos = if let Some(cam) = world.resource::<CameraPosition>() {
        cam.0
    } else {
        // Fallback: first entity with a GlobalTransform.
        let query = Query::<&GlobalTransform>::new(world);
        match query.iter().next() {
            Some(gt) => gt.0.translation,
            None => Vec3::ZERO,
        }
    };

    // Read config (copy to release borrow).
    // SAFETY: resource was just ensured above.
    let config = world
        .resource::<StreamingConfig>()
        .expect("StreamingConfig resource missing")
        .clone();

    let camera_chunk = world_to_chunk(camera_pos, config.chunk_size);

    // 2. Compute desired loaded set.
    let desired: HashSet<(i32, i32)> = chunks_in_radius(camera_chunk, config.load_radius)
        .into_iter()
        .collect();

    // 3. Determine what to load and unload.
    //    We collect into Vecs to avoid borrowing StreamingState while mutating world.
    let (to_load, to_unload) = {
        let state = world
            .resource::<StreamingState>()
            .expect("StreamingState resource missing");

        let to_load: Vec<(i32, i32)> = desired
            .iter()
            .filter(|id| !state.is_loaded(**id))
            .copied()
            .collect();

        let unload_radius = config.unload_radius;
        let to_unload: Vec<(i32, i32)> = state
            .loaded
            .keys()
            .filter(|id| {
                let dx = (id.0 - camera_chunk.0).abs();
                let dz = (id.1 - camera_chunk.1).abs();
                dx > unload_radius || dz > unload_radius
            })
            .copied()
            .collect();

        (to_load, to_unload)
    };

    // 4. Unload: despawn chunk entities and remove from state.
    for chunk_id in &to_unload {
        let chunk = {
            let state = world
                .resource_mut::<StreamingState>()
                .expect("StreamingState resource missing");
            state.loaded.remove(chunk_id)
        };
        if let Some(chunk) = chunk {
            for entity in &chunk.entities {
                world.despawn(*entity);
            }
        }
    }

    // 5. Load: use ChunkLoader (or NullChunkLoader) to get data, spawn entities.
    //    We temporarily remove the loader to avoid borrow conflicts.
    //    `catch_unwind` ensures the loader is always re-inserted even if a
    //    load callback panics, preventing the resource from being lost.
    let loader_data: Vec<((i32, i32), Option<ChunkData>)> =
        if let Some(loader) = world.remove_resource::<Arc<dyn ChunkLoader>>() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                to_load.iter().map(|&id| (id, loader.load(id))).collect()
            }));
            match result {
                Ok(data) => {
                    world.insert_resource(loader);
                    data
                }
                Err(payload) => {
                    // Always re-insert the loader before propagating.
                    world.insert_resource(loader);
                    std::panic::resume_unwind(payload);
                }
            }
        } else {
            // No loader registered: produce empty chunks.
            let null = NullChunkLoader;
            to_load.iter().map(|&id| (id, null.load(id))).collect()
        };

    for (chunk_id, data) in loader_data {
        let mut spawned_entities = Vec::new();

        if let Some(chunk_data) = data {
            let chunk_origin = Vec3::new(
                chunk_id.0 as f32 * config.chunk_size,
                0.0,
                chunk_id.1 as f32 * config.chunk_size,
            );

            for entity_data in &chunk_data.entities {
                let world_pos = chunk_origin + entity_data.position;
                let transform = euca_math::Transform::from_translation(world_pos);
                let entity = world.spawn(LocalTransform(transform));
                world.insert(entity, GlobalTransform(transform));
                spawned_entities.push(entity);
            }
        }

        let chunk = WorldChunk {
            chunk_id,
            entities: spawned_entities,
            loaded: true,
        };
        let state = world
            .resource_mut::<StreamingState>()
            .expect("StreamingState resource missing");
        state.loaded.insert(chunk_id, chunk);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn set_camera(world: &mut World, pos: Vec3) {
        world.insert_resource(CameraPosition(pos));
    }

    #[test]
    fn world_to_chunk_positive_position() {
        assert_eq!(world_to_chunk(Vec3::new(100.0, 0.0, 200.0), 64.0), (1, 3));
    }

    #[test]
    fn world_to_chunk_negative_position() {
        assert_eq!(
            world_to_chunk(Vec3::new(-10.0, 0.0, -130.0), 64.0),
            (-1, -3)
        );
    }

    #[test]
    fn world_to_chunk_on_boundary() {
        assert_eq!(world_to_chunk(Vec3::new(64.0, 0.0, 0.0), 64.0), (1, 0));
        assert_eq!(world_to_chunk(Vec3::new(63.9, 0.0, 0.0), 64.0), (0, 0));
    }

    #[test]
    fn chunks_in_radius_count() {
        let chunks = chunks_in_radius((0, 0), 1);
        assert_eq!(chunks.len(), 9);
        assert!(chunks.contains(&(0, 0)));
        assert!(chunks.contains(&(-1, -1)));
        assert!(chunks.contains(&(1, 1)));
    }

    #[test]
    fn chunks_in_radius_zero() {
        let chunks = chunks_in_radius((5, 5), 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], (5, 5));
    }

    #[test]
    fn system_loads_chunks_within_radius() {
        let mut world = World::new();
        world.insert_resource(StreamingConfig {
            chunk_size: 64.0,
            load_radius: 1,
            unload_radius: 3,
        });
        set_camera(&mut world, Vec3::ZERO);

        streaming_update_system(&mut world);

        let state = world.resource::<StreamingState>().unwrap();
        assert_eq!(state.loaded_count(), 9);
        assert!(state.is_loaded((0, 0)));
        assert!(state.is_loaded((-1, -1)));
        assert!(state.is_loaded((1, 1)));
        assert!(!state.is_loaded((2, 0)));
    }

    #[test]
    fn system_unloads_chunks_beyond_radius() {
        let mut world = World::new();
        world.insert_resource(StreamingConfig {
            chunk_size: 64.0,
            load_radius: 1,
            unload_radius: 2,
        });

        set_camera(&mut world, Vec3::ZERO);
        streaming_update_system(&mut world);
        assert_eq!(
            world.resource::<StreamingState>().unwrap().loaded_count(),
            9
        );

        set_camera(&mut world, Vec3::new(640.0, 0.0, 0.0));
        streaming_update_system(&mut world);

        let state = world.resource::<StreamingState>().unwrap();
        assert!(!state.is_loaded((0, 0)));
        assert!(state.is_loaded((10, 0)));
        assert!(state.is_loaded((9, 0)));
        assert!(state.is_loaded((11, 0)));
    }

    #[test]
    fn camera_movement_triggers_incremental_load() {
        let mut world = World::new();
        world.insert_resource(StreamingConfig {
            chunk_size: 64.0,
            load_radius: 1,
            unload_radius: 3,
        });

        set_camera(&mut world, Vec3::ZERO);
        streaming_update_system(&mut world);
        assert_eq!(
            world.resource::<StreamingState>().unwrap().loaded_count(),
            9
        );

        set_camera(&mut world, Vec3::new(64.0, 0.0, 0.0));
        streaming_update_system(&mut world);

        let state = world.resource::<StreamingState>().unwrap();
        assert!(state.is_loaded((2, 0)));
        assert!(state.is_loaded((2, 1)));
        assert!(state.is_loaded((2, -1)));
        assert!(state.is_loaded((0, 0)));
    }

    #[test]
    fn chunk_loader_spawns_entities() {
        struct TestLoader;
        impl ChunkLoader for TestLoader {
            fn load(&self, chunk_id: (i32, i32)) -> Option<ChunkData> {
                Some(ChunkData {
                    chunk_id,
                    entities: vec![ChunkEntityData {
                        position: Vec3::new(32.0, 0.0, 32.0),
                        name: Some("test_entity".into()),
                    }],
                })
            }
            fn unload(&self, _chunk_id: (i32, i32)) {}
        }

        let mut world = World::new();
        world.insert_resource(StreamingConfig {
            chunk_size: 64.0,
            load_radius: 0,
            unload_radius: 2,
        });
        world.insert_resource(Arc::new(TestLoader) as Arc<dyn ChunkLoader>);
        set_camera(&mut world, Vec3::ZERO);

        streaming_update_system(&mut world);

        let state = world.resource::<StreamingState>().unwrap();
        assert!(state.is_loaded((0, 0)));
        let chunk = state.loaded.get(&(0, 0)).unwrap();
        assert_eq!(chunk.entities.len(), 1);

        let entity = chunk.entities[0];
        let gt = world.get::<GlobalTransform>(entity).unwrap();
        assert!((gt.0.translation.x - 32.0).abs() < 1e-5);
        assert!((gt.0.translation.z - 32.0).abs() < 1e-5);
    }

    #[test]
    fn unload_despawns_entities_from_world() {
        struct OneEntityLoader;
        impl ChunkLoader for OneEntityLoader {
            fn load(&self, chunk_id: (i32, i32)) -> Option<ChunkData> {
                Some(ChunkData {
                    chunk_id,
                    entities: vec![ChunkEntityData {
                        position: Vec3::ZERO,
                        name: None,
                    }],
                })
            }
            fn unload(&self, _chunk_id: (i32, i32)) {}
        }

        let mut world = World::new();
        world.insert_resource(StreamingConfig {
            chunk_size: 64.0,
            load_radius: 0,
            unload_radius: 1,
        });
        world.insert_resource(Arc::new(OneEntityLoader) as Arc<dyn ChunkLoader>);

        set_camera(&mut world, Vec3::ZERO);
        streaming_update_system(&mut world);

        let entity = world
            .resource::<StreamingState>()
            .unwrap()
            .loaded
            .get(&(0, 0))
            .unwrap()
            .entities[0];
        assert!(world.is_alive(entity));

        set_camera(&mut world, Vec3::new(640.0, 0.0, 0.0));
        streaming_update_system(&mut world);

        assert!(!world.is_alive(entity));
        assert!(
            !world
                .resource::<StreamingState>()
                .unwrap()
                .is_loaded((0, 0))
        );
    }

    #[test]
    fn default_config_values() {
        let config = StreamingConfig::default();
        assert_eq!(config.chunk_size, 64.0);
        assert_eq!(config.load_radius, 3);
        assert_eq!(config.unload_radius, 5);
    }

    #[test]
    fn system_creates_default_resources() {
        let mut world = World::new();
        streaming_update_system(&mut world);

        assert!(world.resource::<StreamingConfig>().is_some());
        assert!(world.resource::<StreamingState>().is_some());
    }
}
