//! Terrain and heightmap system for EucaEngine.
//!
//! Provides a complete outdoor terrain pipeline: heightmap sampling, chunked
//! mesh generation, quad-tree LOD, texture splatting, physics colliders, and
//! runtime editing tools.
//!
//! # Quick start
//! ```ignore
//! use euca_terrain::*;
//!
//! // Create a flat 128x128 heightmap.
//! let heightmap = Heightmap::flat(128, 128)
//!     .with_cell_size(1.0)
//!     .with_max_height(50.0);
//!
//! // Build a terrain component.
//! let terrain = TerrainComponent::new(heightmap, 32);
//!
//! // Generate chunks and meshes.
//! let chunks = build_chunks(&terrain.heightmap, terrain.chunk_size);
//! for chunk in &chunks {
//!     let lod = select_chunk_lod(chunk, camera_pos, &LodConfig::default());
//!     let mesh = generate_chunk_mesh(&terrain.heightmap, chunk, lod.step);
//!     // ... upload mesh to GPU ...
//! }
//! ```

pub mod chunk;
pub mod component;
pub mod editing;
#[cfg(feature = "gpu-terrain")]
pub mod gpu_terrain;
pub mod heightmap;
pub mod lod;
pub mod mesh;
pub mod physics;
pub mod splat;

// Re-export the most commonly used items at crate root.
pub use chunk::{TerrainChunk, aabb_in_frustum, build_chunks, generate_chunk_mesh};
pub use component::{MAX_TERRAIN_LAYERS, TerrainComponent, TerrainLayer, TextureHandle};
pub use editing::{flatten_terrain, lower_terrain, paint_splat, raise_terrain, smooth_terrain};
pub use heightmap::Heightmap;
pub use lod::{ChunkLod, LodConfig, select_all_lods, select_chunk_lod};
pub use mesh::{TerrainMesh, TerrainVertex, generate_terrain_mesh};
pub use physics::{HeightfieldTile, generate_heightfield_colliders, height_at};
pub use splat::SplatMap;
