//! ECS component describing a terrain entity.

use crate::heightmap::Heightmap;
use crate::splat::SplatMap;

/// Opaque handle to a texture layer (engine-specific).
///
/// In a real renderer this would reference a GPU texture; here we store
/// a lightweight identifier so the terrain crate stays render-agnostic.
#[derive(Clone, Debug, Default)]
pub struct TextureHandle(pub u64);

/// Describes a single terrain texture layer for splatting.
#[derive(Clone, Debug)]
pub struct TerrainLayer {
    /// Texture identifier (diffuse / albedo).
    pub texture: TextureHandle,
    /// UV tiling scale — higher values repeat the texture more often.
    pub uv_scale: f32,
}

impl Default for TerrainLayer {
    fn default() -> Self {
        Self {
            texture: TextureHandle::default(),
            uv_scale: 1.0,
        }
    }
}

/// Maximum number of blendable texture layers per terrain.
pub const MAX_TERRAIN_LAYERS: usize = 4;

/// ECS component that fully describes one terrain entity.
///
/// Attach this to an entity alongside a `Transform`; terrain systems will
/// read it to generate chunks, LOD meshes, physics colliders, and so on.
#[derive(Clone, Debug)]
pub struct TerrainComponent {
    /// The elevation data.
    pub heightmap: Heightmap,
    /// Per-vertex blend weights for up to 4 texture layers.
    pub splat_map: SplatMap,
    /// Up to 4 texture layers blended via the splat map.
    pub layers: [TerrainLayer; MAX_TERRAIN_LAYERS],
    /// Side length (in grid cells) of each chunk for culling and LOD.
    pub chunk_size: u32,
}

impl TerrainComponent {
    /// Create a new terrain component from a heightmap.
    ///
    /// `chunk_size` specifies how many grid cells each chunk spans along one
    /// axis.  A reasonable default is 32.
    pub fn new(heightmap: Heightmap, chunk_size: u32) -> Self {
        let splat_map = SplatMap::uniform(heightmap.width, heightmap.height);
        Self {
            heightmap,
            splat_map,
            layers: Default::default(),
            chunk_size: chunk_size.max(2),
        }
    }

    /// Builder helper: set a texture layer.
    pub fn with_layer(mut self, index: usize, layer: TerrainLayer) -> Self {
        if index < MAX_TERRAIN_LAYERS {
            self.layers[index] = layer;
        }
        self
    }
}
