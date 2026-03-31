//! Runtime loader for cooked `.emesh` assets produced by `euca-cook`.
//!
//! Cooked assets are pre-processed binary files that load instantly — no GLB
//! parsing, no mesh optimization, no texture conversion. Just deserialize and
//! upload to GPU.

use std::path::Path;

use euca_render::{Material, Mesh, Vertex};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Cooked format types (must match euca-cook's output)
// ---------------------------------------------------------------------------

/// A complete cooked mesh asset.
#[derive(Serialize, Deserialize)]
pub struct CookedMesh {
    pub name: String,
    pub lods: Vec<CookedLod>,
    pub textures: Vec<CookedTexture>,
    pub material: CookedMaterial,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub ground_offset: f32,
}

/// One LOD level.
#[derive(Serialize, Deserialize)]
pub struct CookedLod {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub vertex_count: u32,
    pub index_count: u32,
}

/// Pre-extracted texture (RGBA8).
#[derive(Serialize, Deserialize)]
pub struct CookedTexture {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Material properties.
#[derive(Serialize, Deserialize)]
pub struct CookedMaterial {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub albedo_tex_index: Option<usize>,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load a cooked `.emesh` file from disk.
pub fn load_cooked(path: &Path) -> Result<CookedMesh, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("Failed to read '{}': {e}", path.display()))?;
    bincode::deserialize(&bytes)
        .map_err(|e| format!("Failed to deserialize '{}': {e}", path.display()))
}

/// Convert a [`CookedLod`] into an engine [`Mesh`] ready for GPU upload.
pub fn lod_to_mesh(lod: &CookedLod) -> Mesh {
    let vertices: Vec<Vertex> = (0..lod.vertex_count as usize)
        .map(|i| Vertex {
            position: lod.positions[i],
            normal: lod.normals[i],
            tangent: [1.0, 0.0, 0.0], // Tangent not stored in cooked format yet
            uv: lod.uvs[i],
        })
        .collect();
    Mesh {
        vertices,
        indices: lod.indices.clone(),
    }
}

/// Convert a [`CookedMaterial`] into an engine [`Material`].
pub fn cooked_material_to_material(mat: &CookedMaterial) -> Material {
    Material::new(mat.albedo, mat.metallic, mat.roughness)
}

/// Select the appropriate LOD level based on a distance-to-camera metric.
///
/// Returns the LOD index (0 = highest detail, 3 = lowest).
/// `screen_size` is the approximate screen-space size of the object (0.0–1.0).
pub fn select_lod(lod_count: usize, screen_size: f32) -> usize {
    if lod_count <= 1 {
        return 0;
    }
    // Simple threshold-based selection
    let level = if screen_size > 0.15 {
        0 // Close up: full detail
    } else if screen_size > 0.05 {
        1 // Medium distance: 50% detail
    } else if screen_size > 0.02 {
        2 // Far: 25% detail
    } else {
        3 // Very far: 10% detail
    };
    level.min(lod_count - 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_lod_close() {
        assert_eq!(select_lod(4, 0.5), 0);
    }

    #[test]
    fn select_lod_medium() {
        assert_eq!(select_lod(4, 0.1), 1);
    }

    #[test]
    fn select_lod_far() {
        assert_eq!(select_lod(4, 0.03), 2);
    }

    #[test]
    fn select_lod_very_far() {
        assert_eq!(select_lod(4, 0.01), 3);
    }

    #[test]
    fn select_lod_clamps_to_available() {
        assert_eq!(select_lod(2, 0.01), 1);
        assert_eq!(select_lod(1, 0.01), 0);
    }

    #[test]
    fn lod_to_mesh_correct_counts() {
        let lod = CookedLod {
            positions: vec![[0.0; 3]; 4],
            normals: vec![[0.0, 1.0, 0.0]; 4],
            uvs: vec![[0.0; 2]; 4],
            indices: vec![0, 1, 2, 0, 2, 3],
            vertex_count: 4,
            index_count: 6,
        };
        let mesh = lod_to_mesh(&lod);
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices.len(), 6);
    }
}
