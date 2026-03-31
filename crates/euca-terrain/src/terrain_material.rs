//! Terrain material pipeline — maps [`SurfaceType`] to PBR [`Material`]s with
//! textures.
//!
//! [`TerrainMaterialSet`] is the engine-level resource that games configure to
//! define what each surface type looks like. The terrain renderer consults it
//! when spawning terrain chunks.
//!
//! # Usage
//!
//! ```ignore
//! let mut materials = TerrainMaterialSet::new();
//!
//! // Option 1: Load textures from disk
//! let grass_tex = renderer.upload_texture_from_file(gpu, "assets/textures/terrain/grass.png");
//! materials.set(SurfaceType::Grass, TerrainSurfaceMaterial {
//!     albedo_texture: Some(grass_tex),
//!     ..Default::default()
//! });
//!
//! // Option 2: Just set a color (fallback)
//! materials.set(SurfaceType::Water, TerrainSurfaceMaterial {
//!     base_color: [0.15, 0.35, 0.65, 0.9],
//!     metallic: 0.3,
//!     roughness: 0.2,
//!     ..Default::default()
//! });
//!
//! // Spawn terrain with materials
//! spawn_terrain(world, gpu, renderer, &level, &materials);
//! ```

use std::collections::HashMap;
use std::path::Path;

use crate::level_data::SurfaceType;

// ---------------------------------------------------------------------------
// Surface material definition
// ---------------------------------------------------------------------------

/// PBR material properties for a terrain surface type.
///
/// Textures are optional — when absent, the base color/metallic/roughness
/// values are used as flat colors (the existing fallback behavior).
#[derive(Clone, Debug)]
pub struct TerrainSurfaceMaterial {
    /// Base albedo color (multiplied with albedo texture if present).
    pub base_color: [f32; 4],
    /// Metallic factor (0.0 = dielectric, 1.0 = metal).
    pub metallic: f32,
    /// Roughness factor (0.0 = mirror, 1.0 = rough).
    pub roughness: f32,
    /// Path to albedo texture on disk (PNG/JPEG). Loaded on demand.
    pub albedo_texture_path: Option<String>,
    /// Path to normal map on disk. Loaded on demand.
    pub normal_texture_path: Option<String>,
    /// UV tiling factor — how many times the texture repeats per terrain cell.
    /// Default: 1.0 (one tile per cell). Use 4.0 for grass to avoid stretching.
    pub uv_scale: f32,
}

impl Default for TerrainSurfaceMaterial {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.8,
            albedo_texture_path: None,
            normal_texture_path: None,
            uv_scale: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// TerrainMaterialSet
// ---------------------------------------------------------------------------

/// Maps [`SurfaceType`] → [`TerrainSurfaceMaterial`].
///
/// Games configure this set to define how each terrain surface looks. The
/// terrain rendering pipeline consults it when spawning terrain chunk entities.
///
/// Falls back to [`crate::level_render::surface_color()`] for surface types
/// that have no entry.
#[derive(Clone, Debug, Default)]
pub struct TerrainMaterialSet {
    materials: HashMap<SurfaceType, TerrainSurfaceMaterial>,
}

impl TerrainMaterialSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the material for a surface type.
    pub fn set(&mut self, surface: SurfaceType, material: TerrainSurfaceMaterial) {
        self.materials.insert(surface, material);
    }

    /// Get the material for a surface type, or `None` if not configured.
    pub fn get(&self, surface: SurfaceType) -> Option<&TerrainSurfaceMaterial> {
        self.materials.get(&surface)
    }

    /// Get the UV scale for a surface type (defaults to 1.0).
    pub fn uv_scale(&self, surface: SurfaceType) -> f32 {
        self.materials
            .get(&surface)
            .map(|m| m.uv_scale)
            .unwrap_or(1.0)
    }

    /// Configure a surface type from a texture directory.
    ///
    /// Looks for `{dir}/{name}.png` (albedo) and `{dir}/{name}_normal.png`
    /// (normal map). If found, sets the paths. If not, leaves as fallback color.
    pub fn configure_from_dir(
        &mut self,
        surface: SurfaceType,
        dir: &str,
        name: &str,
        base_color: [f32; 4],
        metallic: f32,
        roughness: f32,
        uv_scale: f32,
    ) {
        let albedo_path = format!("{dir}/{name}.png");
        let normal_path = format!("{dir}/{name}_normal.png");

        let albedo_exists = Path::new(&albedo_path).exists();
        let normal_exists = Path::new(&normal_path).exists();

        self.set(
            surface,
            TerrainSurfaceMaterial {
                base_color,
                metallic,
                roughness,
                albedo_texture_path: if albedo_exists {
                    Some(albedo_path)
                } else {
                    None
                },
                normal_texture_path: if normal_exists {
                    Some(normal_path)
                } else {
                    None
                },
                uv_scale,
            },
        );
    }

    /// Create a default material set with standard terrain textures.
    ///
    /// Looks for textures in `texture_dir` (e.g. `"assets/textures/terrain"`).
    /// Falls back to flat colors for any missing textures.
    pub fn with_standard_textures(texture_dir: &str) -> Self {
        let mut set = Self::new();

        set.configure_from_dir(
            SurfaceType::Grass,
            texture_dir,
            "grass",
            [0.3, 0.6, 0.15, 1.0],
            0.0,
            0.9,
            4.0, // Grass tiles 4x per cell
        );
        set.configure_from_dir(
            SurfaceType::Dirt,
            texture_dir,
            "dirt",
            [0.55, 0.37, 0.20, 1.0],
            0.0,
            0.85,
            4.0,
        );
        set.configure_from_dir(
            SurfaceType::Stone,
            texture_dir,
            "stone",
            [0.5, 0.5, 0.5, 1.0],
            0.0,
            0.7,
            2.0, // Stone tiles 2x per cell
        );
        set.configure_from_dir(
            SurfaceType::Water,
            texture_dir,
            "water",
            [0.15, 0.35, 0.65, 0.9],
            0.3,
            0.15,
            2.0,
        );
        set.configure_from_dir(
            SurfaceType::Sand,
            texture_dir,
            "sand",
            [0.85, 0.78, 0.55, 1.0],
            0.0,
            0.9,
            4.0,
        );
        set.configure_from_dir(
            SurfaceType::Snow,
            texture_dir,
            "snow",
            [0.95, 0.95, 0.97, 1.0],
            0.0,
            0.85,
            4.0,
        );
        set.configure_from_dir(
            SurfaceType::Mud,
            texture_dir,
            "mud",
            [0.40, 0.30, 0.20, 1.0],
            0.0,
            0.9,
            4.0,
        );
        set.configure_from_dir(
            SurfaceType::Road,
            texture_dir,
            "road",
            [0.35, 0.35, 0.35, 1.0],
            0.0,
            0.7,
            2.0,
        );
        set.configure_from_dir(
            SurfaceType::Cliff,
            texture_dir,
            "cliff",
            [0.60, 0.55, 0.50, 1.0],
            0.0,
            0.8,
            2.0,
        );
        set.configure_from_dir(
            SurfaceType::Void,
            texture_dir,
            "void",
            [0.05, 0.05, 0.05, 1.0],
            0.0,
            1.0,
            1.0,
        );

        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_set_is_empty() {
        let set = TerrainMaterialSet::new();
        assert!(set.get(SurfaceType::Grass).is_none());
    }

    #[test]
    fn set_and_get() {
        let mut set = TerrainMaterialSet::new();
        set.set(
            SurfaceType::Grass,
            TerrainSurfaceMaterial {
                base_color: [0.2, 0.5, 0.1, 1.0],
                ..Default::default()
            },
        );
        let mat = set.get(SurfaceType::Grass).unwrap();
        assert!((mat.base_color[0] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn uv_scale_defaults_to_one() {
        let set = TerrainMaterialSet::new();
        assert!((set.uv_scale(SurfaceType::Grass) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn standard_textures_has_all_types() {
        // Uses a non-existent dir so all textures fall back to colors.
        let set = TerrainMaterialSet::with_standard_textures("/nonexistent");
        assert!(set.get(SurfaceType::Grass).is_some());
        assert!(set.get(SurfaceType::Water).is_some());
        assert!(set.get(SurfaceType::Stone).is_some());
        // Paths should be None since directory doesn't exist.
        assert!(set.get(SurfaceType::Grass).unwrap().albedo_texture_path.is_none());
    }
}
