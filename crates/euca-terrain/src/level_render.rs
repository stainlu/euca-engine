//! Terrain-to-mesh renderer: converts level data into renderable 3D meshes
//! grouped by surface type.
//!
//! The canonical entry point is [`generate_mesh_from_level`] which takes a
//! [`LevelData`] and respects its `interpolate_height` flag:
//!
//! * `true`  → bilinear height interpolation (smooth terrain)
//! * `false` → flat per-cell height (tile-like surfaces)
//!
//! Lower-level helpers [`generate_terrain_meshes`] and
//! [`generate_terrain_mesh_simple`] operate on raw [`Heightmap`] + surface map
//! and always interpolate.

use std::collections::HashMap;

use euca_render::{Mesh, Vertex};

use crate::heightmap::Heightmap;
use crate::level_data::{LevelData, SurfaceType};

// ---------------------------------------------------------------------------
// Surface color mapping
// ---------------------------------------------------------------------------

/// Return an RGBA color representative of the given surface type.
pub fn surface_color(surface: SurfaceType) -> [f32; 4] {
    match surface {
        SurfaceType::Grass => [0.30, 0.60, 0.15, 1.0],
        SurfaceType::Dirt => [0.55, 0.37, 0.20, 1.0],
        SurfaceType::Stone => [0.50, 0.50, 0.50, 1.0],
        SurfaceType::Water => [0.15, 0.35, 0.65, 1.0],
        SurfaceType::Sand => [0.85, 0.78, 0.55, 1.0],
        SurfaceType::Snow => [0.95, 0.95, 0.97, 1.0],
        SurfaceType::Mud => [0.40, 0.30, 0.20, 1.0],
        SurfaceType::Road => [0.35, 0.35, 0.35, 1.0],
        SurfaceType::Cliff => [0.60, 0.55, 0.50, 1.0],
        SurfaceType::Void => [0.05, 0.05, 0.05, 1.0],
        SurfaceType::Custom(_) => [0.70, 0.30, 0.70, 1.0], // magenta for custom
    }
}

// ---------------------------------------------------------------------------
// Render chunk output
// ---------------------------------------------------------------------------

/// A renderable terrain chunk associated with a specific surface type.
pub struct TerrainRenderChunk {
    /// The surface type this chunk represents.
    pub surface: SurfaceType,
    /// GPU-ready mesh geometry.
    pub mesh: Mesh,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a cell quad with bilinearly interpolated corner heights.
fn build_cell_quad_smooth(
    heightmap: &Heightmap,
    col: u32,
    row: u32,
    base_vertex: u32,
) -> ([Vertex; 4], [u32; 6]) {
    let cs = heightmap.cell_size;

    let x0 = col as f32 * cs;
    let z0 = row as f32 * cs;
    let x1 = (col + 1) as f32 * cs;
    let z1 = (row + 1) as f32 * cs;

    let y00 = heightmap.sample(x0, z0);
    let y10 = heightmap.sample(x1, z0);
    let y01 = heightmap.sample(x0, z1);
    let y11 = heightmap.sample(x1, z1);

    let n = heightmap.normal_at((x0 + x1) * 0.5, (z0 + z1) * 0.5);
    let normal = [n.x, n.y, n.z];
    let tangent = [1.0, 0.0, 0.0];

    build_quad(
        x0,
        z0,
        x1,
        z1,
        y00,
        y10,
        y01,
        y11,
        normal,
        tangent,
        base_vertex,
    )
}

/// Build a cell quad where all four corners share the same height (flat tile).
fn build_cell_quad_flat(
    heightmap: &Heightmap,
    col: u32,
    row: u32,
    base_vertex: u32,
) -> ([Vertex; 4], [u32; 6]) {
    let cs = heightmap.cell_size;

    let x0 = col as f32 * cs;
    let z0 = row as f32 * cs;
    let x1 = (col + 1) as f32 * cs;
    let z1 = (row + 1) as f32 * cs;

    // Use the raw cell value (no interpolation) — all 4 corners at same height.
    let y = heightmap.raw_at(col, row) * heightmap.max_height;

    let normal = [0.0, 1.0, 0.0]; // Flat surface → straight up.
    let tangent = [1.0, 0.0, 0.0];

    build_quad(x0, z0, x1, z1, y, y, y, y, normal, tangent, base_vertex)
}

/// Construct the 4 vertices + 6 indices for a quad with given corners.
///
/// `uv_scale` controls how many times the texture tiles across one cell.
/// Default 1.0 = one tile per cell. Use 4.0 for grass to avoid stretching.
#[allow(clippy::too_many_arguments)]
fn build_quad(
    x0: f32,
    z0: f32,
    x1: f32,
    z1: f32,
    y00: f32,
    y10: f32,
    y01: f32,
    y11: f32,
    normal: [f32; 3],
    tangent: [f32; 3],
    base_vertex: u32,
) -> ([Vertex; 4], [u32; 6]) {
    build_quad_uv(
        x0,
        z0,
        x1,
        z1,
        y00,
        y10,
        y01,
        y11,
        normal,
        tangent,
        base_vertex,
        1.0,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_quad_uv(
    x0: f32,
    z0: f32,
    x1: f32,
    z1: f32,
    y00: f32,
    y10: f32,
    y01: f32,
    y11: f32,
    normal: [f32; 3],
    tangent: [f32; 3],
    base_vertex: u32,
    uv_scale: f32,
) -> ([Vertex; 4], [u32; 6]) {
    let verts = [
        Vertex {
            position: [x0, y00, z0],
            normal,
            tangent,
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [x1, y10, z0],
            normal,
            tangent,
            uv: [uv_scale, 0.0],
        },
        Vertex {
            position: [x1, y11, z1],
            normal,
            tangent,
            uv: [uv_scale, uv_scale],
        },
        Vertex {
            position: [x0, y01, z1],
            normal,
            tangent,
            uv: [0.0, uv_scale],
        },
    ];
    let indices = [
        base_vertex,
        base_vertex + 1,
        base_vertex + 2,
        base_vertex,
        base_vertex + 2,
        base_vertex + 3,
    ];
    (verts, indices)
}

// ---------------------------------------------------------------------------
// Canonical entry point
// ---------------------------------------------------------------------------

/// Generate renderable terrain meshes from a [`LevelData`].
///
/// This is the **single renderer** for all game types. It respects:
/// - `interpolate_height`: smooth vs flat per-cell
/// - `max_height`: height scaling
/// - `surface`: per-cell surface type → one mesh per unique type
///
/// Returns one [`TerrainRenderChunk`] per unique surface type in the level.
pub fn generate_mesh_from_level(level: &LevelData) -> Vec<TerrainRenderChunk> {
    let heightmap = level.to_heightmap();
    let cell_cols = level.width.saturating_sub(1);
    let cell_rows = level.height.saturating_sub(1);
    let total_cells = (cell_cols * cell_rows) as usize;

    let mut buckets: HashMap<SurfaceType, (Vec<Vertex>, Vec<u32>)> = HashMap::new();

    for row in 0..cell_rows {
        for col in 0..cell_cols {
            let cell_idx = (row * cell_cols + col) as usize;
            let surface = if cell_idx < level.surface.len() {
                level.surface[cell_idx]
            } else {
                SurfaceType::Grass
            };

            let (verts_buf, indices_buf) = buckets.entry(surface).or_insert_with(|| {
                (
                    Vec::with_capacity(total_cells * 4),
                    Vec::with_capacity(total_cells * 6),
                )
            });

            let base = verts_buf.len() as u32;
            let (quad_verts, quad_indices) = if level.interpolate_height {
                build_cell_quad_smooth(&heightmap, col, row, base)
            } else {
                build_cell_quad_flat(&heightmap, col, row, base)
            };

            verts_buf.extend_from_slice(&quad_verts);
            indices_buf.extend_from_slice(&quad_indices);
        }
    }

    let mut chunks: Vec<TerrainRenderChunk> = buckets
        .into_iter()
        .map(|(surface, (vertices, indices))| TerrainRenderChunk {
            surface,
            mesh: Mesh { vertices, indices },
        })
        .collect();

    chunks.sort_by(|a, b| format!("{:?}", a.surface).cmp(&format!("{:?}", b.surface)));
    chunks
}

/// Generate terrain meshes with per-surface UV tiling from a
/// [`TerrainMaterialSet`].
///
/// Same as [`generate_mesh_from_level`] but uses the material set's UV scale
/// for each surface type so textures tile correctly.
pub fn generate_mesh_from_level_with_materials(
    level: &LevelData,
    materials: &crate::terrain_material::TerrainMaterialSet,
) -> Vec<TerrainRenderChunk> {
    let heightmap = level.to_heightmap();
    let cell_cols = level.width.saturating_sub(1);
    let cell_rows = level.height.saturating_sub(1);
    let total_cells = (cell_cols * cell_rows) as usize;

    let mut buckets: HashMap<SurfaceType, (Vec<Vertex>, Vec<u32>)> = HashMap::new();

    for row in 0..cell_rows {
        for col in 0..cell_cols {
            let cell_idx = (row * cell_cols + col) as usize;
            let surface = if cell_idx < level.surface.len() {
                level.surface[cell_idx]
            } else {
                SurfaceType::Grass
            };

            let uv_scale = materials.uv_scale(surface);

            let (verts_buf, indices_buf) = buckets.entry(surface).or_insert_with(|| {
                (
                    Vec::with_capacity(total_cells * 4),
                    Vec::with_capacity(total_cells * 6),
                )
            });

            let base = verts_buf.len() as u32;
            let cs = heightmap.cell_size;
            let x0 = col as f32 * cs;
            let z0 = row as f32 * cs;
            let x1 = (col + 1) as f32 * cs;
            let z1 = (row + 1) as f32 * cs;

            let (y00, y10, y01, y11, normal) = if level.interpolate_height {
                let y00 = heightmap.sample(x0, z0);
                let y10 = heightmap.sample(x1, z0);
                let y01 = heightmap.sample(x0, z1);
                let y11 = heightmap.sample(x1, z1);
                let n = heightmap.normal_at((x0 + x1) * 0.5, (z0 + z1) * 0.5);
                (y00, y10, y01, y11, [n.x, n.y, n.z])
            } else {
                let y = heightmap.raw_at(col, row) * heightmap.max_height;
                (y, y, y, y, [0.0, 1.0, 0.0])
            };

            let tangent = [1.0, 0.0, 0.0];
            let (quad_verts, quad_indices) = build_quad_uv(
                x0, z0, x1, z1, y00, y10, y01, y11, normal, tangent, base, uv_scale,
            );

            verts_buf.extend_from_slice(&quad_verts);
            indices_buf.extend_from_slice(&quad_indices);
        }
    }

    let mut chunks: Vec<TerrainRenderChunk> = buckets
        .into_iter()
        .map(|(surface, (vertices, indices))| TerrainRenderChunk {
            surface,
            mesh: Mesh { vertices, indices },
        })
        .collect();

    chunks.sort_by(|a, b| format!("{:?}", a.surface).cmp(&format!("{:?}", b.surface)));
    chunks
}

// ---------------------------------------------------------------------------
// Lower-level API (raw Heightmap, always interpolates)
// ---------------------------------------------------------------------------

/// Generate one [`Mesh`] per surface type from a raw heightmap + surface map.
///
/// Always uses bilinear interpolation. For the full pipeline that respects
/// `interpolate_height`, use [`generate_mesh_from_level`] instead.
pub fn generate_terrain_meshes(
    heightmap: &Heightmap,
    surface_map: Option<&[SurfaceType]>,
) -> Vec<TerrainRenderChunk> {
    let cell_cols = heightmap.width.saturating_sub(1);
    let cell_rows = heightmap.height.saturating_sub(1);
    let total_cells = (cell_cols * cell_rows) as usize;

    let mut buckets: HashMap<SurfaceType, (Vec<Vertex>, Vec<u32>)> = HashMap::new();

    for row in 0..cell_rows {
        for col in 0..cell_cols {
            let cell_idx = (row * cell_cols + col) as usize;
            let surface = surface_map
                .filter(|m| cell_idx < m.len())
                .map_or(SurfaceType::Grass, |m| m[cell_idx]);

            let (verts_buf, indices_buf) = buckets.entry(surface).or_insert_with(|| {
                (
                    Vec::with_capacity(total_cells * 4),
                    Vec::with_capacity(total_cells * 6),
                )
            });

            let base = verts_buf.len() as u32;
            let (quad_verts, quad_indices) = build_cell_quad_smooth(heightmap, col, row, base);

            verts_buf.extend_from_slice(&quad_verts);
            indices_buf.extend_from_slice(&quad_indices);
        }
    }

    let mut chunks: Vec<TerrainRenderChunk> = buckets
        .into_iter()
        .map(|(surface, (vertices, indices))| TerrainRenderChunk {
            surface,
            mesh: Mesh { vertices, indices },
        })
        .collect();

    chunks.sort_by(|a, b| format!("{:?}", a.surface).cmp(&format!("{:?}", b.surface)));
    chunks
}

/// Generate a single combined [`Mesh`] from the entire heightmap, ignoring
/// surface type distinctions.
pub fn generate_terrain_mesh_simple(heightmap: &Heightmap) -> Mesh {
    let cell_cols = heightmap.width.saturating_sub(1);
    let cell_rows = heightmap.height.saturating_sub(1);
    let total_cells = (cell_cols * cell_rows) as usize;

    let mut vertices = Vec::with_capacity(total_cells * 4);
    let mut indices = Vec::with_capacity(total_cells * 6);

    for row in 0..cell_rows {
        for col in 0..cell_cols {
            let base = vertices.len() as u32;
            let (quad_verts, quad_indices) = build_cell_quad_smooth(heightmap, col, row, base);
            vertices.extend_from_slice(&quad_verts);
            indices.extend_from_slice(&quad_indices);
        }
    }

    Mesh { vertices, indices }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_3x3() -> Heightmap {
        Heightmap::flat(3, 3).with_cell_size(1.0)
    }

    // -- generate_terrain_mesh_simple ------------------------------------------

    #[test]
    fn simple_mesh_vertex_and_index_counts() {
        let hm = flat_3x3();
        let mesh = generate_terrain_mesh_simple(&hm);
        assert_eq!(mesh.vertices.len(), 16);
        assert_eq!(mesh.indices.len(), 24);
    }

    #[test]
    fn simple_mesh_indices_in_bounds() {
        let hm = Heightmap::flat(5, 4).with_cell_size(2.0);
        let mesh = generate_terrain_mesh_simple(&hm);
        let max = mesh.vertices.len() as u32;
        for &idx in &mesh.indices {
            assert!(idx < max, "Index {idx} out of bounds ({max} vertices)");
        }
    }

    #[test]
    fn simple_mesh_flat_y_zero() {
        let hm = flat_3x3();
        let mesh = generate_terrain_mesh_simple(&hm);
        for v in &mesh.vertices {
            assert!(
                v.position[1].abs() < 1e-5,
                "Expected y ~ 0, got {}",
                v.position[1]
            );
        }
    }

    // -- generate_terrain_meshes -----------------------------------------------

    #[test]
    fn meshes_default_all_grass_when_no_surface_map() {
        let hm = flat_3x3();
        let chunks = generate_terrain_meshes(&hm, None);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].surface, SurfaceType::Grass);
        assert_eq!(chunks[0].mesh.vertices.len(), 16);
    }

    #[test]
    fn meshes_split_by_surface_type() {
        let hm = flat_3x3();
        let map = [
            SurfaceType::Grass,
            SurfaceType::Stone,
            SurfaceType::Stone,
            SurfaceType::Grass,
        ];
        let chunks = generate_terrain_meshes(&hm, Some(&map));
        assert_eq!(chunks.len(), 2);

        let grass = chunks
            .iter()
            .find(|c| c.surface == SurfaceType::Grass)
            .unwrap();
        let stone = chunks
            .iter()
            .find(|c| c.surface == SurfaceType::Stone)
            .unwrap();
        assert_eq!(grass.mesh.vertices.len(), 8);
        assert_eq!(stone.mesh.vertices.len(), 8);
    }

    #[test]
    fn meshes_indices_in_bounds_per_chunk() {
        let hm = Heightmap::flat(4, 4).with_cell_size(1.0);
        let map = vec![SurfaceType::Sand; 9];
        let chunks = generate_terrain_meshes(&hm, Some(&map));
        for chunk in &chunks {
            let max = chunk.mesh.vertices.len() as u32;
            for &idx in &chunk.mesh.indices {
                assert!(idx < max, "{:?} chunk: index {idx} >= {max}", chunk.surface);
            }
        }
    }

    // -- surface_color ---------------------------------------------------------

    #[test]
    fn surface_color_alpha_always_one() {
        let types = [
            SurfaceType::Grass,
            SurfaceType::Dirt,
            SurfaceType::Stone,
            SurfaceType::Sand,
            SurfaceType::Snow,
            SurfaceType::Water,
            SurfaceType::Mud,
            SurfaceType::Road,
            SurfaceType::Cliff,
            SurfaceType::Void,
            SurfaceType::Custom(42),
        ];
        for s in types {
            assert_eq!(surface_color(s)[3], 1.0, "{s:?} alpha should be 1.0");
        }
    }

    // -- generate_mesh_from_level (canonical entry point) ----------------------

    #[test]
    fn level_to_mesh_smooth() {
        let mut level = LevelData::new(3, 3, 1.0);
        level.interpolate_height = true;
        level.heightmap[0] = 0.5;
        let chunks = generate_mesh_from_level(&level);
        assert!(!chunks.is_empty());
        let total_verts: usize = chunks.iter().map(|c| c.mesh.vertices.len()).sum();
        assert_eq!(total_verts, 16); // 2x2 cells * 4 verts
    }

    #[test]
    fn level_to_mesh_flat() {
        let mut level = LevelData::new(3, 3, 1.0);
        level.interpolate_height = false;
        level.max_height = 10.0;
        level.heightmap[0] = 0.5; // Cell (0,0) height = 0.5 * 10 = 5.0
        let chunks = generate_mesh_from_level(&level);

        // All 4 corners of cell (0,0) should be at y = 5.0.
        let mesh = &chunks[0].mesh;
        let y_values: Vec<f32> = mesh
            .vertices
            .iter()
            .take(4)
            .map(|v| v.position[1])
            .collect();
        assert!(
            y_values.iter().all(|&y| (y - 5.0).abs() < 1e-3),
            "Flat mode: all corners should be at 5.0, got {y_values:?}"
        );
    }

    #[test]
    fn level_to_mesh_with_mixed_surfaces() {
        let mut level = LevelData::new(3, 3, 2.0);
        level.surface[0] = SurfaceType::Grass;
        level.surface[1] = SurfaceType::Water;
        level.surface[2] = SurfaceType::Grass;
        level.surface[3] = SurfaceType::Water;
        let chunks = generate_mesh_from_level(&level);
        assert_eq!(chunks.len(), 2, "Should have grass and water chunks");
    }

    // -- Height preservation ---------------------------------------------------

    #[test]
    fn simple_mesh_preserves_heights() {
        let hm = Heightmap::from_raw(3, 2, vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
            .with_cell_size(1.0)
            .with_max_height(10.0);
        let mesh = generate_terrain_mesh_simple(&hm);
        let y_origin = mesh.vertices[0].position[1];
        let y_mid = mesh.vertices[1].position[1];
        assert!(y_origin.abs() < 1e-3, "Expected y ~ 0, got {y_origin}");
        assert!((y_mid - 5.0).abs() < 1e-3, "Expected y ~ 5, got {y_mid}");
    }
}
