//! Terrain-to-mesh renderer: converts heightmap level data into renderable 3D
//! meshes grouped by surface type.
//!
//! Each cell of the heightmap is classified with a [`SurfaceType`] and assigned
//! a flat-shaded color via [`surface_color`].  Two generation strategies are
//! provided:
//!
//! * [`generate_terrain_meshes`] -- one [`euca_render::Mesh`] per surface type,
//!   suitable for draw-call batching by material.
//! * [`generate_terrain_mesh_simple`] -- a single combined mesh for quick
//!   previews or minimal-drawcall renderers.

use std::collections::HashMap;

use euca_render::{Mesh, Vertex};

use crate::heightmap::Heightmap;

// ---------------------------------------------------------------------------
// Surface classification
// ---------------------------------------------------------------------------

/// Classification of a terrain cell's surface material.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum SurfaceType {
    Grass = 0,
    Dirt = 1,
    Rock = 2,
    Sand = 3,
    Snow = 4,
    Water = 5,
}

/// Return an RGBA color (each channel in `[0.0, 1.0]`) representative of the
/// given surface type.  The alpha channel is always `1.0`.
pub fn surface_color(surface: SurfaceType) -> [f32; 4] {
    match surface {
        SurfaceType::Grass => [0.30, 0.60, 0.15, 1.0],
        SurfaceType::Dirt => [0.55, 0.37, 0.20, 1.0],
        SurfaceType::Rock => [0.50, 0.50, 0.50, 1.0],
        SurfaceType::Sand => [0.85, 0.78, 0.55, 1.0],
        SurfaceType::Snow => [0.95, 0.95, 0.97, 1.0],
        SurfaceType::Water => [0.15, 0.35, 0.65, 1.0],
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

/// Build the four vertices and six indices (two triangles) for one cell quad.
///
/// `col` / `row` are the top-left grid coordinates of the cell.  Heights are
/// sampled from `heightmap` at the four corners.  The returned index values are
/// already offset by `base_vertex`.
fn build_cell_quad(
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

    // Tangent aligned with the X (U) axis of the cell.
    let tangent = [1.0, 0.0, 0.0];

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
            uv: [1.0, 0.0],
        },
        Vertex {
            position: [x1, y11, z1],
            normal,
            tangent,
            uv: [1.0, 1.0],
        },
        Vertex {
            position: [x0, y01, z1],
            normal,
            tangent,
            uv: [0.0, 1.0],
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
// Public generation API
// ---------------------------------------------------------------------------

/// Generate one [`Mesh`] per [`SurfaceType`], grouping cells that share the
/// same surface classification into a single mesh for draw-call batching.
///
/// `surface_map` must have exactly `(heightmap.width - 1) * (heightmap.height - 1)`
/// entries laid out in row-major order, one per terrain cell.  If it is `None`,
/// every cell defaults to [`SurfaceType::Grass`].
///
/// Returns a list of [`TerrainRenderChunk`]s -- one per unique surface type
/// present in the map.
pub fn generate_terrain_meshes(
    heightmap: &Heightmap,
    surface_map: Option<&[SurfaceType]>,
) -> Vec<TerrainRenderChunk> {
    let cell_cols = heightmap.width.saturating_sub(1);
    let cell_rows = heightmap.height.saturating_sub(1);
    let total_cells = (cell_cols * cell_rows) as usize;

    // Accumulate vertices + indices per surface type.
    let mut buckets: HashMap<SurfaceType, (Vec<Vertex>, Vec<u32>)> = HashMap::new();

    for row in 0..cell_rows {
        for col in 0..cell_cols {
            let cell_idx = (row * cell_cols + col) as usize;
            let surface = surface_map
                .filter(|m| cell_idx < m.len())
                .map_or(SurfaceType::Grass, |m| m[cell_idx]);

            let (verts_buf, indices_buf) = buckets.entry(surface).or_insert_with(|| {
                // Pre-allocate for the worst case (all cells of one type).
                (
                    Vec::with_capacity(total_cells * 4),
                    Vec::with_capacity(total_cells * 6),
                )
            });

            let base = verts_buf.len() as u32;
            let (quad_verts, quad_indices) = build_cell_quad(heightmap, col, row, base);

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

    // Deterministic output order (useful for tests and snapshot comparisons).
    chunks.sort_by_key(|c| c.surface);
    chunks
}

/// Generate a single combined [`Mesh`] from the entire heightmap, ignoring
/// surface type distinctions.
///
/// This is the simplest path for quick terrain previews where material
/// batching is unnecessary.
pub fn generate_terrain_mesh_simple(heightmap: &Heightmap) -> Mesh {
    let cell_cols = heightmap.width.saturating_sub(1);
    let cell_rows = heightmap.height.saturating_sub(1);
    let total_cells = (cell_cols * cell_rows) as usize;

    let mut vertices = Vec::with_capacity(total_cells * 4);
    let mut indices = Vec::with_capacity(total_cells * 6);

    for row in 0..cell_rows {
        for col in 0..cell_cols {
            let base = vertices.len() as u32;
            let (quad_verts, quad_indices) = build_cell_quad(heightmap, col, row, base);
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

    /// Helper: create a flat 3x3 heightmap (2x2 = 4 cells).
    fn flat_3x3() -> Heightmap {
        Heightmap::flat(3, 3).with_cell_size(1.0)
    }

    // -- generate_terrain_mesh_simple tests ----------------------------------

    #[test]
    fn simple_mesh_vertex_and_index_counts() {
        let hm = flat_3x3();
        let mesh = generate_terrain_mesh_simple(&hm);

        // 2x2 cells, 4 verts each = 16 vertices; 6 indices each = 24 indices.
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
                "Expected y ~ 0 on flat map, got {}",
                v.position[1]
            );
        }
    }

    // -- generate_terrain_meshes tests (multi-surface) ----------------------

    #[test]
    fn meshes_default_all_grass_when_no_surface_map() {
        let hm = flat_3x3();
        let chunks = generate_terrain_meshes(&hm, None);

        assert_eq!(chunks.len(), 1, "All cells should coalesce into one chunk");
        assert_eq!(chunks[0].surface, SurfaceType::Grass);
        assert_eq!(chunks[0].mesh.vertices.len(), 16); // 4 cells * 4 verts
    }

    #[test]
    fn meshes_split_by_surface_type() {
        let hm = flat_3x3(); // 2x2 = 4 cells

        // Two grass, two rock cells (checkerboard).
        let map = [
            SurfaceType::Grass,
            SurfaceType::Rock,
            SurfaceType::Rock,
            SurfaceType::Grass,
        ];
        let chunks = generate_terrain_meshes(&hm, Some(&map));

        assert_eq!(chunks.len(), 2, "Should produce two batches");

        let grass = chunks
            .iter()
            .find(|c| c.surface == SurfaceType::Grass)
            .unwrap();
        let rock = chunks
            .iter()
            .find(|c| c.surface == SurfaceType::Rock)
            .unwrap();

        // Each batch covers 2 cells => 8 vertices, 12 indices.
        assert_eq!(grass.mesh.vertices.len(), 8);
        assert_eq!(rock.mesh.vertices.len(), 8);
        assert_eq!(grass.mesh.indices.len(), 12);
        assert_eq!(rock.mesh.indices.len(), 12);
    }

    #[test]
    fn meshes_indices_in_bounds_per_chunk() {
        let hm = Heightmap::flat(4, 4).with_cell_size(1.0); // 3x3 = 9 cells
        let map = vec![SurfaceType::Sand; 9];
        let chunks = generate_terrain_meshes(&hm, Some(&map));

        for chunk in &chunks {
            let max = chunk.mesh.vertices.len() as u32;
            for &idx in &chunk.mesh.indices {
                assert!(idx < max, "{:?} chunk: index {idx} >= {max}", chunk.surface);
            }
        }
    }

    // -- SurfaceType / surface_color tests -----------------------------------

    #[test]
    fn surface_color_alpha_always_one() {
        let all = [
            SurfaceType::Grass,
            SurfaceType::Dirt,
            SurfaceType::Rock,
            SurfaceType::Sand,
            SurfaceType::Snow,
            SurfaceType::Water,
        ];
        for s in all {
            let c = surface_color(s);
            assert_eq!(c[3], 1.0, "{s:?} alpha should be 1.0");
        }
    }

    #[test]
    fn surface_colors_are_distinct() {
        let all = [
            SurfaceType::Grass,
            SurfaceType::Dirt,
            SurfaceType::Rock,
            SurfaceType::Sand,
            SurfaceType::Snow,
            SurfaceType::Water,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(
                    surface_color(*a),
                    surface_color(*b),
                    "{a:?} and {b:?} should have distinct colors"
                );
            }
        }
    }

    // -- Height preservation -------------------------------------------------

    #[test]
    fn simple_mesh_preserves_heights() {
        // 3x2 ramp: heights increase along X.
        let hm = Heightmap::from_raw(3, 2, vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
            .with_cell_size(1.0)
            .with_max_height(10.0);
        let mesh = generate_terrain_mesh_simple(&hm);

        // First cell (col 0, row 0): top-left corner should be at y=0, top-right at y=5.
        let y_origin = mesh.vertices[0].position[1];
        let y_mid = mesh.vertices[1].position[1];
        assert!(y_origin.abs() < 1e-3, "Expected y ~ 0, got {y_origin}");
        assert!((y_mid - 5.0).abs() < 1e-3, "Expected y ~ 5, got {y_mid}");
    }
}
