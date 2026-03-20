//! Terrain mesh generation from a heightmap.
//!
//! Produces position, normal, UV, and splat-weight attributes suitable for
//! GPU rendering.  The generated mesh uses a regular grid of triangles with
//! per-vertex normals derived from the heightmap gradient.

use euca_math::{Vec2, Vec3, Vec4};

use crate::heightmap::Heightmap;

/// A single terrain vertex.
#[derive(Clone, Copy, Debug)]
pub struct TerrainVertex {
    /// World-space position.
    pub position: Vec3,
    /// Surface normal.
    pub normal: Vec3,
    /// Texture coordinate.
    pub uv: Vec2,
    /// Per-layer blend weights for texture splatting (4 layers).
    pub splat_weights: Vec4,
}

/// A generated terrain mesh (CPU-side).
#[derive(Clone, Debug)]
pub struct TerrainMesh {
    pub vertices: Vec<TerrainVertex>,
    /// Triangle indices (three per triangle).
    pub indices: Vec<u32>,
}

/// Generate a full-resolution mesh from a heightmap.
///
/// The mesh covers the entire heightmap footprint on the XZ plane.
/// Splat weights default to layer 0 fully opaque; use the splatting API
/// to paint layers after generation.
pub fn generate_terrain_mesh(heightmap: &Heightmap) -> TerrainMesh {
    generate_terrain_mesh_region(heightmap, 0, 0, heightmap.width, heightmap.height, 1)
}

/// Generate a mesh for a rectangular sub-region of the heightmap.
///
/// `col_start` / `row_start` are inclusive grid indices.
/// `col_end` / `row_end` are exclusive grid indices.
/// `step` controls vertex skipping for LOD (1 = full resolution, 2 = half, etc.).
pub fn generate_terrain_mesh_region(
    heightmap: &Heightmap,
    col_start: u32,
    row_start: u32,
    col_end: u32,
    row_end: u32,
    step: u32,
) -> TerrainMesh {
    let step = step.max(1);
    let col_end = col_end.min(heightmap.width);
    let row_end = row_end.min(heightmap.height);

    let inv_w = if heightmap.width > 1 {
        1.0 / (heightmap.width - 1) as f32
    } else {
        1.0
    };
    let inv_h = if heightmap.height > 1 {
        1.0 / (heightmap.height - 1) as f32
    } else {
        1.0
    };

    // Collect grid coordinates we will use.
    let cols: Vec<u32> = (col_start..col_end).step_by(step as usize).collect();
    let rows: Vec<u32> = (row_start..row_end).step_by(step as usize).collect();

    // Ensure the last column/row is included if step doesn't land on it.
    let cols = ensure_last(cols, col_end.saturating_sub(1));
    let rows = ensure_last(rows, row_end.saturating_sub(1));

    let num_cols = cols.len();
    let num_rows = rows.len();

    let mut vertices = Vec::with_capacity(num_cols * num_rows);

    for &row in &rows {
        for &col in &cols {
            let x = col as f32 * heightmap.cell_size;
            let z = row as f32 * heightmap.cell_size;
            let y = heightmap.sample(x, z);

            let position = Vec3::new(x, y, z);
            let normal = heightmap.normal_at(x, z);
            let uv = Vec2::new(col as f32 * inv_w, row as f32 * inv_h);

            // Default splat: 100 % layer 0.
            let splat_weights = Vec4::new(1.0, 0.0, 0.0, 0.0);

            vertices.push(TerrainVertex {
                position,
                normal,
                uv,
                splat_weights,
            });
        }
    }

    // Build triangle indices (two triangles per quad).
    let mut indices = Vec::with_capacity((num_cols - 1) * (num_rows - 1) * 6);
    for r in 0..(num_rows - 1) {
        for c in 0..(num_cols - 1) {
            let tl = (r * num_cols + c) as u32;
            let tr = tl + 1;
            let bl = tl + num_cols as u32;
            let br = bl + 1;

            // First triangle (top-left, bottom-left, top-right).
            indices.push(tl);
            indices.push(bl);
            indices.push(tr);

            // Second triangle (top-right, bottom-left, bottom-right).
            indices.push(tr);
            indices.push(bl);
            indices.push(br);
        }
    }

    TerrainMesh { vertices, indices }
}

/// Ensure the last value in a sorted vec is `last_val` (append if missing).
fn ensure_last(mut v: Vec<u32>, last_val: u32) -> Vec<u32> {
    if v.last().copied() != Some(last_val) {
        v.push(last_val);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_from_flat_heightmap() {
        let hm = Heightmap::flat(3, 3).with_cell_size(1.0);
        let mesh = generate_terrain_mesh(&hm);

        // 3x3 grid => 9 vertices, 2x2 quads => 8 triangles => 24 indices.
        assert_eq!(mesh.vertices.len(), 9);
        assert_eq!(mesh.indices.len(), 24);

        // All Y positions should be 0.
        for v in &mesh.vertices {
            assert!((v.position.y).abs() < 1e-6);
        }
    }

    #[test]
    fn mesh_indices_within_bounds() {
        let hm = Heightmap::flat(5, 4).with_cell_size(2.0);
        let mesh = generate_terrain_mesh(&hm);

        let max_idx = mesh.vertices.len() as u32;
        for &idx in &mesh.indices {
            assert!(idx < max_idx, "Index {idx} out of bounds (max {max_idx})");
        }
    }

    #[test]
    fn mesh_region_with_step() {
        let hm = Heightmap::flat(9, 9).with_cell_size(1.0);
        // Step 2: should pick cols [0,2,4,6,8], rows [0,2,4,6,8] => 5x5 = 25 vertices.
        let mesh = generate_terrain_mesh_region(&hm, 0, 0, 9, 9, 2);
        assert_eq!(mesh.vertices.len(), 25);
    }

    #[test]
    fn mesh_preserves_heights() {
        // 3x2 ramp along X (two identical rows).
        let hm = Heightmap::from_raw(3, 2, vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
            .with_cell_size(1.0)
            .with_max_height(10.0);
        let mesh = generate_terrain_mesh(&hm);

        // Row 0: vertices 0, 1, 2.
        assert!((mesh.vertices[0].position.y - 0.0).abs() < 1e-4);
        assert!((mesh.vertices[1].position.y - 5.0).abs() < 1e-4);
        assert!((mesh.vertices[2].position.y - 10.0).abs() < 1e-4);
    }
}
