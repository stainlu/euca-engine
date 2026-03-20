//! Chunk-based terrain subdivision.
//!
//! The terrain heightmap is divided into a grid of equally-sized chunks.
//! Each chunk is an independent mesh that can be frustum-culled and rendered
//! at a different LOD level.

use euca_math::{Aabb, Vec3};

use crate::heightmap::Heightmap;
use crate::mesh::{TerrainMesh, generate_terrain_mesh_region};

/// Metadata describing one terrain chunk.
#[derive(Clone, Debug)]
pub struct TerrainChunk {
    /// Column index of this chunk in the chunk grid.
    pub chunk_col: u32,
    /// Row index of this chunk in the chunk grid.
    pub chunk_row: u32,
    /// First heightmap column (inclusive).
    pub col_start: u32,
    /// First heightmap row (inclusive).
    pub row_start: u32,
    /// One past the last heightmap column (exclusive, clamped to grid).
    pub col_end: u32,
    /// One past the last heightmap row (exclusive, clamped to grid).
    pub row_end: u32,
    /// World-space bounding box for frustum culling.
    pub bounds: Aabb,
}

/// Divide a heightmap into a grid of chunks.
///
/// `chunk_size` is the number of grid cells along one side of each chunk.
pub fn build_chunks(heightmap: &Heightmap, chunk_size: u32) -> Vec<TerrainChunk> {
    let chunk_size = chunk_size.max(2);
    let chunks_x = (heightmap.width.saturating_sub(1) + chunk_size - 1) / chunk_size;
    let chunks_z = (heightmap.height.saturating_sub(1) + chunk_size - 1) / chunk_size;

    let mut chunks = Vec::with_capacity((chunks_x * chunks_z) as usize);

    for cz in 0..chunks_z {
        for cx in 0..chunks_x {
            let col_start = cx * chunk_size;
            let row_start = cz * chunk_size;
            // Inclusive end vertex = col_start + chunk_size, clamped.
            let col_end = (col_start + chunk_size + 1).min(heightmap.width);
            let row_end = (row_start + chunk_size + 1).min(heightmap.height);

            let bounds = compute_chunk_bounds(heightmap, col_start, row_start, col_end, row_end);

            chunks.push(TerrainChunk {
                chunk_col: cx,
                chunk_row: cz,
                col_start,
                row_start,
                col_end,
                row_end,
                bounds,
            });
        }
    }

    chunks
}

/// Compute the AABB for a sub-region of the heightmap.
fn compute_chunk_bounds(
    heightmap: &Heightmap,
    col_start: u32,
    row_start: u32,
    col_end: u32,
    row_end: u32,
) -> Aabb {
    let x_min = col_start as f32 * heightmap.cell_size;
    let x_max = (col_end.saturating_sub(1)) as f32 * heightmap.cell_size;
    let z_min = row_start as f32 * heightmap.cell_size;
    let z_max = (row_end.saturating_sub(1)) as f32 * heightmap.cell_size;

    // Scan for min/max height in this region using raw grid values
    // (avoids the cost of bilinear interpolation at exact grid points).
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;
    for row in row_start..row_end {
        for col in col_start..col_end {
            let h = heightmap.raw_at(col, row) * heightmap.max_height;
            y_min = y_min.min(h);
            y_max = y_max.max(h);
        }
    }

    // Guard against degenerate case.
    if y_min > y_max {
        y_min = 0.0;
        y_max = 0.0;
    }

    Aabb::new(Vec3::new(x_min, y_min, z_min), Vec3::new(x_max, y_max, z_max))
}

/// Test whether an AABB is (at least partially) inside a frustum defined by
/// six plane normals + distances.
///
/// Each plane is `(Vec3 normal, f32 distance)` where a point is on the
/// positive side if `dot(normal, point) + distance >= 0`.
///
/// Returns `true` if the AABB is potentially visible.
pub fn aabb_in_frustum(bounds: &Aabb, planes: &[(Vec3, f32); 6]) -> bool {
    for &(normal, dist) in planes {
        // Find the corner of the AABB most in the direction of the plane normal.
        let positive_vertex = Vec3::new(
            if normal.x >= 0.0 {
                bounds.max.x
            } else {
                bounds.min.x
            },
            if normal.y >= 0.0 {
                bounds.max.y
            } else {
                bounds.min.y
            },
            if normal.z >= 0.0 {
                bounds.max.z
            } else {
                bounds.min.z
            },
        );

        if normal.dot(positive_vertex) + dist < 0.0 {
            return false; // Entirely behind this plane.
        }
    }
    true
}

/// Generate a mesh for a single chunk at the given LOD step.
pub fn generate_chunk_mesh(heightmap: &Heightmap, chunk: &TerrainChunk, lod_step: u32) -> TerrainMesh {
    generate_terrain_mesh_region(
        heightmap,
        chunk.col_start,
        chunk.row_start,
        chunk.col_end,
        chunk.row_end,
        lod_step,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_count() {
        let hm = Heightmap::flat(33, 33).with_cell_size(1.0);
        // chunk_size=16: (32 cells / 16) = 2 chunks per axis => 4 chunks.
        let chunks = build_chunks(&hm, 16);
        assert_eq!(chunks.len(), 4);
    }

    #[test]
    fn chunk_bounds_cover_terrain() {
        let hm = Heightmap::flat(5, 5).with_cell_size(2.0);
        let chunks = build_chunks(&hm, 4);
        assert_eq!(chunks.len(), 1);

        let b = &chunks[0].bounds;
        assert!((b.min.x).abs() < 1e-4);
        assert!((b.max.x - 8.0).abs() < 1e-4); // (5-1)*2
    }

    #[test]
    fn frustum_culling_accepts_visible() {
        let bounds = Aabb::new(Vec3::ZERO, Vec3::ONE);
        // All planes far away — everything visible.
        let planes = [
            (Vec3::X, 10.0),
            (-Vec3::X, 10.0),
            (Vec3::Y, 10.0),
            (-Vec3::Y, 10.0),
            (Vec3::Z, 10.0),
            (-Vec3::Z, 10.0),
        ];
        assert!(aabb_in_frustum(&bounds, &planes));
    }

    #[test]
    fn frustum_culling_rejects_behind() {
        let bounds = Aabb::new(Vec3::new(5.0, 0.0, 0.0), Vec3::new(6.0, 1.0, 1.0));
        // Left plane at x=0 pointing +X with distance 0 => requires x >= 0. OK.
        // Right plane at x=3 pointing -X => normal=(-1,0,0), dist=3 => -x + 3 >= 0 => x <= 3.
        // AABB at x=[5,6] is entirely to the right of x=3.
        let planes = [
            (Vec3::X, 0.0),
            (Vec3::new(-1.0, 0.0, 0.0), 3.0), // right clip at x=3
            (Vec3::Y, 10.0),
            (-Vec3::Y, 10.0),
            (Vec3::Z, 10.0),
            (-Vec3::Z, 10.0),
        ];
        assert!(!aabb_in_frustum(&bounds, &planes));
    }
}
