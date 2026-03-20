//! Quad-tree LOD selection for terrain chunks.
//!
//! Given a camera position and a set of terrain chunks, determine the LOD step
//! (vertex skip count) for each chunk.  Closer chunks use full resolution;
//! distant chunks skip vertices to reduce triangle count.

use euca_math::Vec3;

use crate::chunk::TerrainChunk;

/// LOD configuration controlling distance thresholds.
#[derive(Clone, Debug)]
pub struct LodConfig {
    /// Distance ranges for each LOD level.  `ranges[i]` is the maximum
    /// distance (from camera to chunk centre) at which LOD level `i` is used.
    /// LOD 0 = full resolution (step 1), LOD 1 = step 2, etc.
    ///
    /// Distances should be in ascending order.  Chunks beyond the last
    /// range use the coarsest LOD.
    pub ranges: Vec<f32>,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            ranges: vec![50.0, 100.0, 200.0, 400.0],
        }
    }
}

/// The result of LOD selection for a single chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkLod {
    /// The LOD level (0 = highest detail).
    pub level: u32,
    /// Vertex step: 1 = every vertex, 2 = every other, 4 = every fourth, etc.
    pub step: u32,
}

/// Select LOD level for a single chunk based on distance to the camera.
pub fn select_chunk_lod(
    chunk: &TerrainChunk,
    camera_position: Vec3,
    config: &LodConfig,
) -> ChunkLod {
    let center = chunk.bounds.center();
    let distance = center.distance(camera_position);

    for (i, &max_dist) in config.ranges.iter().enumerate() {
        if distance <= max_dist {
            let step = 1u32 << i; // 1, 2, 4, 8, ...
            return ChunkLod {
                level: i as u32,
                step,
            };
        }
    }

    // Beyond all configured ranges — use the coarsest level.
    let level = config.ranges.len() as u32;
    let step = 1u32 << level;
    ChunkLod { level, step }
}

/// Select LOD levels for all chunks at once. Returns a vec parallel to `chunks`.
pub fn select_all_lods(
    chunks: &[TerrainChunk],
    camera_position: Vec3,
    config: &LodConfig,
) -> Vec<ChunkLod> {
    chunks
        .iter()
        .map(|c| select_chunk_lod(c, camera_position, config))
        .collect()
}

#[cfg(test)]
mod tests {
    use euca_math::Aabb;

    use super::*;

    fn make_chunk(center_x: f32, center_z: f32) -> TerrainChunk {
        TerrainChunk {
            chunk_col: 0,
            chunk_row: 0,
            col_start: 0,
            row_start: 0,
            col_end: 10,
            row_end: 10,
            bounds: Aabb::new(
                Vec3::new(center_x - 5.0, 0.0, center_z - 5.0),
                Vec3::new(center_x + 5.0, 10.0, center_z + 5.0),
            ),
        }
    }

    #[test]
    fn close_chunk_gets_lod0() {
        let config = LodConfig::default(); // [50, 100, 200, 400]
        let chunk = make_chunk(0.0, 0.0);
        let camera = Vec3::new(0.0, 5.0, 0.0); // at chunk centre
        let lod = select_chunk_lod(&chunk, camera, &config);
        assert_eq!(lod.level, 0);
        assert_eq!(lod.step, 1);
    }

    #[test]
    fn distant_chunk_gets_coarser_lod() {
        let config = LodConfig::default();
        // Place chunk ~150 units away (between range[1]=100 and range[2]=200).
        let chunk = make_chunk(150.0, 0.0);
        let camera = Vec3::ZERO;
        let lod = select_chunk_lod(&chunk, camera, &config);
        assert_eq!(lod.level, 2);
        assert_eq!(lod.step, 4);
    }

    #[test]
    fn very_far_chunk_gets_coarsest() {
        let config = LodConfig {
            ranges: vec![10.0, 20.0],
        };
        let chunk = make_chunk(500.0, 0.0);
        let camera = Vec3::ZERO;
        let lod = select_chunk_lod(&chunk, camera, &config);
        assert_eq!(lod.level, 2);
        assert_eq!(lod.step, 4);
    }

    #[test]
    fn select_all_returns_parallel_vec() {
        let config = LodConfig::default();
        let chunks = vec![make_chunk(0.0, 0.0), make_chunk(300.0, 0.0)];
        let camera = Vec3::ZERO;
        let lods = select_all_lods(&chunks, camera, &config);
        assert_eq!(lods.len(), 2);
        // First should be closer (lower LOD level).
        assert!(lods[0].level < lods[1].level);
    }
}
