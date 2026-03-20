//! Physics heightfield collider generation.
//!
//! Produces AABB colliders from the heightmap so the engine's existing physics
//! pipeline can resolve terrain collisions without a dedicated heightfield
//! narrow-phase.

use euca_math::Vec3;
use euca_physics::Collider;
#[cfg(test)]
use euca_physics::ColliderShape;

use crate::heightmap::Heightmap;

/// A positioned collider tile for one heightmap cell.
#[derive(Clone, Debug)]
pub struct HeightfieldTile {
    /// World-space centre of the tile's collider.
    pub position: Vec3,
    /// The AABB collider for this tile.
    pub collider: Collider,
}

/// Generate AABB colliders for every cell in the heightmap.
///
/// Each cell becomes an axis-aligned box whose top face matches the minimum
/// and maximum heights of that cell's four corner vertices.  This is a
/// conservative approximation that works well with the existing AABB collision
/// pipeline.
///
/// `step` controls the granularity: 1 = one collider per cell,
/// 2 = one per 2x2 block, etc.  Higher values reduce physics cost at the
/// expense of accuracy.
pub fn generate_heightfield_colliders(heightmap: &Heightmap, step: u32) -> Vec<HeightfieldTile> {
    let step = step.max(1);
    let cols = heightmap.width.saturating_sub(1);
    let rows = heightmap.height.saturating_sub(1);

    let mut tiles = Vec::new();

    let mut row = 0u32;
    while row < rows {
        let mut col = 0u32;
        while col < cols {
            let col_end = (col + step).min(cols);
            let row_end = (row + step).min(rows);

            // Find min/max height across this block using raw grid values
            // (avoids bilinear interpolation at exact grid points).
            let mut y_min = f32::MAX;
            let mut y_max = f32::MIN;
            for r in row..=row_end {
                for c in col..=col_end {
                    let h = heightmap.raw_at(c, r) * heightmap.max_height;
                    y_min = y_min.min(h);
                    y_max = y_max.max(h);
                }
            }

            // Ensure a minimum thickness so flat areas still have a collider.
            if (y_max - y_min) < 0.01 {
                y_min -= 0.5;
            }

            let x_min = col as f32 * heightmap.cell_size;
            let x_max = col_end as f32 * heightmap.cell_size;
            let z_min = row as f32 * heightmap.cell_size;
            let z_max = row_end as f32 * heightmap.cell_size;

            let center = Vec3::new(
                (x_min + x_max) * 0.5,
                (y_min + y_max) * 0.5,
                (z_min + z_max) * 0.5,
            );
            let hx = (x_max - x_min) * 0.5;
            let hy = (y_max - y_min) * 0.5;
            let hz = (z_max - z_min) * 0.5;

            tiles.push(HeightfieldTile {
                position: center,
                collider: Collider::aabb(hx, hy, hz)
                    .with_friction(0.8)
                    .with_restitution(0.1),
            });

            col += step;
        }
        row += step;
    }

    tiles
}

/// Perform a vertical ray test against the heightmap.
///
/// Given an `(x, z)` world position, returns the height at that point.
/// This is a lightweight alternative to full raycasting for ground queries.
pub fn height_at(heightmap: &Heightmap, x: f32, z: f32) -> f32 {
    heightmap.sample(x, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_terrain_generates_colliders() {
        let hm = Heightmap::flat(5, 5).with_cell_size(1.0);
        let tiles = generate_heightfield_colliders(&hm, 1);
        // 4x4 cells => 16 tiles.
        assert_eq!(tiles.len(), 16);

        for tile in &tiles {
            // All colliders should have the same shape since terrain is flat.
            match &tile.collider.shape {
                ColliderShape::Aabb { hx, hy: _, hz } => {
                    assert!((*hx - 0.5).abs() < 1e-4);
                    assert!((*hz - 0.5).abs() < 1e-4);
                }
                _ => panic!("Expected AABB collider"),
            }
        }
    }

    #[test]
    fn stepped_colliders_fewer_tiles() {
        let hm = Heightmap::flat(9, 9).with_cell_size(1.0);
        let tiles_full = generate_heightfield_colliders(&hm, 1);
        let tiles_half = generate_heightfield_colliders(&hm, 2);
        assert!(tiles_half.len() < tiles_full.len());
    }

    #[test]
    fn height_at_matches_sample() {
        let hm = Heightmap::from_raw(3, 3, vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
            .with_max_height(10.0);
        let h = height_at(&hm, 1.0, 1.0);
        assert!((h - hm.sample(1.0, 1.0)).abs() < 1e-6);
    }
}
