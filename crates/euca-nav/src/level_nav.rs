//! Generate [`NavMesh`] from level walkability grids.
//!
//! These helpers bridge the gap between a level editor's tile/grid
//! representation and the engine's navigation mesh.  The level grid
//! uses its own cell size and coordinate system; these functions
//! translate into the NavMesh coordinate space.

use crate::navmesh::{GridConfig, NavMesh};

/// Build a [`NavMesh`] from a flat walkability grid produced by a level
/// editor.
///
/// # Parameters
///
/// * `level_width`  / `level_height` – dimensions of the level grid (in
///   cells).
/// * `level_cell_size` – world-space size of each level cell.
/// * `walkable_grid` – row-major flat array (`z * level_width + x`).
///   `true` means the cell is walkable.
/// * `nav_cell_size` – cell size for the resulting `NavMesh`.  When
///   `None`, `level_cell_size` is used directly (1:1 mapping).
/// * `ground_y` – Y height of the walkable surface.
pub fn navmesh_from_level_data(
    level_width: usize,
    level_height: usize,
    level_cell_size: f32,
    walkable_grid: &[bool],
    nav_cell_size: Option<f32>,
    ground_y: f32,
) -> NavMesh {
    let world_w = level_width as f32 * level_cell_size;
    let world_h = level_height as f32 * level_cell_size;
    let cell = nav_cell_size.unwrap_or(level_cell_size);

    let config = GridConfig {
        min: [0.0, 0.0],
        max: [world_w, world_h],
        cell_size: cell,
        ground_y,
    };

    let mut mesh = NavMesh::from_grid(config);

    // For every nav cell, check which level cell its centre falls in.
    // If that level cell is unwalkable, block the nav cell.
    for nz in 0..mesh.height {
        for nx in 0..mesh.width {
            let world_x = (nx as f32 + 0.5) * cell;
            let world_z = (nz as f32 + 0.5) * cell;

            let lx = (world_x / level_cell_size).floor() as usize;
            let lz = (world_z / level_cell_size).floor() as usize;

            // Anything outside the level grid is unwalkable.
            if lx >= level_width || lz >= level_height {
                mesh.block(nx, nz);
                continue;
            }

            if !walkable_grid[lz * level_width + lx] {
                mesh.block(nx, nz);
            }
        }
    }

    mesh
}

/// Build a [`NavMesh`] from level data **plus** a list of AABB
/// obstacles.
///
/// Each obstacle is specified as `(center, half_extents)` in world
/// space.
pub fn navmesh_with_obstacles(
    level_width: usize,
    level_height: usize,
    level_cell_size: f32,
    walkable_grid: &[bool],
    nav_cell_size: Option<f32>,
    ground_y: f32,
    obstacles: &[([f32; 3], [f32; 3])],
) -> NavMesh {
    let mut mesh = navmesh_from_level_data(
        level_width,
        level_height,
        level_cell_size,
        walkable_grid,
        nav_cell_size,
        ground_y,
    );

    for &(center, half) in obstacles {
        mesh.block_aabb(
            euca_math::Vec3::new(center[0], center[1], center[2]),
            euca_math::Vec3::new(half[0], half[1], half[2]),
        );
    }

    mesh
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Helper – count blocked cells.
    fn blocked_count(mesh: &NavMesh) -> usize {
        mesh.walkable.iter().filter(|&&w| !w).count()
    }

    // 1. All-walkable grid produces an entirely walkable navmesh.
    #[test]
    fn all_walkable() {
        let grid = vec![true; 4 * 4];
        let mesh = navmesh_from_level_data(4, 4, 1.0, &grid, None, 0.0);
        assert_eq!(mesh.width, 4);
        assert_eq!(mesh.height, 4);
        assert_eq!(blocked_count(&mesh), 0);
    }

    // 2. Blocked cells in the source grid translate to blocked nav cells.
    #[test]
    fn blocked_cells() {
        let mut grid = vec![true; 4 * 4];
        grid[0] = false; // (0,0)
        grid[1 * 4 + 2] = false; // (2,1)
        let mesh = navmesh_from_level_data(4, 4, 1.0, &grid, None, 0.0);
        assert!(!mesh.is_walkable(0, 0));
        assert!(!mesh.is_walkable(2, 1));
        assert!(mesh.is_walkable(1, 1));
    }

    // 3. Grid mapping: nav cell centres map to the correct level cell.
    #[test]
    fn grid_mapping() {
        // 2x2 level with cell size 2.0 -> world is 4x4.
        // Nav cell size 1.0 -> 4x4 nav grid.
        // Level cell (0,0) covers world [0..2, 0..2] -> nav cells (0,0),(1,0),(0,1),(1,1).
        let mut grid = vec![true; 2 * 2];
        grid[0] = false; // level (0,0) blocked
        let mesh = navmesh_from_level_data(2, 2, 2.0, &grid, Some(1.0), 0.0);
        assert_eq!(mesh.width, 4);
        assert_eq!(mesh.height, 4);
        // All four nav cells whose centres fall inside level cell (0,0).
        assert!(!mesh.is_walkable(0, 0));
        assert!(!mesh.is_walkable(1, 0));
        assert!(!mesh.is_walkable(0, 1));
        assert!(!mesh.is_walkable(1, 1));
        // Nav cells inside level cell (1,0) should be walkable.
        assert!(mesh.is_walkable(2, 0));
        assert!(mesh.is_walkable(3, 0));
    }

    // 4. Different cell sizes: nav cells larger than level cells.
    #[test]
    fn different_cell_sizes() {
        // 8x8 level, cell_size 0.5 -> world 4x4.
        // Nav cell_size 2.0 -> 2x2 nav grid.
        let grid = vec![true; 8 * 8];
        let mesh = navmesh_from_level_data(8, 8, 0.5, &grid, Some(2.0), 5.0);
        assert_eq!(mesh.width, 2);
        assert_eq!(mesh.height, 2);
        assert_eq!(mesh.config.ground_y, 5.0);
        assert_eq!(blocked_count(&mesh), 0);
    }

    // 5. Obstacles block the expected nav cells.
    #[test]
    fn obstacles() {
        let grid = vec![true; 10 * 10];
        let obstacles = vec![
            // centre (5,0,5), half-extents (1,0,1) -> covers world [4..6, 4..6].
            ([5.0, 0.0, 5.0], [1.0, 0.0, 1.0]),
        ];
        let mesh = navmesh_with_obstacles(10, 10, 1.0, &grid, None, 0.0, &obstacles);
        // Cells (4,4), (4,5), (5,4), (5,5) should be blocked.
        assert!(!mesh.is_walkable(4, 4));
        assert!(!mesh.is_walkable(5, 5));
        // Something outside the obstacle should still be walkable.
        assert!(mesh.is_walkable(0, 0));
    }

    // 6. Empty grid (0x0) produces a degenerate but safe navmesh.
    #[test]
    fn empty_grid() {
        let grid: Vec<bool> = vec![];
        let mesh = navmesh_from_level_data(0, 0, 1.0, &grid, None, 0.0);
        assert_eq!(mesh.width, 0);
        assert_eq!(mesh.height, 0);
        assert_eq!(mesh.walkable.len(), 0);
    }

    // 7. World bounds match level dimensions * cell size.
    #[test]
    fn world_bounds() {
        let grid = vec![true; 5 * 3];
        let mesh = navmesh_from_level_data(5, 3, 2.0, &grid, None, 1.0);
        assert_eq!(mesh.config.min, [0.0, 0.0]);
        assert_eq!(mesh.config.max, [10.0, 6.0]);
        assert_eq!(mesh.config.ground_y, 1.0);
    }

    // 8. Edge cells: the very last row/column are correctly handled.
    #[test]
    fn edge_cells() {
        let mut grid = vec![true; 3 * 3];
        // Block the bottom-right corner level cell.
        grid[2 * 3 + 2] = false; // (2,2)
        let mesh = navmesh_from_level_data(3, 3, 1.0, &grid, None, 0.0);
        assert!(!mesh.is_walkable(2, 2));
        // Adjacent edge cells are walkable.
        assert!(mesh.is_walkable(1, 2));
        assert!(mesh.is_walkable(2, 1));
    }
}
