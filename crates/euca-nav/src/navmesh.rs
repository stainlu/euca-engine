//! Grid-based navigation mesh.

use euca_math::Vec3;
use serde::{Deserialize, Serialize};

/// Configuration for generating a grid navmesh.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GridConfig {
    /// World-space min corner of the grid.
    pub min: [f32; 2],
    /// World-space max corner of the grid.
    pub max: [f32; 2],
    /// Cell size (both x and z dimensions).
    pub cell_size: f32,
    /// Y level of the walkable surface.
    pub ground_y: f32,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            min: [-50.0, -50.0],
            max: [50.0, 50.0],
            cell_size: 1.0,
            ground_y: 0.0,
        }
    }
}

/// A grid-based navigation mesh. Each cell is walkable or blocked.
#[derive(Clone, Debug)]
pub struct NavMesh {
    pub config: GridConfig,
    /// Number of cells along X axis.
    pub width: usize,
    /// Number of cells along Z axis.
    pub height: usize,
    /// Flat array: `true` = walkable. Index = z * width + x.
    pub walkable: Vec<bool>,
}

impl NavMesh {
    /// Create an empty (all walkable) navmesh from grid config.
    pub fn from_grid(config: GridConfig) -> Self {
        let width = ((config.max[0] - config.min[0]) / config.cell_size).ceil() as usize;
        let height = ((config.max[1] - config.min[1]) / config.cell_size).ceil() as usize;
        let walkable = vec![true; width * height];
        log::info!(
            "NavMesh: {}x{} grid ({} cells)",
            width,
            height,
            width * height
        );
        Self {
            config,
            width,
            height,
            walkable,
        }
    }

    /// Mark a cell as blocked (not walkable).
    pub fn block(&mut self, gx: usize, gz: usize) {
        if gx < self.width && gz < self.height {
            self.walkable[gz * self.width + gx] = false;
        }
    }

    /// Mark cells that overlap with an AABB obstacle.
    pub fn block_aabb(&mut self, center: Vec3, half_extents: Vec3) {
        let min_x = center.x - half_extents.x;
        let max_x = center.x + half_extents.x;
        let min_z = center.z - half_extents.z;
        let max_z = center.z + half_extents.z;

        let gx_min = self.world_to_grid_x(min_x);
        let gx_max = self.world_to_grid_x(max_x);
        let gz_min = self.world_to_grid_z(min_z);
        let gz_max = self.world_to_grid_z(max_z);

        for gz in gz_min..=gz_max {
            for gx in gx_min..=gx_max {
                self.block(gx, gz);
            }
        }
    }

    /// Check if a cell is walkable.
    pub fn is_walkable(&self, gx: usize, gz: usize) -> bool {
        if gx < self.width && gz < self.height {
            self.walkable[gz * self.width + gx]
        } else {
            false
        }
    }

    /// Convert world X position to grid X index.
    pub fn world_to_grid_x(&self, wx: f32) -> usize {
        ((wx - self.config.min[0]) / self.config.cell_size)
            .floor()
            .max(0.0) as usize
    }

    /// Convert world Z position to grid Z index.
    pub fn world_to_grid_z(&self, wz: f32) -> usize {
        ((wz - self.config.min[1]) / self.config.cell_size)
            .floor()
            .max(0.0) as usize
    }

    /// Convert grid coordinates to world position (cell center).
    pub fn grid_to_world(&self, gx: usize, gz: usize) -> Vec3 {
        Vec3::new(
            self.config.min[0] + (gx as f32 + 0.5) * self.config.cell_size,
            self.config.ground_y,
            self.config.min[1] + (gz as f32 + 0.5) * self.config.cell_size,
        )
    }

    /// Get walkable neighbors of a cell (4-connected: up/down/left/right).
    pub fn neighbors(&self, gx: usize, gz: usize) -> Vec<(usize, usize)> {
        let mut result = Vec::with_capacity(4);
        if gx > 0 && self.is_walkable(gx - 1, gz) {
            result.push((gx - 1, gz));
        }
        if gx + 1 < self.width && self.is_walkable(gx + 1, gz) {
            result.push((gx + 1, gz));
        }
        if gz > 0 && self.is_walkable(gx, gz - 1) {
            result.push((gx, gz - 1));
        }
        if gz + 1 < self.height && self.is_walkable(gx, gz + 1) {
            result.push((gx, gz + 1));
        }
        result
    }
}

/// Build a navmesh from physics colliders in the world.
pub fn build_navmesh_from_world(world: &euca_ecs::World, config: GridConfig) -> NavMesh {
    use euca_physics::{Collider, ColliderShape};
    use euca_scene::GlobalTransform;

    let mut mesh = NavMesh::from_grid(config);

    // Block cells that overlap with static colliders
    let query = euca_ecs::Query::<(&GlobalTransform, &Collider)>::new(world);
    for (gt, collider) in query.iter() {
        let pos = gt.0.translation;
        match &collider.shape {
            ColliderShape::Aabb { hx, hy: _, hz } => {
                mesh.block_aabb(pos, Vec3::new(*hx, 0.0, *hz));
            }
            ColliderShape::Sphere { radius } => {
                mesh.block_aabb(pos, Vec3::new(*radius, 0.0, *radius));
            }
            ColliderShape::Capsule { radius, .. } => {
                mesh.block_aabb(pos, Vec3::new(*radius, 0.0, *radius));
            }
        }
    }

    let blocked = mesh.walkable.iter().filter(|&&w| !w).count();
    log::info!("NavMesh built: {} blocked cells", blocked);
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_creation() {
        let mesh = NavMesh::from_grid(GridConfig {
            min: [0.0, 0.0],
            max: [10.0, 10.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        assert_eq!(mesh.width, 10);
        assert_eq!(mesh.height, 10);
        assert!(mesh.is_walkable(5, 5));
    }

    #[test]
    fn block_cell() {
        let mut mesh = NavMesh::from_grid(GridConfig {
            min: [0.0, 0.0],
            max: [10.0, 10.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        mesh.block(3, 3);
        assert!(!mesh.is_walkable(3, 3));
        assert!(mesh.is_walkable(4, 3));
    }

    #[test]
    fn neighbors_at_corner() {
        let mesh = NavMesh::from_grid(GridConfig {
            min: [0.0, 0.0],
            max: [5.0, 5.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        let n = mesh.neighbors(0, 0);
        assert_eq!(n.len(), 2); // right + down
    }

    #[test]
    fn world_grid_conversion() {
        let mesh = NavMesh::from_grid(GridConfig {
            min: [-5.0, -5.0],
            max: [5.0, 5.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        let gx = mesh.world_to_grid_x(0.0);
        let gz = mesh.world_to_grid_z(0.0);
        assert_eq!(gx, 5);
        assert_eq!(gz, 5);
        let wp = mesh.grid_to_world(5, 5);
        assert!((wp.x - 0.5).abs() < 0.01);
    }
}
