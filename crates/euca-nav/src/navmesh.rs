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

    /// Get walkable neighbors of a cell (8-connected: cardinal + diagonal).
    /// Returns `(gx, gz, cost)` where cost is 1.0 for cardinal, √2 for diagonal.
    pub fn neighbors(&self, gx: usize, gz: usize) -> Vec<(usize, usize, f32)> {
        const SQRT2: f32 = std::f32::consts::SQRT_2;
        let mut result = Vec::with_capacity(8);

        // Cardinal neighbors (cost 1.0)
        if gx > 0 && self.is_walkable(gx - 1, gz) {
            result.push((gx - 1, gz, 1.0));
        }
        if gx + 1 < self.width && self.is_walkable(gx + 1, gz) {
            result.push((gx + 1, gz, 1.0));
        }
        if gz > 0 && self.is_walkable(gx, gz - 1) {
            result.push((gx, gz - 1, 1.0));
        }
        if gz + 1 < self.height && self.is_walkable(gx, gz + 1) {
            result.push((gx, gz + 1, 1.0));
        }

        // Diagonal neighbors (cost √2) — only if both adjacent cardinal cells are walkable
        // (prevents corner cutting through obstacles)
        if gx > 0
            && gz > 0
            && self.is_walkable(gx - 1, gz - 1)
            && self.is_walkable(gx - 1, gz)
            && self.is_walkable(gx, gz - 1)
        {
            result.push((gx - 1, gz - 1, SQRT2));
        }
        if gx + 1 < self.width
            && gz > 0
            && self.is_walkable(gx + 1, gz - 1)
            && self.is_walkable(gx + 1, gz)
            && self.is_walkable(gx, gz - 1)
        {
            result.push((gx + 1, gz - 1, SQRT2));
        }
        if gx > 0
            && gz + 1 < self.height
            && self.is_walkable(gx - 1, gz + 1)
            && self.is_walkable(gx - 1, gz)
            && self.is_walkable(gx, gz + 1)
        {
            result.push((gx - 1, gz + 1, SQRT2));
        }
        if gx + 1 < self.width
            && gz + 1 < self.height
            && self.is_walkable(gx + 1, gz + 1)
            && self.is_walkable(gx + 1, gz)
            && self.is_walkable(gx, gz + 1)
        {
            result.push((gx + 1, gz + 1, SQRT2));
        }

        result
    }

    /// Check if all cells along a grid line are walkable (Bresenham).
    /// Used for path smoothing line-of-sight checks.
    pub fn line_of_sight(&self, x0: usize, z0: usize, x1: usize, z1: usize) -> bool {
        let mut x = x0 as i32;
        let mut z = z0 as i32;
        let dx = (x1 as i32 - x0 as i32).abs();
        let dz = (z1 as i32 - z0 as i32).abs();
        let sx = if (x1 as i32) > x { 1 } else { -1 };
        let sz = if (z1 as i32) > z { 1 } else { -1 };
        let mut err = dx - dz;

        loop {
            if !self.is_walkable(x as usize, z as usize) {
                return false;
            }
            if x == x1 as i32 && z == z1 as i32 {
                break;
            }
            let e2 = 2 * err;
            if e2 > -dz {
                err -= dz;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                z += sz;
            }
        }
        true
    }
}

/// Build a navmesh from physics colliders in the world.
///
/// `agent_radius` inflates each obstacle by the agent's collision radius
/// so pathfinding keeps agents away from obstacle edges.
pub fn build_navmesh_from_world(world: &euca_ecs::World, config: GridConfig) -> NavMesh {
    build_navmesh_from_world_with_radius(world, config, 0.0)
}

/// Build a navmesh with agent radius inflation.
pub fn build_navmesh_from_world_with_radius(
    world: &euca_ecs::World,
    config: GridConfig,
    agent_radius: f32,
) -> NavMesh {
    use euca_physics::{Collider, ColliderShape};
    use euca_scene::GlobalTransform;

    let mut mesh = NavMesh::from_grid(config);

    // Block cells that overlap with static colliders (inflated by agent_radius)
    let query = euca_ecs::Query::<(&GlobalTransform, &Collider)>::new(world);
    for (gt, collider) in query.iter() {
        let pos = gt.0.translation;
        let pad = agent_radius;
        match &collider.shape {
            ColliderShape::Aabb { hx, hy: _, hz } => {
                mesh.block_aabb(pos, Vec3::new(*hx + pad, 0.0, *hz + pad));
            }
            ColliderShape::Sphere { radius } => {
                mesh.block_aabb(pos, Vec3::new(*radius + pad, 0.0, *radius + pad));
            }
            ColliderShape::Capsule { radius, .. } => {
                mesh.block_aabb(pos, Vec3::new(*radius + pad, 0.0, *radius + pad));
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
        // 8-connected: at corner (0,0) → right, down, and diagonal (1,1) = 3
        assert_eq!(n.len(), 3);
    }

    #[test]
    fn line_of_sight_clear() {
        let mesh = NavMesh::from_grid(GridConfig {
            min: [0.0, 0.0],
            max: [10.0, 10.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        assert!(mesh.line_of_sight(0, 0, 9, 9));
    }

    #[test]
    fn line_of_sight_blocked() {
        let mut mesh = NavMesh::from_grid(GridConfig {
            min: [0.0, 0.0],
            max: [10.0, 10.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        mesh.block(5, 5);
        assert!(!mesh.line_of_sight(0, 0, 9, 9));
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
