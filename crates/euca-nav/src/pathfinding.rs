//! A* pathfinding on grid navmesh.

use euca_math::Vec3;
use std::collections::{BinaryHeap, HashMap};

use crate::navmesh::NavMesh;

/// A node in the A* open set.
#[derive(Clone, PartialEq)]
struct AStarNode {
    gx: usize,
    gz: usize,
    f_cost: f32,
}

impl Eq for AStarNode {}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: reverse comparison so lowest f_cost comes first
        other
            .f_cost
            .partial_cmp(&self.f_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Heuristic: Euclidean distance (admissible for 8-connected).
fn heuristic(ax: usize, az: usize, bx: usize, bz: usize) -> f32 {
    let dx = ax.abs_diff(bx) as f32;
    let dz = az.abs_diff(bz) as f32;
    (dx * dx + dz * dz).sqrt()
}

/// Find a path from `start` to `goal` on the navmesh using A*.
///
/// Returns a list of world-space waypoints, or None if no path exists.
pub fn find_path(mesh: &NavMesh, start: Vec3, goal: Vec3) -> Option<Vec<Vec3>> {
    let sx = mesh.world_to_grid_x(start.x);
    let sz = mesh.world_to_grid_z(start.z);
    let gx = mesh.world_to_grid_x(goal.x);
    let gz = mesh.world_to_grid_z(goal.z);

    // Quick checks
    if !mesh.is_walkable(sx, sz) || !mesh.is_walkable(gx, gz) {
        return None;
    }
    if sx == gx && sz == gz {
        return Some(vec![goal]);
    }

    let mut open = BinaryHeap::new();
    let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut g_score: HashMap<(usize, usize), f32> = HashMap::new();

    g_score.insert((sx, sz), 0.0);
    open.push(AStarNode {
        gx: sx,
        gz: sz,
        f_cost: heuristic(sx, sz, gx, gz),
    });

    while let Some(current) = open.pop() {
        if current.gx == gx && current.gz == gz {
            // Reconstruct path
            let mut path = Vec::new();
            let mut node = (gx, gz);
            path.push(goal); // Use exact goal position for last waypoint
            while let Some(&prev) = came_from.get(&node) {
                if prev == (sx, sz) {
                    break;
                }
                path.push(mesh.grid_to_world(prev.0, prev.1));
                node = prev;
            }
            path.reverse();
            return Some(path);
        }

        let current_g = g_score
            .get(&(current.gx, current.gz))
            .copied()
            .unwrap_or(f32::MAX);

        for (nx, nz, move_cost) in mesh.neighbors(current.gx, current.gz) {
            let tentative_g = current_g + move_cost;
            let prev_g = g_score.get(&(nx, nz)).copied().unwrap_or(f32::MAX);

            if tentative_g < prev_g {
                came_from.insert((nx, nz), (current.gx, current.gz));
                g_score.insert((nx, nz), tentative_g);
                let f = tentative_g + heuristic(nx, nz, gx, gz);
                open.push(AStarNode {
                    gx: nx,
                    gz: nz,
                    f_cost: f,
                });
            }
        }
    }

    None // No path found
}

/// Smooth a path by removing unnecessary intermediate waypoints.
///
/// Uses line-of-sight checks on the grid: if we can "see" a later waypoint
/// directly (all cells along the line are walkable), skip the intermediates.
/// This turns zigzag grid paths into smooth direct-line paths.
pub fn smooth_path(mesh: &NavMesh, path: &[Vec3]) -> Vec<Vec3> {
    if path.len() <= 2 {
        return path.to_vec();
    }

    let mut smoothed = vec![path[0]];
    let mut current = 0;

    while current < path.len() - 1 {
        // Try to skip as far ahead as possible
        let mut farthest = current + 1;
        for candidate in (current + 2)..path.len() {
            let from_gx = mesh.world_to_grid_x(path[current].x);
            let from_gz = mesh.world_to_grid_z(path[current].z);
            let to_gx = mesh.world_to_grid_x(path[candidate].x);
            let to_gz = mesh.world_to_grid_z(path[candidate].z);
            if mesh.line_of_sight(from_gx, from_gz, to_gx, to_gz) {
                farthest = candidate;
            } else {
                break;
            }
        }
        smoothed.push(path[farthest]);
        current = farthest;
    }

    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::navmesh::{GridConfig, NavMesh};

    #[test]
    fn path_straight_line() {
        let mesh = NavMesh::from_grid(GridConfig {
            min: [0.0, 0.0],
            max: [10.0, 10.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        let path = find_path(&mesh, Vec3::new(0.5, 0.0, 0.5), Vec3::new(5.5, 0.0, 0.5));
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn path_around_obstacle() {
        let mut mesh = NavMesh::from_grid(GridConfig {
            min: [0.0, 0.0],
            max: [10.0, 10.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        // Block a wall from z=0 to z=8 at x=5
        for z in 0..8 {
            mesh.block(5, z);
        }
        let path = find_path(&mesh, Vec3::new(2.5, 0.0, 2.5), Vec3::new(7.5, 0.0, 2.5));
        assert!(path.is_some());
        let path = path.unwrap();
        // Path must go around the wall — should have waypoints at z > 8
        assert!(path.len() > 5); // longer than straight line
    }

    #[test]
    fn no_path_blocked() {
        let mut mesh = NavMesh::from_grid(GridConfig {
            min: [0.0, 0.0],
            max: [10.0, 10.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        // Block complete wall
        for z in 0..10 {
            mesh.block(5, z);
        }
        let path = find_path(&mesh, Vec3::new(2.5, 0.0, 2.5), Vec3::new(7.5, 0.0, 2.5));
        assert!(path.is_none());
    }

    #[test]
    fn same_cell_path() {
        let mesh = NavMesh::from_grid(GridConfig {
            min: [0.0, 0.0],
            max: [10.0, 10.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        let path = find_path(&mesh, Vec3::new(5.5, 0.0, 5.5), Vec3::new(5.8, 0.0, 5.2));
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 1); // already there
    }
}
