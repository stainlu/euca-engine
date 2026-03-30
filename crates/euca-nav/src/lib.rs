//! Navigation and pathfinding for EucaEngine.
//!
//! Grid-based navmesh with A* pathfinding and steering behaviors.
//!
//! # Usage
//! ```ignore
//! // 1. Build navmesh from scene colliders
//! let navmesh = NavMesh::from_grid(GridConfig::default());
//! world.insert_resource(navmesh);
//!
//! // 2. Give an entity a pathfinding goal
//! world.insert(entity, NavAgent::new(5.0));
//! world.insert(entity, PathGoal::new(Vec3::new(10.0, 0.0, 5.0)));
//!
//! // 3. Run systems each tick
//! pathfinding_system(&mut world);
//! steering_system(&mut world, dt);
//! ```

pub mod level_nav;
pub mod navmesh;
mod pathfinding;
pub mod rvo;
mod steering;

pub use level_nav::{navmesh_from_level_data, navmesh_with_obstacles};
pub use navmesh::{
    GridConfig, NavMesh, build_navmesh_from_world, build_navmesh_from_world_with_radius,
};
pub use pathfinding::{find_path, smooth_path};
pub use steering::{NavAgent, PathGoal, pathfinding_system, steering_system};
