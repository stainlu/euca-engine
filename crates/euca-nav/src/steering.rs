//! Steering behaviors and navigation systems.

use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_physics::Velocity;
use euca_scene::GlobalTransform;

use crate::navmesh::NavMesh;
use crate::pathfinding::find_path;

/// ECS component: navigation agent properties.
#[derive(Clone, Debug)]
pub struct NavAgent {
    /// Movement speed.
    pub speed: f32,
    /// Agent radius (for obstacle avoidance).
    pub radius: f32,
}

impl NavAgent {
    pub fn new(speed: f32) -> Self {
        Self { speed, radius: 0.5 }
    }
}

/// ECS component: pathfinding target and cached path.
#[derive(Clone, Debug)]
pub struct PathGoal {
    /// Target world position.
    pub target: Vec3,
    /// Computed path (list of waypoints).
    pub path: Option<Vec<Vec3>>,
    /// Index of next waypoint to reach.
    pub current_waypoint: usize,
    /// Whether path needs recomputation.
    pub dirty: bool,
}

impl PathGoal {
    pub fn new(target: Vec3) -> Self {
        Self {
            target,
            path: None,
            current_waypoint: 0,
            dirty: true,
        }
    }
}

/// System: compute A* paths for entities with dirty PathGoals.
pub fn pathfinding_system(world: &mut World) {
    // Get navmesh
    let navmesh = match world.resource::<NavMesh>() {
        Some(m) => m.clone(),
        None => return,
    };

    // Collect entities needing pathfinding
    let needs_path: Vec<(Entity, Vec3, Vec3)> = {
        let query = Query::<(Entity, &GlobalTransform, &PathGoal)>::new(world);
        query
            .iter()
            .filter(|(_, _, pg)| pg.dirty)
            .map(|(e, gt, pg)| (e, gt.0.translation, pg.target))
            .collect()
    };

    for (entity, pos, target) in needs_path {
        let path = find_path(&navmesh, pos, target);

        if let Some(goal) = world.get_mut::<PathGoal>(entity) {
            goal.path = path;
            goal.current_waypoint = 0;
            goal.dirty = false;
        }
    }
}

/// System: steer entities along their computed paths.
pub fn steering_system(world: &mut World, dt: f32) {
    let _ = dt; // dt not needed for velocity-based steering

    struct SteerData {
        entity: Entity,
        pos: Vec3,
        speed: f32,
        path: Option<Vec<Vec3>>,
        waypoint_idx: usize,
    }

    let nav_data: Vec<SteerData> = {
        let query = Query::<(Entity, &GlobalTransform, &NavAgent, &PathGoal)>::new(world);
        query
            .iter()
            .map(|(e, gt, agent, goal)| SteerData {
                entity: e,
                pos: gt.0.translation,
                speed: agent.speed,
                path: goal.path.clone(),
                waypoint_idx: goal.current_waypoint,
            })
            .collect()
    };

    for s in nav_data {
        let SteerData {
            entity,
            pos,
            speed,
            path,
            waypoint_idx,
        } = s;
        let path = match path {
            Some(p) => p,
            None => {
                // No path — stop moving
                if let Some(vel) = world.get_mut::<Velocity>(entity) {
                    vel.linear = Vec3::ZERO;
                }
                continue;
            }
        };

        if waypoint_idx >= path.len() {
            // Reached end of path — stop
            if let Some(vel) = world.get_mut::<Velocity>(entity) {
                vel.linear = Vec3::ZERO;
            }
            continue;
        }

        let target_wp = path[waypoint_idx];
        let to_target = target_wp - pos;
        let dist = Vec3::new(to_target.x, 0.0, to_target.z).length(); // XZ distance only

        let arrival_threshold = 0.5;

        if dist < arrival_threshold {
            // Advance to next waypoint
            if let Some(goal) = world.get_mut::<PathGoal>(entity) {
                goal.current_waypoint += 1;
            }
        } else {
            // Move toward waypoint
            let dir = Vec3::new(to_target.x, 0.0, to_target.z).normalize();
            if let Some(vel) = world.get_mut::<Velocity>(entity) {
                vel.linear.x = dir.x * speed;
                vel.linear.z = dir.z * speed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::navmesh::{GridConfig, NavMesh};
    use euca_math::Transform;
    use euca_scene::LocalTransform;

    #[test]
    fn pathfinding_system_computes_path() {
        let mut world = World::new();

        let mesh = NavMesh::from_grid(GridConfig {
            min: [0.0, 0.0],
            max: [20.0, 20.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        world.insert_resource(mesh);

        let e = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            2.0, 0.0, 2.0,
        ))));
        world.insert(
            e,
            GlobalTransform(Transform::from_translation(Vec3::new(2.0, 0.0, 2.0))),
        );
        world.insert(e, NavAgent::new(5.0));
        world.insert(e, PathGoal::new(Vec3::new(15.0, 0.0, 15.0)));
        world.insert(e, Velocity::default());

        pathfinding_system(&mut world);

        let goal = world.get::<PathGoal>(e).unwrap();
        assert!(!goal.dirty);
        assert!(goal.path.is_some());
        assert!(!goal.path.as_ref().unwrap().is_empty());
    }

    #[test]
    fn steering_moves_toward_waypoint() {
        let mut world = World::new();

        let mesh = NavMesh::from_grid(GridConfig {
            min: [0.0, 0.0],
            max: [20.0, 20.0],
            cell_size: 1.0,
            ground_y: 0.0,
        });
        world.insert_resource(mesh);

        let e = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            2.0, 0.0, 2.0,
        ))));
        world.insert(
            e,
            GlobalTransform(Transform::from_translation(Vec3::new(2.0, 0.0, 2.0))),
        );
        world.insert(e, NavAgent::new(5.0));
        world.insert(e, PathGoal::new(Vec3::new(15.0, 0.0, 2.0)));
        world.insert(e, Velocity::default());

        pathfinding_system(&mut world);
        steering_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(e).unwrap();
        // Should be moving in +X direction
        assert!(vel.linear.x > 0.0);
    }
}
