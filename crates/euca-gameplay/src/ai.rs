//! AI goal-driven behavior.
//!
//! Components: `AiGoal`.
//! Systems: `ai_system`.

use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_physics::Velocity;
use euca_scene::LocalTransform;

/// What this AI entity wants to do.
#[derive(Clone, Debug)]
pub enum AiBehavior {
    /// Stand still.
    Idle,
    /// Move between waypoints in order.
    Patrol,
    /// Move toward a target entity.
    Chase,
    /// Move away from a target entity.
    Flee,
}

/// AI decision state. Attached to entities controlled by AI.
#[derive(Clone, Debug)]
pub struct AiGoal {
    pub behavior: AiBehavior,
    pub target: Option<Entity>,
    pub home: Vec3,
    pub waypoints: Vec<Vec3>,
    pub waypoint_index: usize,
    pub speed: f32,
}

impl AiGoal {
    pub fn idle(home: Vec3) -> Self {
        Self {
            behavior: AiBehavior::Idle,
            target: None,
            home,
            waypoints: Vec::new(),
            waypoint_index: 0,
            speed: 3.0,
        }
    }

    pub fn patrol(waypoints: Vec<Vec3>, speed: f32) -> Self {
        let home = waypoints.first().copied().unwrap_or(Vec3::ZERO);
        Self {
            behavior: AiBehavior::Patrol,
            target: None,
            home,
            waypoints,
            waypoint_index: 0,
            speed,
        }
    }

    pub fn chase(target: Entity, speed: f32) -> Self {
        Self {
            behavior: AiBehavior::Chase,
            target: Some(target),
            home: Vec3::ZERO,
            waypoints: Vec::new(),
            waypoint_index: 0,
            speed,
        }
    }
}

/// Evaluate AI goals and set velocity accordingly.
pub fn ai_system(world: &mut World, _dt: f32) {
    // Collect AI entities
    #[allow(clippy::type_complexity)]
    let ai_entities: Vec<(Entity, AiBehavior, Option<Entity>, Vec<Vec3>, usize, f32)> = {
        let query = Query::<(Entity, &AiGoal)>::new(world);
        query
            .iter()
            .map(|(e, g)| {
                (
                    e,
                    g.behavior.clone(),
                    g.target,
                    g.waypoints.clone(),
                    g.waypoint_index,
                    g.speed,
                )
            })
            .collect()
    };

    for (entity, behavior, target, waypoints, wp_idx, speed) in ai_entities {
        let my_pos = match world.get::<LocalTransform>(entity) {
            Some(lt) => lt.0.translation,
            None => continue,
        };

        let desired_velocity = match behavior {
            AiBehavior::Idle => Vec3::ZERO,
            AiBehavior::Patrol => {
                if waypoints.is_empty() {
                    Vec3::ZERO
                } else {
                    let target_pos = waypoints[wp_idx % waypoints.len()];
                    let to_target = target_pos - my_pos;
                    let dist = to_target.length();
                    if dist < 0.5 {
                        // Advance to next waypoint
                        if let Some(goal) = world.get_mut::<AiGoal>(entity) {
                            goal.waypoint_index = (wp_idx + 1) % waypoints.len();
                        }
                        Vec3::ZERO
                    } else {
                        to_target.normalize() * speed
                    }
                }
            }
            AiBehavior::Chase => {
                if let Some(target_entity) = target
                    && let Some(target_lt) = world.get::<LocalTransform>(target_entity)
                {
                    let to_target = target_lt.0.translation - my_pos;
                    if to_target.length() > 1.0 {
                        to_target.normalize() * speed
                    } else {
                        Vec3::ZERO
                    }
                } else {
                    Vec3::ZERO
                }
            }
            AiBehavior::Flee => {
                if let Some(target_entity) = target
                    && let Some(target_lt) = world.get::<LocalTransform>(target_entity)
                {
                    let away = my_pos - target_lt.0.translation;
                    if away.length() < 10.0 {
                        away.normalize() * speed
                    } else {
                        Vec3::ZERO
                    }
                } else {
                    Vec3::ZERO
                }
            }
        };

        // Set velocity (horizontal only, preserve Y for gravity)
        if let Some(vel) = world.get_mut::<Velocity>(entity) {
            vel.linear.x = desired_velocity.x;
            vel.linear.z = desired_velocity.z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    #[test]
    fn idle_produces_zero_velocity() {
        let mut world = World::new();
        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(entity, AiGoal::idle(Vec3::ZERO));
        world.insert(entity, Velocity::default());

        ai_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(entity).unwrap();
        assert_eq!(vel.linear.x, 0.0);
        assert_eq!(vel.linear.z, 0.0);
    }

    #[test]
    fn chase_moves_toward_target() {
        let mut world = World::new();

        let target = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            10.0, 0.0, 0.0,
        ))));

        let chaser = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(chaser, AiGoal::chase(target, 5.0));
        world.insert(chaser, Velocity::default());

        ai_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(chaser).unwrap();
        assert!(vel.linear.x > 0.0); // moving toward target (positive X)
    }

    #[test]
    fn patrol_advances_waypoints() {
        let mut world = World::new();

        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 0.0, 0.0,
        ))));
        world.insert(
            entity,
            AiGoal::patrol(
                vec![
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(5.0, 0.0, 0.0),
                    Vec3::new(5.0, 0.0, 5.0),
                ],
                3.0,
            ),
        );
        world.insert(entity, Velocity::default());

        // At waypoint 0 (position matches) — should advance to waypoint 1
        ai_system(&mut world, 0.016);

        let goal = world.get::<AiGoal>(entity).unwrap();
        assert_eq!(goal.waypoint_index, 1);
    }
}
