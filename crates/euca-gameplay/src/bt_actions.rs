//! Behavior tree action handlers — bridge BT intent to ECS mutations.
//!
//! The [`euca_ai::BtAction::MoveTo`] action signals intent without moving the
//! entity. This system reads the blackboard target and applies velocity,
//! mirroring the logic in [`crate::ai::ai_system`].

use euca_ai::BehaviorTreeExecutor;
use euca_ecs::{Entity, Query, World};
use euca_math::{Quat, Vec3};
use euca_physics::Velocity;
use euca_scene::LocalTransform;

/// Default movement speed for BT-controlled entities.
const BT_MOVE_SPEED: f32 = 4.0;

/// Apply movement for entities whose behavior tree has a `MoveTo` action running.
///
/// Reads `"enemy_position"` or `"patrol_target"` from the blackboard (whichever
/// the active MoveTo references) and steers the entity toward it.
pub fn bt_moveto_system(world: &mut World) {
    let updates: Vec<(Entity, Vec3)> = {
        let query = Query::<(Entity, &BehaviorTreeExecutor, &LocalTransform)>::new(world);
        query
            .iter()
            .filter_map(|(e, exec, lt)| {
                // Check common target keys in priority order.
                let target_pos = exec
                    .blackboard
                    .get_vec3("enemy_position")
                    .or_else(|| exec.blackboard.get_vec3("patrol_target"))?;
                let self_pos = lt.0.translation;
                let delta = target_pos - self_pos;
                let horizontal = Vec3::new(delta.x, 0.0, delta.z);
                if horizontal.length() < 0.3 {
                    return None; // Close enough
                }
                Some((e, horizontal.normalize()))
            })
            .collect()
    };

    for (entity, dir) in updates {
        if let Some(vel) = world.get_mut::<Velocity>(entity) {
            vel.linear.x = dir.x * BT_MOVE_SPEED;
            vel.linear.z = dir.z * BT_MOVE_SPEED;
        }
        if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
            let target_angle = dir.x.atan2(dir.z);
            lt.0.rotation = Quat::from_axis_angle(Vec3::Y, target_angle);
        }
    }
}
