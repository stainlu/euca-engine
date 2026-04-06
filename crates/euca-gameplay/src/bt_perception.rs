//! Perception system for behavior tree AI entities.
//!
//! Runs before [`euca_ai::behavior_tree_system`] to populate each entity's
//! blackboard with sensory data: own position, nearest enemy, distance, etc.

use euca_ai::{BehaviorTreeExecutor, BlackboardValue};
use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_scene::GlobalTransform;

use crate::health::{Dead, Health};
use crate::teams::Team;

/// Populate behavior tree blackboards with perception data.
///
/// For each entity with a [`BehaviorTreeExecutor`], writes:
/// - `"self_position"` — own world position (Vec3)
/// - `"alive"` — whether the entity is alive (Bool)
/// - `"nearest_enemy"` — entity ID of closest living enemy (Entity)
/// - `"enemy_position"` — world position of nearest enemy (Vec3)
/// - `"enemy_distance"` — distance to nearest enemy (Float)
pub fn bt_perception_system(world: &mut World) {
    // Collect all living combatants for enemy scanning.
    let combatants: Vec<(Entity, Vec3, u8)> = {
        let query = Query::<(Entity, &GlobalTransform, &Team)>::new(world);
        query
            .iter()
            .filter(|(e, _, _)| world.get::<Dead>(*e).is_none())
            .filter(|(e, _, _)| world.get::<Health>(*e).is_some())
            .map(|(e, gt, t)| (e, gt.0.translation, t.0))
            .collect()
    };

    // Collect BT entities to update.
    let bt_entities: Vec<Entity> = {
        let query = Query::<(Entity, &BehaviorTreeExecutor)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    for entity in bt_entities {
        let (self_pos, self_team, is_alive) = {
            let pos = world
                .get::<GlobalTransform>(entity)
                .map(|gt| gt.0.translation)
                .unwrap_or(Vec3::ZERO);
            let team = world.get::<Team>(entity).map(|t| t.0).unwrap_or(0);
            let alive = world.get::<Dead>(entity).is_none();
            (pos, team, alive)
        };

        let executor = match world.get_mut::<BehaviorTreeExecutor>(entity) {
            Some(exec) => exec,
            None => continue,
        };

        let bb = &mut executor.blackboard;
        bb.set("self_position", BlackboardValue::Vec3(self_pos));
        bb.set("alive", BlackboardValue::Bool(is_alive));

        // Find nearest living enemy.
        let mut nearest_dist = f32::MAX;
        let mut nearest_entity = None;
        let mut nearest_pos = Vec3::ZERO;

        for &(other, other_pos, other_team) in &combatants {
            if other == entity || other_team == self_team {
                continue;
            }
            let dist = (other_pos - self_pos).length();
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest_entity = Some(other);
                nearest_pos = other_pos;
            }
        }

        if let Some(enemy) = nearest_entity {
            bb.set("nearest_enemy", BlackboardValue::Entity(enemy));
            bb.set("enemy_position", BlackboardValue::Vec3(nearest_pos));
            bb.set("enemy_distance", BlackboardValue::Float(nearest_dist));
        } else {
            bb.remove("nearest_enemy");
            bb.remove("enemy_position");
            bb.remove("enemy_distance");
        }
    }
}
