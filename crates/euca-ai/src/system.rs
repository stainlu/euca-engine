//! ECS system that ticks all behavior tree executors each frame.

use euca_ecs::{Entity, Query, World};

use crate::executor::BehaviorTreeExecutor;

/// Tick every entity's behavior tree once per frame.
///
/// This system iterates all entities with a [`BehaviorTreeExecutor`] component,
/// calls `tick(dt)`, and stores the result back.
///
/// To feed world data into the tree, a *perception* system should run **before**
/// this one and populate the blackboard (e.g. writing `"self_position"`,
/// `"nearest_enemy"`, etc.).
pub fn behavior_tree_system(world: &mut World, dt: f32) {
    // Collect entities first to avoid borrow conflicts.
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &BehaviorTreeExecutor)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    for entity in entities {
        if let Some(executor) = world.get_mut::<BehaviorTreeExecutor>(entity) {
            executor.tick(dt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackboard::BlackboardValue;
    use crate::node::*;

    #[test]
    fn system_ticks_all_entities() {
        let mut world = World::new();

        // Entity A: sets "done" = true
        let tree_a = BtNode::Action(BtAction::SetBlackboard {
            key: "done".into(),
            value: BlackboardValue::Bool(true),
        });
        let a = world.spawn(BehaviorTreeExecutor::new(tree_a));

        // Entity B: sets "count" = 42
        let tree_b = BtNode::Action(BtAction::SetBlackboard {
            key: "count".into(),
            value: BlackboardValue::Int(42),
        });
        let b = world.spawn(BehaviorTreeExecutor::new(tree_b));

        behavior_tree_system(&mut world, 0.016);

        let exec_a = world.get::<BehaviorTreeExecutor>(a).unwrap();
        assert_eq!(exec_a.blackboard.get_bool("done"), Some(true));

        let exec_b = world.get::<BehaviorTreeExecutor>(b).unwrap();
        assert_eq!(exec_b.blackboard.get_int("count"), Some(42));
    }

    #[test]
    fn system_handles_empty_world() {
        let mut world = World::new();
        // Should not panic.
        behavior_tree_system(&mut world, 0.016);
    }
}
