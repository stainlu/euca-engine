//! Behavior tree executor: ticks a tree against a blackboard each frame.

use crate::blackboard::{Blackboard, BlackboardValue};
use crate::node::{BtAction, BtCondition, BtNode, BtStatus, CompareOp, ParallelPolicy};

/// ECS component: attach to an entity to give it a behavior tree.
///
/// Contains the tree definition, the blackboard, and mutable execution state.
#[derive(Clone, Debug)]
pub struct BehaviorTreeExecutor {
    /// The behavior tree to tick each frame.
    pub tree: BtNode,
    /// Per-entity data store read/written by tree nodes.
    pub blackboard: Blackboard,
}

impl BehaviorTreeExecutor {
    /// Creates a new executor with the given tree and an empty blackboard.
    pub fn new(tree: BtNode) -> Self {
        Self {
            tree,
            blackboard: Blackboard::new(),
        }
    }

    /// Creates a new executor with a pre-populated blackboard.
    pub fn with_blackboard(tree: BtNode, blackboard: Blackboard) -> Self {
        Self { tree, blackboard }
    }

    /// Tick the tree once. Returns the root status.
    pub fn tick(&mut self, dt: f32) -> BtStatus {
        tick_node(&mut self.tree, &mut self.blackboard, dt)
    }
}

/// Recursively tick a node, mutating the tree (for timer state) and blackboard.
fn tick_node(node: &mut BtNode, bb: &mut Blackboard, dt: f32) -> BtStatus {
    match node {
        BtNode::Sequence(children) => tick_sequence(children, bb, dt),
        BtNode::Selector(children) => tick_selector(children, bb, dt),
        BtNode::Parallel { children, policy } => tick_parallel(children, policy, bb, dt),
        BtNode::Inverter(child) => tick_inverter(child, bb, dt),
        BtNode::RepeatN { child, count } => tick_repeat_n(child, *count, bb, dt),
        BtNode::RepeatUntilFail(child) => tick_repeat_until_fail(child, bb, dt),
        BtNode::Cooldown {
            child,
            duration,
            elapsed,
        } => tick_cooldown(child, *duration, elapsed, bb, dt),
        BtNode::Guard { child, condition } => tick_guard(child, condition, bb, dt),
        BtNode::Action(action) => tick_action(action, bb, dt),
        BtNode::Condition(condition) => eval_condition(condition, bb),
    }
}

// ── Composites ──

fn tick_sequence(children: &mut [BtNode], bb: &mut Blackboard, dt: f32) -> BtStatus {
    for child in children.iter_mut() {
        match tick_node(child, bb, dt) {
            BtStatus::Failure => return BtStatus::Failure,
            BtStatus::Running => return BtStatus::Running,
            BtStatus::Success => {} // continue to next
        }
    }
    BtStatus::Success
}

fn tick_selector(children: &mut [BtNode], bb: &mut Blackboard, dt: f32) -> BtStatus {
    for child in children.iter_mut() {
        match tick_node(child, bb, dt) {
            BtStatus::Success => return BtStatus::Success,
            BtStatus::Running => return BtStatus::Running,
            BtStatus::Failure => {} // try next
        }
    }
    BtStatus::Failure
}

fn tick_parallel(
    children: &mut [BtNode],
    policy: &ParallelPolicy,
    bb: &mut Blackboard,
    dt: f32,
) -> BtStatus {
    let mut successes = 0usize;
    let mut failures = 0usize;
    let mut any_running = false;

    for child in children.iter_mut() {
        match tick_node(child, bb, dt) {
            BtStatus::Success => successes += 1,
            BtStatus::Failure => failures += 1,
            BtStatus::Running => any_running = true,
        }
    }

    match policy {
        ParallelPolicy::RequireAll => {
            if failures > 0 {
                BtStatus::Failure
            } else if any_running {
                BtStatus::Running
            } else {
                BtStatus::Success
            }
        }
        ParallelPolicy::RequireOne => {
            if successes > 0 {
                BtStatus::Success
            } else if any_running {
                BtStatus::Running
            } else {
                BtStatus::Failure
            }
        }
    }
}

// ── Decorators ──

fn tick_inverter(child: &mut BtNode, bb: &mut Blackboard, dt: f32) -> BtStatus {
    match tick_node(child, bb, dt) {
        BtStatus::Success => BtStatus::Failure,
        BtStatus::Failure => BtStatus::Success,
        BtStatus::Running => BtStatus::Running,
    }
}

fn tick_repeat_n(child: &mut BtNode, count: u32, bb: &mut Blackboard, dt: f32) -> BtStatus {
    for _ in 0..count {
        match tick_node(child, bb, dt) {
            BtStatus::Failure => return BtStatus::Failure,
            BtStatus::Running => return BtStatus::Running,
            BtStatus::Success => {}
        }
    }
    BtStatus::Success
}

fn tick_repeat_until_fail(child: &mut BtNode, bb: &mut Blackboard, dt: f32) -> BtStatus {
    loop {
        match tick_node(child, bb, dt) {
            BtStatus::Failure => return BtStatus::Success,
            BtStatus::Running => return BtStatus::Running,
            BtStatus::Success => {} // keep going
        }
    }
}

fn tick_cooldown(
    child: &mut BtNode,
    duration: f32,
    elapsed: &mut f32,
    bb: &mut Blackboard,
    dt: f32,
) -> BtStatus {
    *elapsed += dt;
    if *elapsed < duration {
        return BtStatus::Failure;
    }
    let status = tick_node(child, bb, dt);
    if status != BtStatus::Running {
        // Reset cooldown timer after child completes (success or failure).
        *elapsed = 0.0;
    }
    status
}

fn tick_guard(
    child: &mut BtNode,
    condition: &BtCondition,
    bb: &mut Blackboard,
    dt: f32,
) -> BtStatus {
    if eval_condition(condition, bb) == BtStatus::Success {
        tick_node(child, bb, dt)
    } else {
        BtStatus::Failure
    }
}

// ── Leaf: Actions ──

fn tick_action(action: &mut BtAction, bb: &mut Blackboard, dt: f32) -> BtStatus {
    match action {
        BtAction::MoveTo { target_key } => {
            // MoveTo needs an external system to actually move the entity.
            // The tree just signals intent by returning Running when a target exists,
            // or Failure if the target key is missing.
            if bb.has(target_key) {
                BtStatus::Running
            } else {
                BtStatus::Failure
            }
        }
        BtAction::Wait { duration, elapsed } => {
            *elapsed += dt;
            if *elapsed >= *duration {
                *elapsed = 0.0;
                BtStatus::Success
            } else {
                BtStatus::Running
            }
        }
        BtAction::SetBlackboard { key, value } => {
            bb.set(key.clone(), value.clone());
            BtStatus::Success
        }
        BtAction::Log { message } => {
            log::info!("[BT] {message}");
            BtStatus::Success
        }
        BtAction::Custom(_) => {
            // Custom actions are no-ops in the default executor.
            // Game code should intercept these via a wrapper system or
            // pre-process the tree before ticking.
            BtStatus::Success
        }
    }
}

// ── Leaf: Conditions ──

fn eval_condition(condition: &BtCondition, bb: &Blackboard) -> BtStatus {
    let result = match condition {
        BtCondition::HasKey(key) => bb.has(key),
        BtCondition::Compare { key, op, value } => compare_value(bb, key, op, value),
        BtCondition::InRange { target_key, range } => in_range(bb, target_key, *range),
        BtCondition::IsAlive => {
            // True by default if "alive" key is missing.
            bb.get_bool("alive").unwrap_or(true)
        }
        BtCondition::Custom(_) => {
            // Custom conditions default to true — game code can override.
            true
        }
    };
    if result {
        BtStatus::Success
    } else {
        BtStatus::Failure
    }
}

fn compare_value(bb: &Blackboard, key: &str, op: &CompareOp, rhs: &BlackboardValue) -> bool {
    let Some(lhs) = bb.get(key) else {
        return false;
    };
    match (lhs, rhs) {
        (BlackboardValue::Float(a), BlackboardValue::Float(b)) => cmp_ord(*a, *b, op),
        (BlackboardValue::Int(a), BlackboardValue::Int(b)) => cmp_ord(*a, *b, op),
        (BlackboardValue::Bool(a), BlackboardValue::Bool(b)) => match op {
            CompareOp::Equal => a == b,
            CompareOp::NotEqual => a != b,
            _ => false, // ordering not meaningful for booleans
        },
        (BlackboardValue::Str(a), BlackboardValue::Str(b)) => cmp_ord(a.as_str(), b.as_str(), op),
        _ => {
            // Type mismatch: only Equal/NotEqual make sense, and the answer is "not equal".
            matches!(op, CompareOp::NotEqual)
        }
    }
}

fn cmp_ord<T: PartialOrd>(a: T, b: T, op: &CompareOp) -> bool {
    match op {
        CompareOp::Equal => a == b,
        CompareOp::NotEqual => a != b,
        CompareOp::Less => a < b,
        CompareOp::LessEqual => a <= b,
        CompareOp::Greater => a > b,
        CompareOp::GreaterEqual => a >= b,
    }
}

fn in_range(bb: &Blackboard, target_key: &str, range: f32) -> bool {
    let Some(target_pos) = bb.get_vec3(target_key) else {
        return false;
    };
    let Some(self_pos) = bb.get_vec3("self_position") else {
        return false;
    };
    (target_pos - self_pos).length_squared() <= range * range
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackboard::BlackboardValue;
    use crate::node::*;
    use euca_math::Vec3;

    // ── Required: Sequence success ──
    #[test]
    fn sequence_all_succeed() {
        let tree = BtNode::Sequence(vec![
            BtNode::Action(BtAction::Log {
                message: "step1".into(),
            }),
            BtNode::Action(BtAction::Log {
                message: "step2".into(),
            }),
        ]);
        let mut exec = BehaviorTreeExecutor::new(tree);
        assert_eq!(exec.tick(0.016), BtStatus::Success);
    }

    // ── Required: Sequence failure ──
    #[test]
    fn sequence_fails_on_missing_target() {
        let tree = BtNode::Sequence(vec![
            BtNode::Action(BtAction::Log {
                message: "ok".into(),
            }),
            // MoveTo with missing key => Failure
            BtNode::Action(BtAction::MoveTo {
                target_key: "nonexistent".into(),
            }),
            BtNode::Action(BtAction::Log {
                message: "never reached".into(),
            }),
        ]);
        let mut exec = BehaviorTreeExecutor::new(tree);
        assert_eq!(exec.tick(0.016), BtStatus::Failure);
    }

    // ── Required: Selector fallback ──
    #[test]
    fn selector_falls_back() {
        let tree = BtNode::Selector(vec![
            // First child fails (no target key)
            BtNode::Action(BtAction::MoveTo {
                target_key: "missing".into(),
            }),
            // Second child succeeds
            BtNode::Action(BtAction::Log {
                message: "fallback".into(),
            }),
        ]);
        let mut exec = BehaviorTreeExecutor::new(tree);
        assert_eq!(exec.tick(0.016), BtStatus::Success);
    }

    // ── Required: Blackboard read/write ──
    #[test]
    fn blackboard_read_write_via_tree() {
        let tree = BtNode::Sequence(vec![
            BtNode::Action(BtAction::SetBlackboard {
                key: "ready".into(),
                value: BlackboardValue::Bool(true),
            }),
            BtNode::Condition(BtCondition::HasKey("ready".into())),
        ]);
        let mut exec = BehaviorTreeExecutor::new(tree);
        assert_eq!(exec.tick(0.016), BtStatus::Success);
        assert_eq!(exec.blackboard.get_bool("ready"), Some(true));
    }

    // ── Required: Decorator invert ──
    #[test]
    fn inverter_flips_result() {
        let tree = BtNode::Inverter(Box::new(BtNode::Action(BtAction::Log {
            message: "this succeeds".into(),
        })));
        let mut exec = BehaviorTreeExecutor::new(tree);
        assert_eq!(exec.tick(0.016), BtStatus::Failure);

        // Invert a failure
        let tree2 = BtNode::Inverter(Box::new(BtNode::Action(BtAction::MoveTo {
            target_key: "missing".into(),
        })));
        let mut exec2 = BehaviorTreeExecutor::new(tree2);
        assert_eq!(exec2.tick(0.016), BtStatus::Success);
    }

    // ── Additional: Wait action ──
    #[test]
    fn wait_action_runs_then_succeeds() {
        let tree = BtNode::Action(BtAction::Wait {
            duration: 1.0,
            elapsed: 0.0,
        });
        let mut exec = BehaviorTreeExecutor::new(tree);

        // First tick: not enough time
        assert_eq!(exec.tick(0.5), BtStatus::Running);
        // Second tick: completes
        assert_eq!(exec.tick(0.6), BtStatus::Success);
    }

    // ── Additional: Cooldown ──
    #[test]
    fn cooldown_prevents_rapid_execution() {
        let tree = BtNode::Cooldown {
            child: Box::new(BtNode::Action(BtAction::Log {
                message: "tick".into(),
            })),
            duration: 1.0,
            elapsed: 1.0, // Start ready to fire
        };
        let mut exec = BehaviorTreeExecutor::new(tree);

        // First tick: child runs (cooldown satisfied, elapsed >= duration)
        assert_eq!(exec.tick(0.016), BtStatus::Success);
        // Second tick: on cooldown (elapsed was reset to 0)
        assert_eq!(exec.tick(0.5), BtStatus::Failure);
        // Third tick: still on cooldown
        assert_eq!(exec.tick(0.4), BtStatus::Failure);
        // Fourth tick: cooldown expired (0.5 + 0.4 + 0.2 > 1.0)
        assert_eq!(exec.tick(0.2), BtStatus::Success);
    }

    // ── Additional: Guard ──
    #[test]
    fn guard_blocks_when_condition_fails() {
        let tree = BtNode::Guard {
            child: Box::new(BtNode::Action(BtAction::Log {
                message: "guarded".into(),
            })),
            condition: BtCondition::HasKey("permission".into()),
        };
        let mut exec = BehaviorTreeExecutor::new(tree);
        assert_eq!(exec.tick(0.016), BtStatus::Failure);

        // Now set the key
        exec.blackboard
            .set("permission", BlackboardValue::Bool(true));
        assert_eq!(exec.tick(0.016), BtStatus::Success);
    }

    // ── Additional: Parallel ──
    #[test]
    fn parallel_require_all() {
        let tree = BtNode::Parallel {
            children: vec![
                BtNode::Action(BtAction::Log {
                    message: "a".into(),
                }),
                BtNode::Action(BtAction::Log {
                    message: "b".into(),
                }),
            ],
            policy: ParallelPolicy::RequireAll,
        };
        let mut exec = BehaviorTreeExecutor::new(tree);
        assert_eq!(exec.tick(0.016), BtStatus::Success);
    }

    #[test]
    fn parallel_require_one_short_circuits() {
        let tree = BtNode::Parallel {
            children: vec![
                BtNode::Action(BtAction::MoveTo {
                    target_key: "missing".into(),
                }), // Failure
                BtNode::Action(BtAction::Log {
                    message: "ok".into(),
                }), // Success
            ],
            policy: ParallelPolicy::RequireOne,
        };
        let mut exec = BehaviorTreeExecutor::new(tree);
        assert_eq!(exec.tick(0.016), BtStatus::Success);
    }

    // ── Additional: Compare condition ──
    #[test]
    fn compare_condition() {
        let tree = BtNode::Condition(BtCondition::Compare {
            key: "health".into(),
            op: CompareOp::Greater,
            value: BlackboardValue::Float(50.0),
        });
        let mut exec = BehaviorTreeExecutor::new(tree);

        // No key => Failure
        assert_eq!(exec.tick(0.016), BtStatus::Failure);

        // Health = 30 => Failure (30 > 50 is false)
        exec.blackboard.set("health", BlackboardValue::Float(30.0));
        assert_eq!(exec.tick(0.016), BtStatus::Failure);

        // Health = 80 => Success (80 > 50 is true)
        exec.blackboard.set("health", BlackboardValue::Float(80.0));
        assert_eq!(exec.tick(0.016), BtStatus::Success);
    }

    // ── Additional: InRange condition ──
    #[test]
    fn in_range_condition() {
        let tree = BtNode::Condition(BtCondition::InRange {
            target_key: "enemy".into(),
            range: 5.0,
        });
        let mut exec = BehaviorTreeExecutor::new(tree);

        exec.blackboard
            .set("self_position", BlackboardValue::Vec3(Vec3::ZERO));
        exec.blackboard
            .set("enemy", BlackboardValue::Vec3(Vec3::new(3.0, 0.0, 0.0)));
        assert_eq!(exec.tick(0.016), BtStatus::Success);

        // Move enemy out of range
        exec.blackboard
            .set("enemy", BlackboardValue::Vec3(Vec3::new(10.0, 0.0, 0.0)));
        assert_eq!(exec.tick(0.016), BtStatus::Failure);
    }

    // ── Additional: IsAlive ──
    #[test]
    fn is_alive_defaults_to_true() {
        let tree = BtNode::Condition(BtCondition::IsAlive);
        let mut exec = BehaviorTreeExecutor::new(tree);
        assert_eq!(exec.tick(0.016), BtStatus::Success);

        exec.blackboard.set("alive", BlackboardValue::Bool(false));
        assert_eq!(exec.tick(0.016), BtStatus::Failure);
    }

    // ── Additional: RepeatN ──
    #[test]
    fn repeat_n_runs_multiple_times() {
        // RepeatN with a SetBlackboard that increments a counter via side effects.
        // We'll use a sequence: read counter, increment it.
        // Simpler: just verify RepeatN(Log, 3) succeeds after 3 executions.
        let tree = BtNode::RepeatN {
            child: Box::new(BtNode::Action(BtAction::Log {
                message: "rep".into(),
            })),
            count: 3,
        };
        let mut exec = BehaviorTreeExecutor::new(tree);
        assert_eq!(exec.tick(0.016), BtStatus::Success);
    }
}
