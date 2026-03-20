//! Behavior tree node definitions.
//!
//! The tree is a recursive enum — no trait objects, no allocator, just data.
//! This makes trees serializable, cloneable, and easy to inspect.

use crate::blackboard::BlackboardValue;

/// Result of ticking a behavior tree node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BtStatus {
    /// The node completed successfully.
    Success,
    /// The node failed.
    Failure,
    /// The node needs more ticks to finish (e.g. a `Wait` action mid-timer).
    Running,
}

/// Policy for [`BtNode::Parallel`] completion.
#[derive(Clone, Debug, PartialEq)]
pub enum ParallelPolicy {
    /// Succeed only when **all** children succeed. Fail if any child fails.
    RequireAll,
    /// Succeed as soon as **one** child succeeds.
    RequireOne,
}

/// Comparison operators for [`BtCondition::Compare`].
#[derive(Clone, Debug, PartialEq)]
pub enum CompareOp {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

/// Leaf actions that the behavior tree can execute.
#[derive(Clone, Debug, PartialEq)]
pub enum BtAction {
    /// Move toward a position stored under `target_key` in the blackboard.
    MoveTo { target_key: String },
    /// Wait for `duration` seconds. `elapsed` tracks progress across ticks.
    Wait { duration: f32, elapsed: f32 },
    /// Write a value into the blackboard.
    SetBlackboard { key: String, value: BlackboardValue },
    /// Log a message (via the `log` crate at info level).
    Log { message: String },
    /// Named extension point — game code maps the string to custom logic.
    Custom(String),
}

/// Conditions that evaluate to `Success` (true) or `Failure` (false).
#[derive(Clone, Debug, PartialEq)]
pub enum BtCondition {
    /// Succeeds if the blackboard contains `key`.
    HasKey(String),
    /// Compares a blackboard value against a literal.
    Compare {
        key: String,
        op: CompareOp,
        value: BlackboardValue,
    },
    /// Succeeds if a `Vec3` stored under `target_key` is within `range` of the
    /// entity's position (read from `"self_position"` key).
    InRange { target_key: String, range: f32 },
    /// Succeeds if `"alive"` key is `Bool(true)` (or missing — alive by default).
    IsAlive,
    /// Named extension point for game-specific conditions.
    Custom(String),
}

/// A single node in the behavior tree.
///
/// Composite, decorator, leaf — all represented as enum variants.
/// Recursive via `Vec<BtNode>` or `Box<BtNode>`.
#[derive(Clone, Debug, PartialEq)]
pub enum BtNode {
    // ── Composites ──
    /// Runs children left-to-right. Succeeds if **all** succeed.
    /// Fails immediately when any child fails.
    Sequence(Vec<BtNode>),

    /// Runs children left-to-right. Succeeds as soon as **one** succeeds.
    /// Fails only if all children fail.
    Selector(Vec<BtNode>),

    /// Ticks all children every frame.
    Parallel {
        children: Vec<BtNode>,
        policy: ParallelPolicy,
    },

    // ── Decorators ──
    /// Inverts the child's result: Success becomes Failure and vice versa.
    /// Running passes through unchanged.
    Inverter(Box<BtNode>),

    /// Repeats the child `count` times. Fails immediately if the child fails.
    RepeatN { child: Box<BtNode>, count: u32 },

    /// Repeats the child until it returns Failure, then succeeds.
    RepeatUntilFail(Box<BtNode>),

    /// Prevents the child from running more often than once per `duration` seconds.
    /// Returns Failure while on cooldown.
    Cooldown {
        child: Box<BtNode>,
        duration: f32,
        elapsed: f32,
    },

    /// Only ticks the child if `condition` passes; otherwise returns Failure.
    Guard {
        child: Box<BtNode>,
        condition: BtCondition,
    },

    // ── Leaves ──
    /// An action to execute.
    Action(BtAction),

    /// A condition to evaluate.
    Condition(BtCondition),
}
