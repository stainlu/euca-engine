//! Ergonomic builder API for constructing behavior trees.
//!
//! # Example
//! ```ignore
//! use euca_ai::builder::BtBuilder;
//!
//! let tree = BtBuilder::selector()
//!     .child(BtBuilder::sequence()
//!         .condition_in_range("enemy", 5.0)
//!         .action_log("Attacking!")
//!         .build())
//!     .child(BtBuilder::sequence()
//!         .action_move_to("patrol_point")
//!         .build())
//!     .build();
//! ```

use crate::blackboard::BlackboardValue;
use crate::node::*;

/// Builder for constructing [`BtNode`] trees with a fluent API.
pub struct BtBuilder {
    children: Vec<BtNode>,
    kind: BuilderKind,
}

enum BuilderKind {
    Sequence,
    Selector,
    Parallel(ParallelPolicy),
}

impl BtBuilder {
    /// Start building a Sequence node.
    pub fn sequence() -> Self {
        Self {
            children: Vec::new(),
            kind: BuilderKind::Sequence,
        }
    }

    /// Start building a Selector node.
    pub fn selector() -> Self {
        Self {
            children: Vec::new(),
            kind: BuilderKind::Selector,
        }
    }

    /// Start building a Parallel node with the given policy.
    pub fn parallel(policy: ParallelPolicy) -> Self {
        Self {
            children: Vec::new(),
            kind: BuilderKind::Parallel(policy),
        }
    }

    /// Add an arbitrary child node.
    pub fn child(mut self, node: BtNode) -> Self {
        self.children.push(node);
        self
    }

    // ── Action shortcuts ──

    /// Add a `MoveTo` action.
    pub fn action_move_to(self, target_key: impl Into<String>) -> Self {
        self.child(BtNode::Action(BtAction::MoveTo {
            target_key: target_key.into(),
        }))
    }

    /// Add a `Wait` action.
    pub fn action_wait(self, duration: f32) -> Self {
        self.child(BtNode::Action(BtAction::Wait {
            duration,
            elapsed: 0.0,
        }))
    }

    /// Add a `SetBlackboard` action.
    pub fn action_set(self, key: impl Into<String>, value: BlackboardValue) -> Self {
        self.child(BtNode::Action(BtAction::SetBlackboard {
            key: key.into(),
            value,
        }))
    }

    /// Add a `Log` action.
    pub fn action_log(self, message: impl Into<String>) -> Self {
        self.child(BtNode::Action(BtAction::Log {
            message: message.into(),
        }))
    }

    /// Add a named `Custom` action.
    pub fn action_custom(self, name: impl Into<String>) -> Self {
        self.child(BtNode::Action(BtAction::Custom(name.into())))
    }

    // ── Condition shortcuts ──

    /// Add a `HasKey` condition.
    pub fn condition_has_key(self, key: impl Into<String>) -> Self {
        self.child(BtNode::Condition(BtCondition::HasKey(key.into())))
    }

    /// Add a `Compare` condition.
    pub fn condition_compare(
        self,
        key: impl Into<String>,
        op: CompareOp,
        value: BlackboardValue,
    ) -> Self {
        self.child(BtNode::Condition(BtCondition::Compare {
            key: key.into(),
            op,
            value,
        }))
    }

    /// Add an `InRange` condition.
    pub fn condition_in_range(self, target_key: impl Into<String>, range: f32) -> Self {
        self.child(BtNode::Condition(BtCondition::InRange {
            target_key: target_key.into(),
            range,
        }))
    }

    /// Add an `IsAlive` condition.
    pub fn condition_is_alive(self) -> Self {
        self.child(BtNode::Condition(BtCondition::IsAlive))
    }

    /// Consume the builder and produce the final [`BtNode`].
    pub fn build(self) -> BtNode {
        match self.kind {
            BuilderKind::Sequence => BtNode::Sequence(self.children),
            BuilderKind::Selector => BtNode::Selector(self.children),
            BuilderKind::Parallel(policy) => BtNode::Parallel {
                children: self.children,
                policy,
            },
        }
    }
}

// ── Standalone decorator constructors ──

/// Wrap a node in an [`BtNode::Inverter`].
pub fn invert(node: BtNode) -> BtNode {
    BtNode::Inverter(Box::new(node))
}

/// Wrap a node in a [`BtNode::RepeatN`].
pub fn repeat_n(node: BtNode, count: u32) -> BtNode {
    BtNode::RepeatN {
        child: Box::new(node),
        count,
    }
}

/// Wrap a node in a [`BtNode::RepeatUntilFail`].
pub fn repeat_until_fail(node: BtNode) -> BtNode {
    BtNode::RepeatUntilFail(Box::new(node))
}

/// Wrap a node in a [`BtNode::Cooldown`].
pub fn cooldown(node: BtNode, duration: f32) -> BtNode {
    BtNode::Cooldown {
        child: Box::new(node),
        duration,
        elapsed: duration, // Start ready to fire on first tick.
    }
}

/// Wrap a node in a [`BtNode::Guard`].
pub fn guard(node: BtNode, condition: BtCondition) -> BtNode {
    BtNode::Guard {
        child: Box::new(node),
        condition,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_produces_correct_tree() {
        let tree = BtBuilder::selector()
            .child(
                BtBuilder::sequence()
                    .condition_has_key("enemy")
                    .action_log("Attack!")
                    .build(),
            )
            .child(BtBuilder::sequence().action_move_to("patrol_point").build())
            .build();

        // Should be a Selector with 2 Sequence children
        match &tree {
            BtNode::Selector(children) => {
                assert_eq!(children.len(), 2);
                assert!(matches!(&children[0], BtNode::Sequence(_)));
                assert!(matches!(&children[1], BtNode::Sequence(_)));
            }
            _ => panic!("Expected Selector"),
        }
    }

    #[test]
    fn decorator_helpers() {
        let log = BtNode::Action(BtAction::Log {
            message: "hi".into(),
        });
        let inverted = invert(log.clone());
        assert!(matches!(inverted, BtNode::Inverter(_)));

        let repeated = repeat_n(log.clone(), 5);
        match &repeated {
            BtNode::RepeatN { count, .. } => assert_eq!(*count, 5),
            _ => panic!("Expected RepeatN"),
        }

        let guarded = guard(log, BtCondition::IsAlive);
        assert!(matches!(guarded, BtNode::Guard { .. }));
    }
}
