//! Behavior trees and blackboard for NPC AI.
//!
//! This crate provides an ECS-native behavior tree system. Attach a
//! [`BehaviorTreeExecutor`] component to any entity, then run
//! [`behavior_tree_system`] each frame.
//!
//! # Architecture
//!
//! - **Tree**: Enum-based recursive structure ([`BtNode`]). No trait objects.
//! - **Blackboard**: Per-entity typed key-value store ([`Blackboard`]).
//! - **Executor**: Component holding tree + blackboard + running state.
//! - **System**: `fn(world: &mut World, dt: f32)` — ticks all executors.
//! - **Builder**: Fluent API for constructing trees ergonomically.

pub mod blackboard;
pub mod builder;
pub mod executor;
pub mod node;
pub mod system;

// Re-export key types at crate root.
pub use blackboard::{Blackboard, BlackboardValue};
pub use builder::BtBuilder;
pub use executor::BehaviorTreeExecutor;
pub use node::{BtAction, BtCondition, BtNode, BtStatus, CompareOp, ParallelPolicy};
pub use system::behavior_tree_system;
