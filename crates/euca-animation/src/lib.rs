//! Runtime animation system for the Euca engine.
//!
//! This crate handles animation evaluation at runtime: state machines,
//! pose blending, blend spaces, root motion extraction, animation events,
//! and montages. It builds on top of the clip data loaded by `euca-asset`.
//!
//! # Architecture
//!
//! ```text
//! AnimStateMachine -> selects clip + time
//!         |
//!    sample_clip -> AnimPose
//!         |
//! AnimationBlender -> blends crossfade poses
//!         |
//! MontagePlayer -> overlays one-shot animations
//!         |
//! RootMotionReceiver -> extracts entity-level movement
//!         |
//! Skeleton::compute_joint_matrices -> BoneTransforms
//! ```
//!
//! # Key types
//!
//! - [`Animator`] -- ECS component replacing `SkeletalAnimator`
//! - [`AnimStateMachine`] -- parametric state machine with conditions
//! - [`AnimPose`] -- sampled per-joint transforms (the central data type)
//! - [`AnimationBlender`] -- multi-layer pose blending
//! - [`BlendSpace1D`] / [`BlendSpace2D`] -- parametric blend spaces
//! - [`MontagePlayer`] -- one-shot overlay animations
//! - [`RootMotionReceiver`] -- root bone to entity transform extraction
//! - [`AnimationEvent`] -- time-stamped clip callbacks

pub mod blend;
pub mod blend_space;
pub mod clip;
pub mod event;
pub mod montage;
pub mod root_motion;
pub mod state_machine;
pub mod system;

// Re-exports for ergonomic access
pub use blend::{AnimationBlender, Crossfade};
pub use blend_space::{BlendSample1D, BlendSample2D, BlendSpace1D, BlendSpace2D};
pub use clip::AnimPose;
pub use event::{
    AnimationEvent, AnimationEventLibrary, ClipEvents, EventValue, FiredAnimationEvents, FiredEvent,
};
pub use montage::{ActiveMontage, AnimationMontage, MontagePlayer};
pub use root_motion::{RootMotionDelta, RootMotionReceiver};
pub use state_machine::{
    AnimState, AnimStateMachine, CompareOp, ParamValue, StateTransition, TransitionCondition,
};
pub use system::{Animator, animation_evaluate_system};
