//! Animation system for the Euca engine: blending, state machines, root motion, events, montages.
//!
//! This crate builds on the clip/skeleton types from `euca-asset` and provides:
//! - **Blending**: Per-bone lerp/slerp with weighted layers.
//! - **State machines**: Parametric transitions with crossfade blending.
//! - **Blend spaces**: 1D parametric blending (e.g., speed -> walk/run).
//! - **Root motion**: Extract translation/rotation delta from hip bone.
//! - **Events**: Time-stamped callbacks emitted as ECS events.
//! - **Montages**: Interruptable one-shot overlays (attacks, reloads, emotes).

pub mod blend;
pub mod blend_space;
pub mod event;
pub mod montage;
pub mod root_motion;
pub mod state_machine;
pub mod systems;

// Re-export key types for convenience.
pub use blend::{BlendLayer, blend_poses, crossfade_layers};
pub use blend_space::{BlendSpace1D, BlendSpaceSample};
pub use event::{AnimationEvent, AnimationEventLibrary, AnimationEventMarker, ClipEventMarkers};
pub use montage::{ActiveMontage, MontageDefinition, MontagePhase, MontagePlayer};
pub use root_motion::{RootMotionConfig, RootMotionDelta, RootMotionOutput, extract_root_motion};
pub use state_machine::{
    ActiveTransition, AnimationParameters, AnimationState, AnimationStateMachine,
    StateTransition, TransitionCondition,
};
pub use systems::{
    animation_blend_system, animation_event_system, animation_state_machine_system,
    root_motion_system,
};

// Re-export foundational types from euca-asset for convenience.
pub use euca_asset::animation::{AnimationClipData, AnimationProperty};
pub use euca_asset::skeleton::Skeleton;
pub use euca_asset::systems::{AnimationLibrary, BoneTransforms, SkeletalAnimator};
