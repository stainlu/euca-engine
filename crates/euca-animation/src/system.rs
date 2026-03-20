//! The main animation evaluation system.
//!
//! Each frame this system:
//! 1. Evaluates the state machine (transitions, playback time)
//! 2. Samples clips into poses
//! 3. Blends crossfade poses
//! 4. Overlays montage poses
//! 5. Extracts root motion (if configured)
//! 6. Fires animation events
//! 7. Computes final bone matrices

use euca_asset::animation::AnimationClipData;
use euca_asset::skeleton::Skeleton;
use euca_asset::systems::{AnimationLibrary, BoneTransforms};
use euca_ecs::{Entity, Query, World};
use euca_math::Transform;

use crate::blend::AnimationBlender;
use crate::clip::{AnimPose, sample_clip};
use crate::event::{AnimationEventLibrary, FiredAnimationEvents, FiredEvent};
use crate::montage::MontagePlayer;
use crate::root_motion::{RootMotionDelta, RootMotionReceiver, extract_root_motion};
use crate::state_machine::AnimStateMachine;

/// ECS component: the main animator attached to skeletal entities.
///
/// Replaces `SkeletalAnimator` with full state machine + montage support.
#[derive(Clone, Debug)]
pub struct Animator {
    /// Which skeleton to use (index into `AnimationLibrary.skeletons`).
    pub skeleton_index: usize,
    /// The previous frame's pose (for root motion delta computation).
    pub previous_pose: Option<AnimPose>,
}

impl Animator {
    pub fn new(skeleton_index: usize) -> Self {
        Self {
            skeleton_index,
            previous_pose: None,
        }
    }
}

/// Main animation system. Call once per frame.
///
/// Evaluates state machines, blends poses, handles montages, extracts root
/// motion, fires events, and writes bone matrices.
pub fn animation_evaluate_system(world: &mut World, dt: f32) {
    // Collect entities that have an Animator
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &Animator)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    // Clone the library data we need (clips, skeletons)
    let library_data: Option<(Vec<AnimationClipData>, Vec<Skeleton>)> = world
        .resource::<AnimationLibrary>()
        .map(|lib| (lib.clips.clone(), lib.skeletons.clone()));

    let (clips, skeletons) = match library_data {
        Some(d) => d,
        None => return,
    };

    let clip_durations: Vec<f32> = clips.iter().map(|c| c.duration).collect();

    // Clone event library if available
    let event_library: Option<AnimationEventLibrary> =
        world.resource::<AnimationEventLibrary>().cloned();

    for entity in entities {
        // Read animator data
        let skeleton_index = match world.get::<Animator>(entity) {
            Some(a) => a.skeleton_index,
            None => continue,
        };

        let skeleton = match skeletons.get(skeleton_index) {
            Some(s) => s,
            None => continue,
        };

        // Step 1: Evaluate state machine
        let (sm_pose, prev_time, current_clip_idx) =
            evaluate_state_machine(world, entity, dt, &clip_durations, &clips, skeleton);

        // Step 2: Overlay montage
        let mut final_pose = overlay_montage(world, entity, dt, &clips, skeleton, sm_pose);

        // Step 3: Extract root motion
        let root_motion_delta = extract_root_motion_for_entity(world, entity, &mut final_pose);

        if let Some(delta) = root_motion_delta {
            if let Some(existing) = world.get_mut::<RootMotionDelta>(entity) {
                *existing = delta;
            } else {
                world.insert(entity, delta);
            }
        }

        // Store current pose as previous for next frame
        if let Some(animator) = world.get_mut::<Animator>(entity) {
            animator.previous_pose = Some(final_pose.clone());
        }

        // Step 4: Fire animation events
        if let Some(ref event_lib) = event_library {
            fire_events(world, entity, event_lib, current_clip_idx, prev_time);
        }

        // Step 5: Compute bone matrices
        let matrices = skeleton.compute_joint_matrices(&final_pose.joints);
        if let Some(bones) = world.get_mut::<BoneTransforms>(entity) {
            bones.matrices = matrices;
        } else {
            world.insert(entity, BoneTransforms { matrices });
        }
    }
}

/// Evaluate the state machine for an entity and return the blended pose.
/// Returns `(pose, prev_time, current_clip_index)`.
fn evaluate_state_machine(
    world: &mut World,
    entity: Entity,
    dt: f32,
    clip_durations: &[f32],
    clips: &[AnimationClipData],
    skeleton: &Skeleton,
) -> (AnimPose, f32, Option<usize>) {
    // Read state machine pre-update state
    let sm_data = world
        .get::<AnimStateMachine>(entity)
        .map(|sm| (sm.current_time, sm.current_clip_index()));

    let prev_time = match sm_data {
        Some((t, _)) => t,
        None => {
            return (AnimPose::from_skeleton(skeleton), 0.0, None);
        }
    };

    // Advance the state machine
    if let Some(sm) = world.get_mut::<AnimStateMachine>(entity) {
        sm.update(dt, clip_durations);
    }

    // Read updated state after advancing
    let updated_data = world.get::<AnimStateMachine>(entity).map(|sm| {
        (
            sm.current_time,
            sm.current_clip_index(),
            sm.crossfade_info(),
        )
    });

    let (current_time, current_clip, crossfade) = match updated_data {
        Some(d) => d,
        None => return (AnimPose::from_skeleton(skeleton), 0.0, None),
    };

    // Sample the current clip
    let current_pose = match current_clip {
        Some(clip_idx) if clip_idx < clips.len() => {
            sample_clip(&clips[clip_idx], skeleton, current_time)
        }
        _ => AnimPose::from_skeleton(skeleton),
    };

    // If crossfading, blend with the outgoing pose
    let pose = if let Some((from_clip_idx, from_time, outgoing_w, incoming_w)) = crossfade {
        let from_pose = if from_clip_idx < clips.len() {
            sample_clip(&clips[from_clip_idx], skeleton, from_time)
        } else {
            AnimPose::from_skeleton(skeleton)
        };

        let mut blender = AnimationBlender::new();
        blender.add_layer(from_pose, outgoing_w);
        blender.add_layer(current_pose, incoming_w);
        blender.evaluate(skeleton.joints.len())
    } else {
        current_pose
    };

    (pose, prev_time, current_clip)
}

/// Overlay montage pose on top of the base state machine pose.
fn overlay_montage(
    world: &mut World,
    entity: Entity,
    dt: f32,
    clips: &[AnimationClipData],
    skeleton: &Skeleton,
    base_pose: AnimPose,
) -> AnimPose {
    // Advance montage
    if let Some(player) = world.get_mut::<MontagePlayer>(entity) {
        player.advance(dt);
    }

    // Read montage state
    let montage_data = world.get::<MontagePlayer>(entity).and_then(|player| {
        let active = player.active()?;
        Some((
            active.montage.clip_index,
            active.time,
            active.weight(),
            active.montage.bone_mask.clone(),
        ))
    });

    let (clip_idx, montage_time, montage_weight, bone_mask) = match montage_data {
        Some(d) => d,
        None => return base_pose,
    };

    if montage_weight <= 0.0 || clip_idx >= clips.len() {
        return base_pose;
    }

    let montage_pose = sample_clip(&clips[clip_idx], skeleton, montage_time);

    // Blend montage over base pose
    match bone_mask {
        None => base_pose.blend(&montage_pose, montage_weight),
        Some(mask) => {
            let mut result = base_pose;
            for &bone_idx in &mask {
                if bone_idx < result.joints.len() && bone_idx < montage_pose.joints.len() {
                    let base = &result.joints[bone_idx];
                    let overlay = &montage_pose.joints[bone_idx];
                    result.joints[bone_idx] = Transform {
                        translation: base.translation.lerp(overlay.translation, montage_weight),
                        rotation: base.rotation.slerp(overlay.rotation, montage_weight),
                        scale: base.scale.lerp(overlay.scale, montage_weight),
                    };
                }
            }
            result
        }
    }
}

/// Extract root motion for an entity if it has a RootMotionReceiver.
fn extract_root_motion_for_entity(
    world: &World,
    entity: Entity,
    current_pose: &mut AnimPose,
) -> Option<RootMotionDelta> {
    let receiver = world.get::<RootMotionReceiver>(entity)?;
    let previous_pose = world
        .get::<Animator>(entity)
        .and_then(|a| a.previous_pose.as_ref())?;

    Some(extract_root_motion(receiver, previous_pose, current_pose))
}

/// Fire animation events that occurred between prev_time and current_time.
fn fire_events(
    world: &mut World,
    entity: Entity,
    event_library: &AnimationEventLibrary,
    current_clip_idx: Option<usize>,
    prev_time: f32,
) {
    let clip_idx = match current_clip_idx {
        Some(idx) => idx,
        None => return,
    };

    let clip_events = match event_library.get_clip_events(clip_idx) {
        Some(events) => events,
        None => return,
    };

    let current_time = world
        .get::<AnimStateMachine>(entity)
        .map(|sm| sm.current_time)
        .unwrap_or(0.0);

    let fired = clip_events.query(prev_time, current_time);
    if fired.is_empty() {
        return;
    }

    let fired_events: Vec<FiredEvent> = fired
        .iter()
        .map(|e| FiredEvent {
            name: e.name.clone(),
            payload: e.payload.clone(),
            clip_index: clip_idx,
        })
        .collect();

    if let Some(existing) = world.get_mut::<FiredAnimationEvents>(entity) {
        existing.events = fired_events;
    } else {
        world.insert(
            entity,
            FiredAnimationEvents {
                events: fired_events,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_asset::animation::{AnimationChannel, AnimationProperty, KeyframeValue};
    use euca_asset::skeleton::Joint;
    use euca_math::{Mat4, Vec3};

    fn test_skeleton(joint_count: usize) -> Skeleton {
        Skeleton {
            joints: (0..joint_count)
                .map(|i| Joint {
                    name: format!("joint_{i}"),
                    parent: if i == 0 { None } else { Some(0) },
                    local_transform: Transform::IDENTITY,
                })
                .collect(),
            inverse_bind_matrices: vec![Mat4::IDENTITY; joint_count],
            joint_node_indices: (0..joint_count).collect(),
        }
    }

    fn test_clip(name: &str, duration: f32) -> AnimationClipData {
        AnimationClipData {
            name: name.into(),
            duration,
            channels: vec![AnimationChannel {
                joint_index: 0,
                property: AnimationProperty::Translation,
                times: vec![0.0, duration],
                values: vec![
                    KeyframeValue::Vec3(Vec3::ZERO),
                    KeyframeValue::Vec3(Vec3::new(duration * 10.0, 0.0, 0.0)),
                ],
            }],
        }
    }

    #[test]
    fn animator_creation() {
        let animator = Animator::new(0);
        assert_eq!(animator.skeleton_index, 0);
        assert!(animator.previous_pose.is_none());
    }

    #[test]
    fn full_system_pipeline() {
        let mut world = World::new();

        let mut lib = AnimationLibrary::default();
        lib.add_clip(test_clip("idle", 1.0));
        lib.add_skeleton(test_skeleton(2));
        world.insert_resource(lib);

        let entity = world.spawn(Animator::new(0));
        let mut sm = AnimStateMachine::new(0);
        sm.add_state("idle", 0);
        world.insert(entity, sm);

        animation_evaluate_system(&mut world, 0.5);

        let bones = world.get::<BoneTransforms>(entity);
        assert!(bones.is_some());
        assert_eq!(bones.unwrap().matrices.len(), 2);
    }

    #[test]
    fn system_without_library_is_noop() {
        let mut world = World::new();
        let entity = world.spawn(Animator::new(0));
        let mut sm = AnimStateMachine::new(0);
        sm.add_state("idle", 0);
        world.insert(entity, sm);

        animation_evaluate_system(&mut world, 0.016);
        assert!(world.get::<BoneTransforms>(entity).is_none());
    }
}
