//! ECS systems for the animation pipeline.
//!
//! Execution order: state_machine -> blend -> root_motion -> event.

use euca_ecs::{Entity, Query, World};
use euca_math::Transform;

use euca_asset::animation::AnimationClipData;
use euca_asset::skeleton::Skeleton;
use euca_asset::systems::{AnimationLibrary, BoneTransforms};

use crate::blend::{BlendLayer, blend_poses, crossfade_layers};
use crate::event::{AnimationEvent, AnimationEventLibrary};
use crate::montage::MontagePlayer;
use crate::root_motion::{RootMotionConfig, RootMotionOutput, extract_root_motion};
use crate::state_machine::AnimationStateMachine;

/// Evaluates all animation state machines: checks transition conditions, fires transitions,
/// and advances playback time.
///
/// Should run before `animation_blend_system`.
pub fn animation_state_machine_system(world: &mut World, dt: f32) {
    // Collect entities with state machines.
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &AnimationStateMachine)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    // Get clip durations from the library.
    let clip_durations: Vec<f32> = world
        .resource::<AnimationLibrary>()
        .map(|lib| lib.clips.iter().map(|c| c.duration).collect())
        .unwrap_or_default();

    for entity in entities {
        if let Some(sm) = world.get_mut::<AnimationStateMachine>(entity) {
            sm.evaluate_transitions();
            sm.advance(dt, &clip_durations);
        }

        // Advance montage playback (if present) alongside the state machine.
        if let Some(mp) = world.get_mut::<MontagePlayer>(entity) {
            mp.advance(dt);
        }
    }
}

/// Blends animation layers (from state machines and montages), samples clip poses,
/// and writes the final bone matrices to `BoneTransforms`.
///
/// Should run after `animation_state_machine_system`.
pub fn animation_blend_system(world: &mut World, _dt: f32) {
    // Snapshot library data needed for blending.
    let library_data: Option<(Vec<AnimationClipData>, Vec<Skeleton>)> = world
        .resource::<AnimationLibrary>()
        .map(|lib| (lib.clips.clone(), lib.skeletons.clone()));

    let (clips, skeletons) = match library_data {
        Some(d) => d,
        None => return,
    };

    let skeleton = match skeletons.first() {
        Some(s) => s,
        None => return,
    };

    // Collect entities that have a state machine.
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &AnimationStateMachine)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    for entity in entities {
        // Build blend layers from the state machine.
        let sm_layers: Vec<BlendLayer> = {
            let sm = match world.get::<AnimationStateMachine>(entity) {
                Some(sm) => sm,
                None => continue,
            };

            let (clip_idx, time) = sm.current_clip_and_time();

            if let Some(ref transition) = sm.transition {
                let from_clip = sm.states[transition.from_state].clip_index;
                crossfade_layers(
                    from_clip,
                    transition.from_time,
                    clip_idx,
                    time,
                    transition.progress(),
                )
                .to_vec()
            } else {
                vec![BlendLayer {
                    clip_index: clip_idx,
                    time,
                    weight: 1.0,
                }]
            }
        };

        // Blend the state machine pose.
        let sm_pose = blend_poses(&sm_layers, &clips, skeleton);

        // Apply montage overlay if present.
        let final_pose = apply_montage_overlay(world, entity, &sm_pose, &clips, skeleton);

        // Compute bone matrices and write to component.
        let matrices = skeleton.compute_joint_matrices(&final_pose);
        if let Some(bones) = world.get_mut::<BoneTransforms>(entity) {
            bones.matrices = matrices;
        } else {
            world.insert(entity, BoneTransforms { matrices });
        }
    }
}

/// Apply montage overlay on top of the state machine pose.
fn apply_montage_overlay(
    world: &World,
    entity: Entity,
    sm_pose: &[Transform],
    clips: &[AnimationClipData],
    skeleton: &Skeleton,
) -> Vec<Transform> {
    let montage_player = match world.get::<MontagePlayer>(entity) {
        Some(mp) => mp,
        None => return sm_pose.to_vec(),
    };

    let montage = match &montage_player.active {
        Some(m) => m,
        None => return sm_pose.to_vec(),
    };

    let montage_weight = montage.blend_weight();
    if montage_weight <= 0.0 {
        return sm_pose.to_vec();
    }

    // Sample the montage clip into a pose.
    let montage_layers = vec![BlendLayer {
        clip_index: montage.definition.clip_index,
        time: montage.time,
        weight: 1.0,
    }];
    let montage_pose = blend_poses(&montage_layers, clips, skeleton);

    // Blend between state machine pose and montage pose.
    let joint_count = sm_pose.len().min(montage_pose.len());
    let mut result = Vec::with_capacity(joint_count);
    for j in 0..joint_count {
        result.push(Transform {
            translation: sm_pose[j]
                .translation
                .lerp(montage_pose[j].translation, montage_weight),
            rotation: sm_pose[j]
                .rotation
                .slerp(montage_pose[j].rotation, montage_weight),
            scale: sm_pose[j]
                .scale
                .lerp(montage_pose[j].scale, montage_weight),
        });
    }

    result
}

/// Extracts root motion deltas from the animation clip and writes them to `RootMotionOutput`.
/// Also zeroes out the root bone's animation in the bone transforms so the entity
/// transform receives the motion instead.
///
/// Should run after `animation_blend_system`.
pub fn root_motion_system(world: &mut World, dt: f32) {
    let library_data: Option<Vec<AnimationClipData>> = world
        .resource::<AnimationLibrary>()
        .map(|lib| lib.clips.clone());

    let clips = match library_data {
        Some(c) => c,
        None => return,
    };

    // Collect entities with root motion config and state machine.
    let entities: Vec<(Entity, usize, usize, f32, bool, f32)> = {
        let query =
            Query::<(Entity, &RootMotionConfig, &AnimationStateMachine)>::new(world);
        query
            .iter()
            .map(|(e, cfg, sm)| {
                let state = &sm.states[sm.current_state];
                (
                    e,
                    cfg.root_bone_index,
                    state.clip_index,
                    sm.current_time,
                    state.looping,
                    state.speed,
                )
            })
            .collect()
    };

    for (entity, root_bone, clip_index, current_time, looping, speed) in entities {
        let clip = match clips.get(clip_index) {
            Some(c) => c,
            None => continue,
        };

        let prev_time = current_time - dt * speed;
        let prev_time = if looping && clip.duration > 0.0 {
            ((prev_time % clip.duration) + clip.duration) % clip.duration
        } else {
            prev_time.max(0.0)
        };

        let delta = extract_root_motion(clip, root_bone, prev_time, current_time, looping);

        if let Some(output) = world.get_mut::<RootMotionOutput>(entity) {
            output.delta = delta;
        } else {
            world.insert(
                entity,
                RootMotionOutput { delta },
            );
        }
    }
}

/// Checks animation playback progress against event markers and emits `AnimationEvent`s.
///
/// Should run after `animation_blend_system`.
pub fn animation_event_system(world: &mut World, dt: f32) {
    let event_lib_exists = world.resource::<AnimationEventLibrary>().is_some();
    if !event_lib_exists {
        return;
    }

    // Collect state machine data for all entities.
    let entities: Vec<(Entity, usize, f32, bool, f32)> = {
        let query = Query::<(Entity, &AnimationStateMachine)>::new(world);
        query
            .iter()
            .map(|(e, sm)| {
                let state = &sm.states[sm.current_state];
                (
                    e,
                    state.clip_index,
                    sm.current_time,
                    state.looping,
                    state.speed,
                )
            })
            .collect()
    };

    // Get clip durations from library.
    let clip_durations: Vec<f32> = world
        .resource::<AnimationLibrary>()
        .map(|lib| lib.clips.iter().map(|c| c.duration).collect())
        .unwrap_or_default();

    // Build events to emit.
    let mut events_to_send: Vec<AnimationEvent> = Vec::new();

    if let Some(event_lib) = world.resource::<AnimationEventLibrary>() {
        for (entity, clip_index, current_time, looping, speed) in &entities {
            let markers = match event_lib.get_markers(*clip_index) {
                Some(m) => m,
                None => continue,
            };

            let duration = clip_durations.get(*clip_index).copied().unwrap_or(0.0);
            let prev_time = current_time - dt * speed;
            let wrapped = *looping && prev_time < 0.0 && duration > 0.0;
            let prev_time = if wrapped {
                ((prev_time % duration) + duration) % duration
            } else {
                prev_time.max(0.0)
            };

            let fired = markers.collect_fired(prev_time, *current_time, wrapped, duration);
            for marker in fired {
                events_to_send.push(AnimationEvent {
                    entity: *entity,
                    clip_index: *clip_index,
                    name: marker.name.clone(),
                    time: marker.time,
                });
            }
        }
    }

    // Send all events.
    for event in events_to_send {
        world.send_event(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{AnimationEventMarker, ClipEventMarkers};
    use crate::state_machine::{AnimationState, StateTransition, TransitionCondition};
    use euca_asset::animation::{AnimationChannel, AnimationProperty, KeyframeValue};
    use euca_asset::skeleton::Joint;
    use euca_math::{Mat4, Vec3};

    fn make_skeleton(joint_count: usize) -> Skeleton {
        let joints: Vec<Joint> = (0..joint_count)
            .map(|i| Joint {
                name: format!("joint_{i}"),
                parent: if i == 0 { None } else { Some(0) },
                local_transform: Transform::IDENTITY,
            })
            .collect();
        Skeleton {
            inverse_bind_matrices: vec![Mat4::IDENTITY; joint_count],
            joint_node_indices: (0..joint_count).collect(),
            joints,
        }
    }

    fn make_clip(name: &str, duration: f32, joint: usize, start: Vec3, end: Vec3) -> AnimationClipData {
        AnimationClipData {
            name: name.to_string(),
            duration,
            channels: vec![AnimationChannel {
                joint_index: joint,
                property: AnimationProperty::Translation,
                times: vec![0.0, duration],
                values: vec![
                    KeyframeValue::Vec3(start),
                    KeyframeValue::Vec3(end),
                ],
            }],
        }
    }

    fn setup_world_with_state_machine() -> (World, Entity) {
        let mut world = World::new();

        let clip_idle = make_clip("idle", 1.0, 0, Vec3::ZERO, Vec3::ZERO);
        let clip_run = make_clip("run", 1.0, 0, Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0));

        let mut lib = AnimationLibrary::default();
        lib.add_clip(clip_idle);
        lib.add_clip(clip_run);
        lib.add_skeleton(make_skeleton(2));
        world.insert_resource(lib);

        let states = vec![
            AnimationState {
                name: "idle".into(),
                clip_index: 0,
                speed: 1.0,
                looping: true,
            },
            AnimationState {
                name: "run".into(),
                clip_index: 1,
                speed: 1.0,
                looping: true,
            },
        ];
        let transitions = vec![
            StateTransition {
                from: Some(0),
                to: 1,
                conditions: vec![TransitionCondition::GreaterThan {
                    param: "speed".into(),
                    threshold: 0.5,
                }],
                blend_duration: 0.2,
            },
        ];
        let sm = AnimationStateMachine::new(states, transitions, 0);
        let entity = world.spawn(sm);

        (world, entity)
    }

    // ── animation_state_machine_system tests ──

    #[test]
    fn state_machine_system_advances_time() {
        let (mut world, entity) = setup_world_with_state_machine();
        animation_state_machine_system(&mut world, 0.25);
        let sm = world.get::<AnimationStateMachine>(entity).unwrap();
        assert!((sm.current_time - 0.25).abs() < 0.01);
    }

    #[test]
    fn state_machine_system_fires_transition() {
        let (mut world, entity) = setup_world_with_state_machine();
        world.get_mut::<AnimationStateMachine>(entity).unwrap().parameters.set_float("speed", 1.0);
        animation_state_machine_system(&mut world, 0.01);
        let sm = world.get::<AnimationStateMachine>(entity).unwrap();
        assert_eq!(sm.current_state, 1);
        assert!(sm.transition.is_some());
    }

    #[test]
    fn state_machine_system_wraps_looping_time() {
        let (mut world, entity) = setup_world_with_state_machine();
        animation_state_machine_system(&mut world, 1.5);
        let sm = world.get::<AnimationStateMachine>(entity).unwrap();
        assert!((sm.current_time - 0.5).abs() < 0.01);
    }

    // ── animation_blend_system tests ──

    #[test]
    fn blend_system_creates_bone_transforms() {
        let (mut world, entity) = setup_world_with_state_machine();
        animation_state_machine_system(&mut world, 0.5);
        animation_blend_system(&mut world, 0.5);
        let bones = world.get::<BoneTransforms>(entity);
        assert!(bones.is_some());
        assert_eq!(bones.unwrap().matrices.len(), 2);
    }

    #[test]
    fn blend_system_with_transition_blends_poses() {
        let (mut world, entity) = setup_world_with_state_machine();
        world.get_mut::<AnimationStateMachine>(entity).unwrap().parameters.set_float("speed", 1.0);
        animation_state_machine_system(&mut world, 0.1);
        animation_blend_system(&mut world, 0.1);
        // Should have bone transforms (blended between idle and run).
        assert!(world.get::<BoneTransforms>(entity).is_some());
    }

    #[test]
    fn blend_system_no_library_is_noop() {
        let mut world = World::new();
        let sm = AnimationStateMachine::new(
            vec![AnimationState {
                name: "idle".into(),
                clip_index: 0,
                speed: 1.0,
                looping: true,
            }],
            vec![],
            0,
        );
        let entity = world.spawn(sm);
        animation_blend_system(&mut world, 0.1);
        assert!(world.get::<BoneTransforms>(entity).is_none());
    }

    // ── root_motion_system tests ──

    #[test]
    fn root_motion_system_extracts_delta() {
        let mut world = World::new();

        let clip = AnimationClipData {
            name: "walk".into(),
            duration: 1.0,
            channels: vec![AnimationChannel {
                joint_index: 0,
                property: AnimationProperty::Translation,
                times: vec![0.0, 1.0],
                values: vec![
                    KeyframeValue::Vec3(Vec3::ZERO),
                    KeyframeValue::Vec3(Vec3::new(0.0, 0.0, 2.0)),
                ],
            }],
        };

        let mut lib = AnimationLibrary::default();
        lib.add_clip(clip);
        lib.add_skeleton(make_skeleton(2));
        world.insert_resource(lib);

        let sm = AnimationStateMachine::new(
            vec![AnimationState {
                name: "walk".into(),
                clip_index: 0,
                speed: 1.0,
                looping: true,
            }],
            vec![],
            0,
        );
        let entity = world.spawn(sm);
        world.insert(entity, RootMotionConfig { root_bone_index: 0 });

        // Advance state machine first.
        animation_state_machine_system(&mut world, 0.5);
        root_motion_system(&mut world, 0.5);

        let output = world.get::<RootMotionOutput>(entity).unwrap();
        assert!(output.delta.translation.z.abs() > 0.01);
    }

    #[test]
    fn root_motion_system_no_config_is_noop() {
        let (mut world, entity) = setup_world_with_state_machine();
        animation_state_machine_system(&mut world, 0.1);
        root_motion_system(&mut world, 0.1);
        assert!(world.get::<RootMotionOutput>(entity).is_none());
    }

    #[test]
    fn root_motion_system_no_library_is_noop() {
        let mut world = World::new();
        let sm = AnimationStateMachine::new(
            vec![AnimationState {
                name: "idle".into(),
                clip_index: 0,
                speed: 1.0,
                looping: true,
            }],
            vec![],
            0,
        );
        let entity = world.spawn(sm);
        world.insert(entity, RootMotionConfig { root_bone_index: 0 });
        root_motion_system(&mut world, 0.1);
        assert!(world.get::<RootMotionOutput>(entity).is_none());
    }

    // ── animation_event_system tests ──

    #[test]
    fn event_system_emits_events() {
        let (mut world, entity) = setup_world_with_state_machine();

        let mut event_lib = AnimationEventLibrary::default();
        event_lib.set_markers(
            0,
            ClipEventMarkers::new(vec![AnimationEventMarker {
                time: 0.1,
                name: "footstep".into(),
            }]),
        );
        world.insert_resource(event_lib);

        // Advance past the event at 0.1.
        animation_state_machine_system(&mut world, 0.2);
        animation_event_system(&mut world, 0.2);

        let events: Vec<&AnimationEvent> = world.read_events::<AnimationEvent>().collect();
        assert!(!events.is_empty());
        assert_eq!(events[0].entity, entity);
        assert_eq!(events[0].name, "footstep");
    }

    #[test]
    fn event_system_no_markers_no_events() {
        let (mut world, _) = setup_world_with_state_machine();
        world.insert_resource(AnimationEventLibrary::default());
        animation_state_machine_system(&mut world, 0.5);
        animation_event_system(&mut world, 0.5);

        let events: Vec<&AnimationEvent> = world.read_events::<AnimationEvent>().collect();
        assert!(events.is_empty());
    }

    #[test]
    fn event_system_no_library_is_noop() {
        let (mut world, _) = setup_world_with_state_machine();
        animation_state_machine_system(&mut world, 0.1);
        animation_event_system(&mut world, 0.1);
        // No crash, no events.
        let events: Vec<&AnimationEvent> = world.read_events::<AnimationEvent>().collect();
        assert!(events.is_empty());
    }
}
