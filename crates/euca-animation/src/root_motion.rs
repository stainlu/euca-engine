//! Root motion extraction — captures per-frame translation/rotation delta from the
//! root (hip) bone and applies it to the entity's world transform instead of the bone.

use euca_asset::animation::{AnimationClipData, AnimationProperty, sample_quat, sample_vec3};
use euca_math::{Quat, Vec3};

/// Per-frame root motion delta extracted from an animation clip.
#[derive(Clone, Copy, Debug)]
pub struct RootMotionDelta {
    /// World-space translation to apply to the entity this frame.
    pub translation: Vec3,
    /// World-space rotation to apply to the entity this frame.
    pub rotation: Quat,
}

impl Default for RootMotionDelta {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

/// ECS component: marks an entity for root motion extraction.
///
/// When present, the `root_motion_system` extracts motion from the specified bone
/// and writes the delta to `RootMotionOutput` instead of animating the bone.
#[derive(Clone, Debug)]
pub struct RootMotionConfig {
    /// The joint index considered the "root" bone (typically the hip).
    pub root_bone_index: usize,
}

/// ECS component: output written by the root motion system each frame.
///
/// Downstream systems (e.g., character controller) consume this delta to move the entity.
#[derive(Clone, Debug, Default)]
pub struct RootMotionOutput {
    pub delta: RootMotionDelta,
}

/// Extract the root motion delta from a clip between two time points.
///
/// This samples the root bone's translation and rotation at `prev_time` and `curr_time`,
/// then computes the delta. For looping clips that wrap around, it splits the delta
/// into two segments: [prev_time..duration] + [0..curr_time].
pub fn extract_root_motion(
    clip: &AnimationClipData,
    root_bone_index: usize,
    prev_time: f32,
    curr_time: f32,
    looping: bool,
) -> RootMotionDelta {
    let wrapped = looping && curr_time < prev_time && clip.duration > 0.0;

    if wrapped {
        // Split across loop boundary.
        let delta_to_end = extract_delta(clip, root_bone_index, prev_time, clip.duration);
        let delta_from_start = extract_delta(clip, root_bone_index, 0.0, curr_time);
        RootMotionDelta {
            translation: delta_to_end.translation + delta_from_start.translation,
            rotation: (delta_to_end.rotation * delta_from_start.rotation).normalize(),
        }
    } else {
        extract_delta(clip, root_bone_index, prev_time, curr_time)
    }
}

/// Sample the root bone at two times and compute the delta.
fn extract_delta(
    clip: &AnimationClipData,
    root_bone_index: usize,
    t0: f32,
    t1: f32,
) -> RootMotionDelta {
    let mut trans_0 = Vec3::ZERO;
    let mut trans_1 = Vec3::ZERO;
    let mut rot_0 = Quat::IDENTITY;
    let mut rot_1 = Quat::IDENTITY;

    for channel in &clip.channels {
        if channel.joint_index != root_bone_index {
            continue;
        }
        match channel.property {
            AnimationProperty::Translation => {
                trans_0 = sample_vec3(&channel.times, &channel.values, t0);
                trans_1 = sample_vec3(&channel.times, &channel.values, t1);
            }
            AnimationProperty::Rotation => {
                rot_0 = sample_quat(&channel.times, &channel.values, t0);
                rot_1 = sample_quat(&channel.times, &channel.values, t1);
            }
            AnimationProperty::Scale => {} // Scale is not part of root motion.
        }
    }

    RootMotionDelta {
        translation: trans_1 - trans_0,
        rotation: (rot_0.inverse() * rot_1).normalize(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_asset::animation::{AnimationChannel, KeyframeValue};

    fn make_root_motion_clip() -> AnimationClipData {
        AnimationClipData {
            name: "walk_forward".into(),
            duration: 1.0,
            channels: vec![
                AnimationChannel {
                    joint_index: 0, // root bone
                    property: AnimationProperty::Translation,
                    times: vec![0.0, 1.0],
                    values: vec![
                        KeyframeValue::Vec3(Vec3::ZERO),
                        KeyframeValue::Vec3(Vec3::new(0.0, 0.0, 2.0)), // moves 2 units forward
                    ],
                },
                AnimationChannel {
                    joint_index: 0,
                    property: AnimationProperty::Rotation,
                    times: vec![0.0, 1.0],
                    values: vec![
                        KeyframeValue::Quat(Quat::IDENTITY),
                        KeyframeValue::Quat(Quat::IDENTITY),
                    ],
                },
            ],
        }
    }

    #[test]
    fn linear_translation_delta() {
        let clip = make_root_motion_clip();
        let delta = extract_root_motion(&clip, 0, 0.0, 0.5, false);
        // Half of the total 2.0 movement = 1.0 in Z.
        assert!((delta.translation.z - 1.0).abs() < 0.01);
        assert!(delta.translation.x.abs() < 0.01);
    }

    #[test]
    fn full_clip_delta() {
        let clip = make_root_motion_clip();
        let delta = extract_root_motion(&clip, 0, 0.0, 1.0, false);
        assert!((delta.translation.z - 2.0).abs() < 0.01);
    }

    #[test]
    fn zero_delta_at_same_time() {
        let clip = make_root_motion_clip();
        let delta = extract_root_motion(&clip, 0, 0.5, 0.5, false);
        assert!(delta.translation.length() < 0.01);
    }

    #[test]
    fn looping_wrap_around() {
        let clip = make_root_motion_clip();
        // Wraps from 0.8 to 0.2 (crossed loop boundary).
        let delta = extract_root_motion(&clip, 0, 0.8, 0.2, true);
        // [0.8..1.0] = 0.4 units + [0.0..0.2] = 0.4 units = 0.8 total
        assert!((delta.translation.z - 0.8).abs() < 0.05);
    }

    #[test]
    fn non_root_bone_ignored() {
        let clip = AnimationClipData {
            name: "test".into(),
            duration: 1.0,
            channels: vec![AnimationChannel {
                joint_index: 5, // not root bone 0
                property: AnimationProperty::Translation,
                times: vec![0.0, 1.0],
                values: vec![
                    KeyframeValue::Vec3(Vec3::ZERO),
                    KeyframeValue::Vec3(Vec3::new(10.0, 0.0, 0.0)),
                ],
            }],
        };
        let delta = extract_root_motion(&clip, 0, 0.0, 1.0, false);
        assert!(delta.translation.length() < 0.01);
    }
}
