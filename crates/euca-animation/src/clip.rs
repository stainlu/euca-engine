//! Sampled animation poses and clip evaluation.

use euca_asset::animation::{
    AnimationChannel, AnimationClipData, AnimationProperty, sample_quat, sample_vec3,
};
use euca_asset::skeleton::Skeleton;
use euca_math::{Transform, Vec3};

/// A sampled pose: one `Transform` per joint in skeleton order.
///
/// This is the central intermediate representation -- clips produce poses,
/// blenders combine poses, and the final pose is written to bone matrices.
#[derive(Clone, Debug)]
pub struct AnimPose {
    /// Per-joint local-space transforms, indexed by joint index.
    pub joints: Vec<Transform>,
}

impl AnimPose {
    /// Create a pose with all joints set to identity.
    pub fn identity(joint_count: usize) -> Self {
        Self {
            joints: vec![Transform::IDENTITY; joint_count],
        }
    }

    /// Create a pose from a skeleton's rest (bind) pose.
    pub fn from_skeleton(skeleton: &Skeleton) -> Self {
        Self {
            joints: skeleton.joints.iter().map(|j| j.local_transform).collect(),
        }
    }

    /// Number of joints in this pose.
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// Linearly blend this pose toward `other` by weight `t` (0.0 = self, 1.0 = other).
    /// Uses lerp for translation/scale, slerp for rotation.
    pub fn blend(&self, other: &Self, t: f32) -> Self {
        let count = self.joints.len().min(other.joints.len());
        let t_clamped = t.clamp(0.0, 1.0);

        let joints = (0..count)
            .map(|i| {
                let a = &self.joints[i];
                let b = &other.joints[i];
                Transform {
                    translation: a.translation.lerp(b.translation, t_clamped),
                    rotation: a.rotation.slerp(b.rotation, t_clamped),
                    scale: a.scale.lerp(b.scale, t_clamped),
                }
            })
            .collect();

        Self { joints }
    }

    /// Copy this pose's joints into `output` without allocation.
    ///
    /// Writes `min(self.joints.len(), output.len())` joints.
    pub fn copy_into(&self, output: &mut [Transform]) {
        let count = self.joints.len().min(output.len());
        output[..count].copy_from_slice(&self.joints[..count]);
    }

    /// Additive blend: applies `additive` pose on top of this pose.
    /// `additive` is assumed to be a delta from some reference pose.
    pub fn add(&self, additive: &Self, weight: f32) -> Self {
        let count = self.joints.len().min(additive.joints.len());

        let joints = (0..count)
            .map(|i| {
                let base = &self.joints[i];
                let delta = &additive.joints[i];
                Transform {
                    translation: Vec3::new(
                        base.translation.x + delta.translation.x * weight,
                        base.translation.y + delta.translation.y * weight,
                        base.translation.z + delta.translation.z * weight,
                    ),
                    rotation: base.rotation.slerp(base.rotation * delta.rotation, weight),
                    scale: Vec3::new(
                        base.scale.x * (1.0 + (delta.scale.x - 1.0) * weight),
                        base.scale.y * (1.0 + (delta.scale.y - 1.0) * weight),
                        base.scale.z * (1.0 + (delta.scale.z - 1.0) * weight),
                    ),
                }
            })
            .collect();

        Self { joints }
    }
}

/// Sample an animation clip at a given time, writing into a pose.
/// Starts from the skeleton's rest pose and overrides only the channels present in the clip.
pub fn sample_clip(clip: &AnimationClipData, skeleton: &Skeleton, time: f32) -> AnimPose {
    let mut pose = AnimPose::from_skeleton(skeleton);
    sample_clip_into(clip, time, &mut pose);
    pose
}

/// Sample an animation clip at a given time into an existing pose.
/// Only overwrites joints/properties that have channels in the clip.
pub fn sample_clip_into(clip: &AnimationClipData, time: f32, pose: &mut AnimPose) {
    for channel in &clip.channels {
        if channel.joint_index >= pose.joints.len() {
            continue;
        }
        apply_channel(channel, time, &mut pose.joints[channel.joint_index]);
    }
}

/// Apply a single animation channel's sampled value to a joint transform.
fn apply_channel(channel: &AnimationChannel, time: f32, joint: &mut Transform) {
    match channel.property {
        AnimationProperty::Translation => {
            joint.translation = sample_vec3(&channel.times, &channel.values, time);
        }
        AnimationProperty::Rotation => {
            joint.rotation = sample_quat(&channel.times, &channel.values, time);
        }
        AnimationProperty::Scale => {
            joint.scale = sample_vec3(&channel.times, &channel.values, time);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_asset::animation::KeyframeValue;
    use euca_asset::skeleton::Joint;
    use euca_math::Mat4;

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

    #[test]
    fn pose_from_skeleton_preserves_rest() {
        let skel = test_skeleton(3);
        let pose = AnimPose::from_skeleton(&skel);
        assert_eq!(pose.joint_count(), 3);
        for j in &pose.joints {
            assert_eq!(j.translation, Vec3::ZERO);
            assert_eq!(j.scale, Vec3::ONE);
        }
    }

    #[test]
    fn blend_halfway() {
        let a = AnimPose {
            joints: vec![Transform::from_translation(Vec3::new(0.0, 0.0, 0.0))],
        };
        let b = AnimPose {
            joints: vec![Transform::from_translation(Vec3::new(10.0, 0.0, 0.0))],
        };
        let blended = a.blend(&b, 0.5);
        assert!((blended.joints[0].translation.x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn blend_clamps_weight() {
        let a = AnimPose {
            joints: vec![Transform::from_translation(Vec3::ZERO)],
        };
        let b = AnimPose {
            joints: vec![Transform::from_translation(Vec3::new(10.0, 0.0, 0.0))],
        };
        let over = a.blend(&b, 1.5);
        assert!((over.joints[0].translation.x - 10.0).abs() < 1e-5);
    }

    #[test]
    fn sample_clip_applies_translation() {
        let skel = test_skeleton(2);
        let clip = AnimationClipData {
            name: "test".into(),
            duration: 1.0,
            channels: vec![AnimationChannel {
                joint_index: 0,
                property: AnimationProperty::Translation,
                times: vec![0.0, 1.0],
                values: vec![
                    KeyframeValue::Vec3(Vec3::ZERO),
                    KeyframeValue::Vec3(Vec3::new(10.0, 0.0, 0.0)),
                ],
            }],
        };
        let pose = sample_clip(&clip, &skel, 0.5);
        assert!((pose.joints[0].translation.x - 5.0).abs() < 1e-4);
        // Joint 1 should remain at rest pose
        assert_eq!(pose.joints[1].translation, Vec3::ZERO);
    }
}
