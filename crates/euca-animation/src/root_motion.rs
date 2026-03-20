//! Root motion: extract movement from a designated root bone and apply
//! it to the entity's transform instead of the bone.
//!
//! In many animations (walk cycles, attacks), the root bone moves through
//! space. Rather than letting the skeleton drift, we capture that delta
//! and apply it to the entity's world-space transform, keeping the skeleton
//! centered.

use euca_math::{Quat, Transform, Vec3};

use crate::clip::AnimPose;

/// Configuration for root motion extraction.
///
/// Attach this as an ECS component to entities that should use root motion.
#[derive(Clone, Debug)]
pub struct RootMotionReceiver {
    /// Index of the root bone in the skeleton (typically 0 = hip/pelvis).
    pub root_bone_index: usize,
    /// Whether to extract translation from the root bone.
    pub extract_translation: bool,
    /// Whether to extract rotation from the root bone.
    pub extract_rotation: bool,
    /// Lock the vertical (Y) component -- useful for ground-based characters
    /// where vertical motion should come from physics, not animation.
    pub lock_vertical: bool,
}

impl Default for RootMotionReceiver {
    fn default() -> Self {
        Self {
            root_bone_index: 0,
            extract_translation: true,
            extract_rotation: true,
            lock_vertical: false,
        }
    }
}

/// The per-frame root motion delta extracted from animation.
///
/// The system writes this each frame; gameplay code reads it to move the entity.
#[derive(Clone, Debug, Default)]
pub struct RootMotionDelta {
    /// Translation delta in the entity's local space.
    pub translation: Vec3,
    /// Rotation delta.
    pub rotation: Quat,
}

impl RootMotionDelta {
    /// Combine with an entity's current transform to get the new transform.
    pub fn apply_to(&self, transform: &Transform) -> Transform {
        let new_rotation = (transform.rotation * self.rotation).normalize();
        let world_translation = transform.rotation * self.translation;
        Transform {
            translation: Vec3::new(
                transform.translation.x + world_translation.x,
                transform.translation.y + world_translation.y,
                transform.translation.z + world_translation.z,
            ),
            rotation: new_rotation,
            scale: transform.scale,
        }
    }
}

/// Extract root motion from two consecutive pose samples and zero out
/// the root bone's movement in the pose.
///
/// Returns the delta and modifies `current_pose` to remove root bone movement.
pub fn extract_root_motion(
    receiver: &RootMotionReceiver,
    previous_pose: &AnimPose,
    current_pose: &mut AnimPose,
) -> RootMotionDelta {
    let idx = receiver.root_bone_index;

    if idx >= current_pose.joints.len() || idx >= previous_pose.joints.len() {
        return RootMotionDelta::default();
    }

    // Copy joint data to avoid borrow conflicts when mutating the pose
    let prev_translation = previous_pose.joints[idx].translation;
    let prev_rotation = previous_pose.joints[idx].rotation;
    let curr_translation = current_pose.joints[idx].translation;
    let curr_rotation = current_pose.joints[idx].rotation;

    let mut delta_translation = Vec3::ZERO;
    let mut delta_rotation = Quat::IDENTITY;

    if receiver.extract_translation {
        delta_translation = Vec3::new(
            curr_translation.x - prev_translation.x,
            curr_translation.y - prev_translation.y,
            curr_translation.z - prev_translation.z,
        );

        if receiver.lock_vertical {
            delta_translation.y = 0.0;
        }

        // Zero out the root bone's translation in the pose (keep it centered)
        if receiver.lock_vertical {
            current_pose.joints[idx].translation = prev_translation;
        } else {
            current_pose.joints[idx].translation = Vec3::ZERO;
        }
    }

    if receiver.extract_rotation {
        // Delta rotation: prev.inverse() * curr
        delta_rotation = prev_rotation.inverse() * curr_rotation;
        // Zero out root bone rotation
        current_pose.joints[idx].rotation = prev_rotation;
    }

    RootMotionDelta {
        translation: delta_translation,
        rotation: delta_rotation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    #[test]
    fn no_motion_when_pose_unchanged() {
        let receiver = RootMotionReceiver::default();
        let prev = AnimPose {
            joints: vec![Transform::from_translation(Vec3::new(1.0, 0.0, 0.0))],
        };
        let mut curr = prev.clone();

        let delta = extract_root_motion(&receiver, &prev, &mut curr);
        assert!((delta.translation.x).abs() < 1e-5);
        assert!((delta.translation.y).abs() < 1e-5);
        assert!((delta.translation.z).abs() < 1e-5);
    }

    #[test]
    fn extracts_translation_delta() {
        let receiver = RootMotionReceiver::default();
        let prev = AnimPose {
            joints: vec![Transform::from_translation(Vec3::new(0.0, 0.0, 0.0))],
        };
        let mut curr = AnimPose {
            joints: vec![Transform::from_translation(Vec3::new(1.0, 0.0, 0.5))],
        };

        let delta = extract_root_motion(&receiver, &prev, &mut curr);
        assert!((delta.translation.x - 1.0).abs() < 1e-5);
        assert!((delta.translation.z - 0.5).abs() < 1e-5);
    }

    #[test]
    fn locks_vertical() {
        let receiver = RootMotionReceiver {
            lock_vertical: true,
            ..Default::default()
        };
        let prev = AnimPose {
            joints: vec![Transform::from_translation(Vec3::ZERO)],
        };
        let mut curr = AnimPose {
            joints: vec![Transform::from_translation(Vec3::new(1.0, 5.0, 0.0))],
        };

        let delta = extract_root_motion(&receiver, &prev, &mut curr);
        assert!((delta.translation.x - 1.0).abs() < 1e-5);
        assert!((delta.translation.y).abs() < 1e-5);
    }

    #[test]
    fn zeros_root_bone_in_pose() {
        let receiver = RootMotionReceiver {
            extract_rotation: false,
            ..Default::default()
        };
        let prev = AnimPose {
            joints: vec![Transform::from_translation(Vec3::ZERO)],
        };
        let mut curr = AnimPose {
            joints: vec![Transform::from_translation(Vec3::new(5.0, 0.0, 0.0))],
        };

        let _delta = extract_root_motion(&receiver, &prev, &mut curr);
        assert!((curr.joints[0].translation.x).abs() < 1e-5);
    }

    #[test]
    fn apply_delta_to_transform() {
        let delta = RootMotionDelta {
            translation: Vec3::new(1.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
        };
        let transform = Transform::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let result = delta.apply_to(&transform);
        assert!((result.translation.x - 11.0).abs() < 1e-5);
    }

    #[test]
    fn out_of_bounds_bone_returns_zero_delta() {
        let receiver = RootMotionReceiver {
            root_bone_index: 99,
            ..Default::default()
        };
        let prev = AnimPose {
            joints: vec![Transform::IDENTITY],
        };
        let mut curr = AnimPose {
            joints: vec![Transform::IDENTITY],
        };

        let delta = extract_root_motion(&receiver, &prev, &mut curr);
        assert!((delta.translation.x).abs() < 1e-5);
    }
}
