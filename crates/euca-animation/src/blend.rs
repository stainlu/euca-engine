//! Animation blending — crossfade between clips using per-bone lerp/slerp.

use euca_asset::animation::{
    AnimationClipData, AnimationProperty, sample_quat, sample_vec3,
};
use euca_asset::skeleton::Skeleton;
use euca_math::Transform;

/// A single animation layer: a clip sampled at a specific time, with a blend weight.
#[derive(Clone, Debug)]
pub struct BlendLayer {
    /// Index into the animation library's clip list.
    pub clip_index: usize,
    /// Current playback time within the clip (seconds).
    pub time: f32,
    /// Blend weight in [0.0, 1.0]. Weights across layers are normalized before blending.
    pub weight: f32,
}

/// Blends multiple animation layers into a single pose using per-bone lerp (translation/scale)
/// and slerp (rotation).
///
/// Layers with zero weight are skipped. If all weights are zero, the skeleton rest pose is
/// returned.
pub fn blend_poses(
    layers: &[BlendLayer],
    clips: &[AnimationClipData],
    skeleton: &Skeleton,
) -> Vec<Transform> {
    let joint_count = skeleton.joints.len();
    let rest_poses: Vec<Transform> = skeleton.joints.iter().map(|j| j.local_transform).collect();

    // Collect layers with positive weight and valid clip indices.
    let active: Vec<(&BlendLayer, &AnimationClipData)> = layers
        .iter()
        .filter(|l| l.weight > 0.0)
        .filter_map(|l| clips.get(l.clip_index).map(|c| (l, c)))
        .collect();

    if active.is_empty() {
        return rest_poses;
    }

    // Normalize weights so they sum to 1.0.
    let total_weight: f32 = active.iter().map(|(l, _)| l.weight).sum();
    if total_weight <= 0.0 {
        return rest_poses;
    }

    // Sample each active layer into a full-skeleton pose.
    let sampled: Vec<(f32, Vec<Transform>)> = active
        .iter()
        .map(|(layer, clip)| {
            let normalized_weight = layer.weight / total_weight;
            let mut pose = rest_poses.clone();
            for channel in &clip.channels {
                if channel.joint_index >= joint_count {
                    continue;
                }
                match channel.property {
                    AnimationProperty::Translation => {
                        pose[channel.joint_index].translation =
                            sample_vec3(&channel.times, &channel.values, layer.time);
                    }
                    AnimationProperty::Rotation => {
                        pose[channel.joint_index].rotation =
                            sample_quat(&channel.times, &channel.values, layer.time);
                    }
                    AnimationProperty::Scale => {
                        pose[channel.joint_index].scale =
                            sample_vec3(&channel.times, &channel.values, layer.time);
                    }
                }
            }
            (normalized_weight, pose)
        })
        .collect();

    // Blend all sampled poses together using weighted lerp/slerp.
    // Start from the first layer and accumulate.
    let (first_weight, first_pose) = &sampled[0];
    let mut result: Vec<Transform> = first_pose
        .iter()
        .map(|t| Transform {
            translation: t.translation * *first_weight,
            rotation: t.rotation, // rotation handled separately
            scale: t.scale * *first_weight,
        })
        .collect();

    // For rotation, we use iterative slerp: accumulate into the first rotation,
    // tracking cumulative weight.
    let mut accumulated_rot_weight = *first_weight;

    for (weight, pose) in &sampled[1..] {
        for (j, transform) in pose.iter().enumerate() {
            result[j].translation = result[j].translation + transform.translation * *weight;
            result[j].scale = result[j].scale + transform.scale * *weight;

            // Incremental slerp: blend accumulated result toward this layer's rotation
            // using t = new_weight / (accumulated + new_weight), which produces the
            // correctly weighted average.
            let t = *weight / (accumulated_rot_weight + *weight);
            result[j].rotation = result[j].rotation.slerp(transform.rotation, t);
        }
        accumulated_rot_weight += *weight;
    }

    result
}

/// Crossfade helper: produces two blend layers for a smooth transition between clips.
///
/// `progress` is in [0.0, 1.0] where 0.0 = fully `from_clip` and 1.0 = fully `to_clip`.
pub fn crossfade_layers(
    from_clip: usize,
    from_time: f32,
    to_clip: usize,
    to_time: f32,
    progress: f32,
) -> [BlendLayer; 2] {
    let progress = progress.clamp(0.0, 1.0);
    [
        BlendLayer {
            clip_index: from_clip,
            time: from_time,
            weight: 1.0 - progress,
        },
        BlendLayer {
            clip_index: to_clip,
            time: to_time,
            weight: progress,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_asset::animation::{AnimationChannel, KeyframeValue};
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

    #[test]
    fn single_layer_full_weight_equals_clip_sample() {
        let skeleton = make_skeleton(2);
        let clip = make_clip("walk", 1.0, 0, Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        let clips = vec![clip];

        let layers = vec![BlendLayer {
            clip_index: 0,
            time: 0.5,
            weight: 1.0,
        }];
        let result = blend_poses(&layers, &clips, &skeleton);
        assert!((result[0].translation.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn equal_weight_blend_produces_midpoint() {
        let skeleton = make_skeleton(1);
        let clip_a = make_clip("a", 1.0, 0, Vec3::ZERO, Vec3::ZERO); // stays at zero
        let clip_b = make_clip("b", 1.0, 0, Vec3::new(10.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0));
        let clips = vec![clip_a, clip_b];

        let layers = vec![
            BlendLayer { clip_index: 0, time: 0.0, weight: 1.0 },
            BlendLayer { clip_index: 1, time: 0.0, weight: 1.0 },
        ];
        let result = blend_poses(&layers, &clips, &skeleton);
        // Equal weight: (0 + 10) / 2 = 5
        assert!((result[0].translation.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn zero_weight_layers_ignored() {
        let skeleton = make_skeleton(1);
        let clip_a = make_clip("a", 1.0, 0, Vec3::new(5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0));
        let clip_b = make_clip("b", 1.0, 0, Vec3::new(99.0, 0.0, 0.0), Vec3::new(99.0, 0.0, 0.0));
        let clips = vec![clip_a, clip_b];

        let layers = vec![
            BlendLayer { clip_index: 0, time: 0.0, weight: 1.0 },
            BlendLayer { clip_index: 1, time: 0.0, weight: 0.0 },
        ];
        let result = blend_poses(&layers, &clips, &skeleton);
        assert!((result[0].translation.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn empty_layers_returns_rest_pose() {
        let skeleton = make_skeleton(1);
        let result = blend_poses(&[], &[], &skeleton);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], Transform::IDENTITY);
    }

    #[test]
    fn crossfade_layers_at_zero_is_fully_from() {
        let layers = crossfade_layers(0, 0.5, 1, 0.0, 0.0);
        assert!((layers[0].weight - 1.0).abs() < 0.001);
        assert!((layers[1].weight - 0.0).abs() < 0.001);
    }

    #[test]
    fn crossfade_layers_at_one_is_fully_to() {
        let layers = crossfade_layers(0, 0.5, 1, 0.0, 1.0);
        assert!((layers[0].weight - 0.0).abs() < 0.001);
        assert!((layers[1].weight - 1.0).abs() < 0.001);
    }
}
