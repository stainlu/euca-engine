//! Animation clip data extracted from glTF files.

use euca_math::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Which transform property a channel targets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnimationProperty {
    Translation,
    Rotation,
    Scale,
}

/// A single keyframe value (time → value).
#[derive(Clone, Debug)]
pub enum KeyframeValue {
    Vec3(Vec3),
    Quat(Quat),
}

/// One animation channel: targets a specific joint's property.
#[derive(Clone, Debug)]
pub struct AnimationChannel {
    /// Index of the target joint in the skeleton.
    pub joint_index: usize,
    /// Which property this channel animates.
    pub property: AnimationProperty,
    /// Keyframe timestamps (seconds).
    pub times: Vec<f32>,
    /// Keyframe values (same length as times).
    pub values: Vec<KeyframeValue>,
}

/// A complete animation clip with multiple channels.
#[derive(Clone, Debug)]
pub struct AnimationClipData {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
}

/// Sample a Vec3 channel at a given time using linear interpolation.
pub fn sample_vec3(times: &[f32], values: &[KeyframeValue], t: f32) -> Vec3 {
    if times.is_empty() {
        return Vec3::ZERO;
    }
    if t <= times[0] {
        return match &values[0] {
            KeyframeValue::Vec3(v) => *v,
            _ => Vec3::ZERO,
        };
    }
    // SAFETY: empty case handled above, so last() is guaranteed Some.
    if t >= *times.last().expect("non-empty times") {
        return match values.last().expect("non-empty values") {
            KeyframeValue::Vec3(v) => *v,
            _ => Vec3::ZERO,
        };
    }
    // Find the two keyframes to interpolate between
    for i in 0..times.len() - 1 {
        if t >= times[i] && t < times[i + 1] {
            let frac = (t - times[i]) / (times[i + 1] - times[i]);
            if let (KeyframeValue::Vec3(a), KeyframeValue::Vec3(b)) = (&values[i], &values[i + 1]) {
                return Vec3::new(
                    a.x + (b.x - a.x) * frac,
                    a.y + (b.y - a.y) * frac,
                    a.z + (b.z - a.z) * frac,
                );
            }
        }
    }
    Vec3::ZERO
}

/// Sample a Quat channel at a given time using SLERP.
pub fn sample_quat(times: &[f32], values: &[KeyframeValue], t: f32) -> Quat {
    if times.is_empty() {
        return Quat::IDENTITY;
    }
    if t <= times[0] {
        return match &values[0] {
            KeyframeValue::Quat(q) => *q,
            _ => Quat::IDENTITY,
        };
    }
    // SAFETY: empty case handled above, so last() is guaranteed Some.
    if t >= *times.last().expect("non-empty times") {
        return match values.last().expect("non-empty values") {
            KeyframeValue::Quat(q) => *q,
            _ => Quat::IDENTITY,
        };
    }
    for i in 0..times.len() - 1 {
        if t >= times[i] && t < times[i + 1] {
            let frac = (t - times[i]) / (times[i + 1] - times[i]);
            if let (KeyframeValue::Quat(a), KeyframeValue::Quat(b)) = (&values[i], &values[i + 1]) {
                return a.slerp(*b, frac);
            }
        }
    }
    Quat::IDENTITY
}

/// Parse animations from a glTF document.
pub fn parse_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    joint_node_indices: &[usize],
) -> Vec<AnimationClipData> {
    let mut clips = Vec::new();

    // Map node index → joint index for quick lookup
    let node_to_joint: std::collections::HashMap<usize, usize> = joint_node_indices
        .iter()
        .enumerate()
        .map(|(ji, &ni)| (ni, ji))
        .collect();

    for animation in document.animations() {
        let name = animation.name().unwrap_or("unnamed").to_string();
        let mut channels = Vec::new();
        let mut duration: f32 = 0.0;

        for channel in animation.channels() {
            let target_node = channel.target().node().index();
            let joint_index = match node_to_joint.get(&target_node) {
                Some(&ji) => ji,
                None => continue, // Not a joint we care about
            };

            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

            let times: Vec<f32> = match reader.read_inputs() {
                Some(iter) => iter.collect(),
                None => continue,
            };

            if let Some(&max_t) = times.last() {
                duration = duration.max(max_t);
            }

            let property;
            let values: Vec<KeyframeValue>;

            match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(iter)) => {
                    property = AnimationProperty::Translation;
                    values = iter
                        .map(|t| KeyframeValue::Vec3(Vec3::new(t[0], t[1], t[2])))
                        .collect();
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(iter)) => {
                    property = AnimationProperty::Rotation;
                    values = iter
                        .into_f32()
                        .map(|r| KeyframeValue::Quat(Quat::from_xyzw(r[0], r[1], r[2], r[3])))
                        .collect();
                }
                Some(gltf::animation::util::ReadOutputs::Scales(iter)) => {
                    property = AnimationProperty::Scale;
                    values = iter
                        .map(|s| KeyframeValue::Vec3(Vec3::new(s[0], s[1], s[2])))
                        .collect();
                }
                _ => continue,
            }

            if times.len() == values.len() {
                channels.push(AnimationChannel {
                    joint_index,
                    property,
                    times,
                    values,
                });
            }
        }

        if !channels.is_empty() {
            log::info!(
                "Parsed animation '{}': {} channels, {:.2}s duration",
                name,
                channels.len(),
                duration,
            );
            clips.push(AnimationClipData {
                name,
                duration,
                channels,
            });
        }
    }

    clips
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_vec3_interpolation() {
        let times = vec![0.0, 1.0];
        let values = vec![
            KeyframeValue::Vec3(Vec3::ZERO),
            KeyframeValue::Vec3(Vec3::new(10.0, 0.0, 0.0)),
        ];
        let v = sample_vec3(&times, &values, 0.5);
        assert!((v.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn sample_vec3_before_first() {
        let times = vec![1.0, 2.0];
        let values = vec![
            KeyframeValue::Vec3(Vec3::new(5.0, 0.0, 0.0)),
            KeyframeValue::Vec3(Vec3::new(10.0, 0.0, 0.0)),
        ];
        let v = sample_vec3(&times, &values, 0.0);
        assert!((v.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn sample_quat_identity() {
        let times = vec![0.0, 1.0];
        let values = vec![
            KeyframeValue::Quat(Quat::IDENTITY),
            KeyframeValue::Quat(Quat::IDENTITY),
        ];
        let q = sample_quat(&times, &values, 0.5);
        assert!((q.w - 1.0).abs() < 0.01);
    }
}
