//! Animation systems — sample keyframes, compute bone matrices.

use euca_ecs::{Entity, Query, World};
use euca_math::Transform;

use crate::animation::{AnimationClipData, AnimationProperty, sample_quat, sample_vec3};
use crate::skeleton::Skeleton;

/// ECS component: references a clip and tracks playback state.
#[derive(Clone, Debug)]
pub struct SkeletalAnimator {
    /// Index into the AnimationLibrary resource.
    pub clip_index: usize,
    /// Current playback time (seconds).
    pub time: f32,
    /// Playback speed multiplier.
    pub speed: f32,
    /// Whether to loop.
    pub looping: bool,
    /// Whether currently playing.
    pub playing: bool,
}

impl SkeletalAnimator {
    pub fn new(clip_index: usize) -> Self {
        Self {
            clip_index,
            time: 0.0,
            speed: 1.0,
            looping: true,
            playing: true,
        }
    }
}

/// ECS component: stores computed bone matrices for GPU upload.
#[derive(Clone, Debug)]
pub struct BoneTransforms {
    /// One Mat4 per joint — final skinning matrices.
    pub matrices: Vec<euca_math::Mat4>,
}

/// World resource: stores all loaded animation clips.
#[derive(Clone, Debug, Default)]
pub struct AnimationLibrary {
    pub clips: Vec<AnimationClipData>,
    pub skeletons: Vec<Skeleton>,
}

impl AnimationLibrary {
    pub fn add_clip(&mut self, clip: AnimationClipData) -> usize {
        let idx = self.clips.len();
        self.clips.push(clip);
        idx
    }

    pub fn add_skeleton(&mut self, skeleton: Skeleton) -> usize {
        let idx = self.skeletons.len();
        self.skeletons.push(skeleton);
        idx
    }
}

/// Each tick: advance animation time, sample keyframes, compute bone matrices.
pub fn skeletal_animation_system(world: &mut World, dt: f32) {
    // Collect entities with animators
    let entities: Vec<(Entity, usize, f32, f32, bool, bool)> = {
        let query = Query::<(Entity, &SkeletalAnimator)>::new(world);
        query
            .iter()
            .map(|(e, anim)| {
                (
                    e,
                    anim.clip_index,
                    anim.time,
                    anim.speed,
                    anim.looping,
                    anim.playing,
                )
            })
            .collect()
    };

    // Get library (immutable borrow for clips/skeletons)
    let library_data: Option<(Vec<AnimationClipData>, Vec<Skeleton>)> = world
        .resource::<AnimationLibrary>()
        .map(|lib| (lib.clips.clone(), lib.skeletons.clone()));

    let (clips, skeletons) = match library_data {
        Some(d) => d,
        None => return,
    };

    for (entity, clip_index, time, speed, looping, playing) in entities {
        if !playing {
            continue;
        }

        let clip = match clips.get(clip_index) {
            Some(c) => c,
            None => continue,
        };

        // We use skeleton 0 for now (single skeleton per scene)
        let skeleton = match skeletons.first() {
            Some(s) => s,
            None => continue,
        };

        // Advance time
        let mut new_time = time + dt * speed;
        if clip.duration > 0.0 {
            if looping {
                new_time %= clip.duration;
            } else {
                new_time = new_time.min(clip.duration);
            }
        }

        // Update animator time
        if let Some(anim) = world.get_mut::<SkeletalAnimator>(entity) {
            anim.time = new_time;
        }

        // Sample keyframes into local poses
        let joint_count = skeleton.joints.len();
        let mut local_poses: Vec<Transform> =
            skeleton.joints.iter().map(|j| j.local_transform).collect();

        for channel in &clip.channels {
            if channel.joint_index >= joint_count {
                continue;
            }
            match channel.property {
                AnimationProperty::Translation => {
                    local_poses[channel.joint_index].translation =
                        sample_vec3(&channel.times, &channel.values, new_time);
                }
                AnimationProperty::Rotation => {
                    local_poses[channel.joint_index].rotation =
                        sample_quat(&channel.times, &channel.values, new_time);
                }
                AnimationProperty::Scale => {
                    local_poses[channel.joint_index].scale =
                        sample_vec3(&channel.times, &channel.values, new_time);
                }
            }
        }

        // Compute final joint matrices
        let matrices = skeleton.compute_joint_matrices(&local_poses);

        // Write to BoneTransforms component
        if let Some(bones) = world.get_mut::<BoneTransforms>(entity) {
            bones.matrices = matrices;
        } else {
            world.insert(entity, BoneTransforms { matrices });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animator_defaults() {
        let anim = SkeletalAnimator::new(0);
        assert_eq!(anim.clip_index, 0);
        assert_eq!(anim.time, 0.0);
        assert!(anim.looping);
        assert!(anim.playing);
    }

    #[test]
    fn animation_library_add() {
        let mut lib = AnimationLibrary::default();
        let idx = lib.add_clip(crate::animation::AnimationClipData {
            name: "walk".into(),
            duration: 1.0,
            channels: vec![],
        });
        assert_eq!(idx, 0);
        assert_eq!(lib.clips.len(), 1);
    }
}
