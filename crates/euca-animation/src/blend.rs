//! Animation blending: crossfade between poses with configurable transition durations.

use crate::clip::AnimPose;

/// A layer in the blend stack: a pose with a weight.
#[derive(Clone, Debug)]
pub struct BlendLayer {
    /// The sampled pose for this layer.
    pub pose: AnimPose,
    /// Blend weight (0.0 = inactive, 1.0 = full influence).
    pub weight: f32,
}

/// Blends multiple animation poses together using normalized weights.
///
/// Typical usage: the state machine produces a primary pose, a crossfade
/// produces a transition blend, and montages add overlay layers.
#[derive(Clone, Debug, Default)]
pub struct AnimationBlender {
    layers: Vec<BlendLayer>,
}

impl AnimationBlender {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Remove all layers.
    pub fn clear(&mut self) {
        self.layers.clear();
    }

    /// Add a layer with the given weight.
    pub fn add_layer(&mut self, pose: AnimPose, weight: f32) {
        if weight > 0.0 {
            self.layers.push(BlendLayer { pose, weight });
        }
    }

    /// Blend all layers into a single output pose.
    ///
    /// Weights are normalized so they sum to 1.0. If no layers are present,
    /// returns an identity pose with `joint_count` joints.
    pub fn evaluate(&self, joint_count: usize) -> AnimPose {
        if self.layers.is_empty() {
            return AnimPose::identity(joint_count);
        }

        if self.layers.len() == 1 {
            return self.layers[0].pose.clone();
        }

        let total_weight: f32 = self.layers.iter().map(|l| l.weight).sum();
        if total_weight <= 0.0 {
            return AnimPose::identity(joint_count);
        }

        // Start with the first layer and blend subsequent layers in
        let first_normalized = self.layers[0].weight / total_weight;
        let mut accumulated = self.layers[0].pose.clone();
        let mut accumulated_weight = first_normalized;

        for layer in &self.layers[1..] {
            let normalized = layer.weight / total_weight;
            // The blend factor is the ratio of the new layer's weight
            // relative to the total accumulated so far.
            let blend_factor = normalized / (accumulated_weight + normalized);
            accumulated = accumulated.blend(&layer.pose, blend_factor);
            accumulated_weight += normalized;
        }

        accumulated
    }
}

/// Tracks a crossfade transition between two animation states.
#[derive(Clone, Debug)]
pub struct Crossfade {
    /// How long the transition takes (seconds).
    pub duration: f32,
    /// Current progress (0.0 to 1.0).
    pub progress: f32,
}

impl Crossfade {
    /// Start a new crossfade with the given duration.
    pub fn new(duration: f32) -> Self {
        Self {
            duration: duration.max(0.001), // prevent division by zero
            progress: 0.0,
        }
    }

    /// Advance the crossfade by `dt` seconds. Returns true when complete.
    pub fn advance(&mut self, dt: f32) -> bool {
        self.progress = (self.progress + dt / self.duration).min(1.0);
        self.is_complete()
    }

    /// Returns true if the crossfade has finished.
    pub fn is_complete(&self) -> bool {
        self.progress >= 1.0
    }

    /// The blend weight for the outgoing (old) pose: 1.0 -> 0.0.
    pub fn outgoing_weight(&self) -> f32 {
        1.0 - self.progress
    }

    /// The blend weight for the incoming (new) pose: 0.0 -> 1.0.
    pub fn incoming_weight(&self) -> f32 {
        self.progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::{Transform, Vec3};

    fn pose_at(x: f32) -> AnimPose {
        AnimPose {
            joints: vec![Transform::from_translation(Vec3::new(x, 0.0, 0.0))],
        }
    }

    #[test]
    fn single_layer_passthrough() {
        let mut blender = AnimationBlender::new();
        blender.add_layer(pose_at(5.0), 1.0);
        let result = blender.evaluate(1);
        assert!((result.joints[0].translation.x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn two_layer_equal_weight() {
        let mut blender = AnimationBlender::new();
        blender.add_layer(pose_at(0.0), 1.0);
        blender.add_layer(pose_at(10.0), 1.0);
        let result = blender.evaluate(1);
        assert!((result.joints[0].translation.x - 5.0).abs() < 1e-4);
    }

    #[test]
    fn two_layer_weighted() {
        let mut blender = AnimationBlender::new();
        blender.add_layer(pose_at(0.0), 3.0);
        blender.add_layer(pose_at(10.0), 1.0);
        let result = blender.evaluate(1);
        // 0.0 * 0.75 + 10.0 * 0.25 = 2.5
        assert!((result.joints[0].translation.x - 2.5).abs() < 1e-4);
    }

    #[test]
    fn zero_weight_ignored() {
        let mut blender = AnimationBlender::new();
        blender.add_layer(pose_at(5.0), 0.0);
        blender.add_layer(pose_at(10.0), 1.0);
        let result = blender.evaluate(1);
        assert!((result.joints[0].translation.x - 10.0).abs() < 1e-5);
    }

    #[test]
    fn empty_blender_returns_identity() {
        let blender = AnimationBlender::new();
        let result = blender.evaluate(2);
        assert_eq!(result.joint_count(), 2);
        assert_eq!(result.joints[0].translation, Vec3::ZERO);
    }

    #[test]
    fn crossfade_progression() {
        let mut cf = Crossfade::new(0.5);
        assert!(!cf.is_complete());
        assert!((cf.outgoing_weight() - 1.0).abs() < 1e-5);

        cf.advance(0.25);
        assert!((cf.progress - 0.5).abs() < 1e-5);
        assert!((cf.outgoing_weight() - 0.5).abs() < 1e-5);
        assert!((cf.incoming_weight() - 0.5).abs() < 1e-5);

        assert!(cf.advance(0.25));
        assert!(cf.is_complete());
        assert!((cf.incoming_weight() - 1.0).abs() < 1e-5);
    }
}
