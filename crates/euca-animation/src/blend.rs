//! Animation blending: crossfade between poses with configurable transition durations.

use crate::clip::AnimPose;
use euca_math::Transform;

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
    ///
    /// This allocates a temporary workspace internally. For hot-path usage,
    /// prefer [`evaluate_into`](Self::evaluate_into) with a reusable [`BlendWorkspace`].
    pub fn evaluate(&self, joint_count: usize) -> AnimPose {
        let mut workspace = BlendWorkspace::new(joint_count);
        self.evaluate_into(joint_count, &mut workspace);
        AnimPose {
            joints: workspace.buffer,
        }
    }

    /// Blend all layers into a pre-allocated workspace, avoiding per-frame allocation.
    ///
    /// The workspace buffer is resized to `joint_count` if needed (this only
    /// allocates when the workspace is too small — a one-time cost). Results
    /// are written in-place into `workspace.buffer`.
    pub fn evaluate_into(&self, joint_count: usize, workspace: &mut BlendWorkspace) {
        workspace.ensure_capacity(joint_count);
        let output = &mut workspace.buffer[..joint_count];

        if self.layers.is_empty() {
            output.fill(Transform::IDENTITY);
            return;
        }

        if self.layers.len() == 1 {
            self.layers[0].pose.copy_into(output);
            return;
        }

        let total_weight: f32 = self.layers.iter().map(|l| l.weight).sum();
        if total_weight <= 0.0 {
            output.fill(Transform::IDENTITY);
            return;
        }

        // Copy first layer into output, then blend subsequent layers in-place.
        self.layers[0].pose.copy_into(output);
        let first_normalized = self.layers[0].weight / total_weight;
        let mut accumulated_weight = first_normalized;

        for layer in &self.layers[1..] {
            let normalized = layer.weight / total_weight;
            let blend_factor = normalized / (accumulated_weight + normalized);

            let t = blend_factor.clamp(0.0, 1.0);
            for (out, b) in output.iter_mut().zip(&layer.pose.joints) {
                let a_translation = out.translation;
                let a_rotation = out.rotation;
                let a_scale = out.scale;
                *out = Transform {
                    translation: a_translation.lerp(b.translation, t),
                    rotation: a_rotation.slerp(b.rotation, t),
                    scale: a_scale.lerp(b.scale, t),
                };
            }

            accumulated_weight += normalized;
        }
    }
}

/// Pre-allocated workspace for animation blend evaluation.
///
/// Created once per animated entity and reused every frame, eliminating
/// per-frame `Vec<Transform>` allocations from pose cloning and blending.
///
/// This follows the "prepare/evaluate split" pattern: the workspace is
/// prepared once (allocated to match the skeleton's joint count) and then
/// reused in every evaluate call.
#[derive(Clone, Debug)]
pub struct BlendWorkspace {
    /// The reusable joint transform buffer. After [`AnimationBlender::evaluate_into`],
    /// the first `joint_count` elements contain the blended result.
    pub buffer: Vec<Transform>,
}

impl BlendWorkspace {
    /// Create a workspace pre-allocated for the given number of joints.
    pub fn new(joint_count: usize) -> Self {
        Self {
            buffer: vec![Transform::IDENTITY; joint_count],
        }
    }

    /// Ensure the buffer can hold at least `joint_count` joints.
    ///
    /// If the buffer is already large enough, this is a no-op.
    /// If not, it extends with identity transforms (one-time cost).
    fn ensure_capacity(&mut self, joint_count: usize) {
        if self.buffer.len() < joint_count {
            self.buffer.resize(joint_count, Transform::IDENTITY);
        }
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

    // --- BlendWorkspace tests ---

    #[test]
    fn evaluate_into_matches_evaluate() {
        let mut blender = AnimationBlender::new();
        blender.add_layer(pose_at(0.0), 3.0);
        blender.add_layer(pose_at(10.0), 1.0);

        let allocated = blender.evaluate(1);
        let mut workspace = BlendWorkspace::new(1);
        blender.evaluate_into(1, &mut workspace);

        assert!(
            (workspace.buffer[0].translation.x - allocated.joints[0].translation.x).abs() < 1e-6
        );
    }

    #[test]
    fn workspace_reuse_no_realloc() {
        let mut workspace = BlendWorkspace::new(4);
        let ptr_before = workspace.buffer.as_ptr();

        let mut blender = AnimationBlender::new();
        blender.add_layer(pose_at(1.0), 1.0);
        blender.evaluate_into(4, &mut workspace);

        // Buffer pointer unchanged — no reallocation occurred.
        assert_eq!(workspace.buffer.as_ptr(), ptr_before);
    }

    #[test]
    fn workspace_grows_if_needed() {
        let mut workspace = BlendWorkspace::new(1);
        assert_eq!(workspace.buffer.len(), 1);

        let mut blender = AnimationBlender::new();
        blender.add_layer(
            AnimPose {
                joints: vec![
                    Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
                    Transform::from_translation(Vec3::new(2.0, 0.0, 0.0)),
                    Transform::from_translation(Vec3::new(3.0, 0.0, 0.0)),
                ],
            },
            1.0,
        );
        blender.evaluate_into(3, &mut workspace);

        assert!(workspace.buffer.len() >= 3);
        assert!((workspace.buffer[2].translation.x - 3.0).abs() < 1e-5);
    }

    #[test]
    fn evaluate_into_empty_writes_identity() {
        let blender = AnimationBlender::new();
        let mut workspace = BlendWorkspace::new(2);
        // Dirty the buffer first
        workspace.buffer[0] = Transform::from_translation(Vec3::new(99.0, 0.0, 0.0));

        blender.evaluate_into(2, &mut workspace);
        assert_eq!(workspace.buffer[0].translation, Vec3::ZERO);
        assert_eq!(workspace.buffer[1].translation, Vec3::ZERO);
    }

    #[test]
    fn evaluate_into_single_layer() {
        let mut blender = AnimationBlender::new();
        blender.add_layer(pose_at(7.0), 1.0);

        let mut workspace = BlendWorkspace::new(1);
        blender.evaluate_into(1, &mut workspace);
        assert!((workspace.buffer[0].translation.x - 7.0).abs() < 1e-5);
    }
}
