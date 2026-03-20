//! Inverse kinematics solvers and ECS integration.
//!
//! Provides two solvers:
//! - [`two_bone_ik`]: analytical solver for two-bone chains (e.g. arm, leg)
//! - [`fabrik_solve`]: iterative FABRIK solver for arbitrary chain lengths
//!
//! Plus ECS components:
//! - [`IkChain`]: defines a bone chain, target, pole vector, and blend weight
//! - [`LookAtConstraint`]: rotates a single bone to face a target
//!
//! # System
//!
//! [`ik_solve_system`] iterates all entities with [`IkChain`] or
//! [`LookAtConstraint`] components and applies the solved rotations
//! into the entity's animation pose (via [`Animator`]).

use euca_asset::skeleton::Skeleton;
use euca_asset::systems::AnimationLibrary;
use euca_ecs::{Entity, Query, World};
use euca_math::{Quat, Transform, Vec3};

use crate::clip::AnimPose;
use crate::system::Animator;

// ── Quaternion helper ────────────────────────────────────────────────────────

/// Compute the shortest-arc rotation quaternion that rotates direction `from`
/// to direction `to`. Both inputs must be unit-length.
fn rotation_arc(from: Vec3, to: Vec3) -> Quat {
    let dot = from.dot(to).clamp(-1.0, 1.0);

    // Nearly identical directions.
    if dot > 0.99999 {
        return Quat::IDENTITY;
    }

    // Nearly opposite directions -- pick an arbitrary perpendicular axis.
    if dot < -0.99999 {
        let perp = arbitrary_perpendicular(from);
        return Quat::from_axis_angle(perp, std::f32::consts::PI);
    }

    let axis = from.cross(to);
    // Quat = (axis * sin(half_angle), cos(half_angle))
    // Using the identity: q = normalize(cross, 1 + dot)
    let q = Quat::from_xyzw(axis.x, axis.y, axis.z, 1.0 + dot);
    q.normalize()
}

/// Find an arbitrary unit vector perpendicular to `v` (assumed normalized).
fn arbitrary_perpendicular(v: Vec3) -> Vec3 {
    let candidate = if v.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    v.cross(candidate).normalize()
}

// ── Free-function solvers ────────────────────────────────────────────────────

/// Analytical two-bone IK solver.
///
/// Given three joint positions (root, mid, tip) forming a two-bone chain,
/// plus a desired `target` position and a `pole_target` that controls the
/// bend plane direction, returns the world-space rotations `(root_rot, mid_rot)`
/// that should be applied to the root and mid joints so the tip reaches the
/// target.
///
/// If the target is unreachable (beyond the chain's total length), the chain
/// extends as far as possible toward the target.
pub fn two_bone_ik(
    root_pos: Vec3,
    mid_pos: Vec3,
    tip_pos: Vec3,
    target: Vec3,
    pole_target: Vec3,
) -> (Quat, Quat) {
    let upper_len = (mid_pos - root_pos).length();
    let lower_len = (tip_pos - mid_pos).length();
    let chain_len = upper_len + lower_len;

    let to_target = target - root_pos;
    let target_dist = to_target.length().max(1e-6);

    // Clamp target distance to the reachable range.
    let clamped_dist = target_dist
        .min(chain_len - 1e-4)
        .max((upper_len - lower_len).abs() + 1e-4);

    // Law of cosines: angle at the root joint.
    let cos_root = ((upper_len * upper_len + clamped_dist * clamped_dist - lower_len * lower_len)
        / (2.0 * upper_len * clamped_dist))
        .clamp(-1.0, 1.0);
    let root_angle = cos_root.acos();

    // Build a coordinate frame for the chain plane.
    let target_dir = to_target * (1.0 / target_dist);

    // The pole target determines which way the mid joint bends.
    // Project onto the plane perpendicular to target_dir.
    let pole_on_target = target_dir * pole_target.dot(target_dir);
    let pole_perp = pole_target - pole_on_target;
    let pole_len = pole_perp.length();
    let bend_dir = if pole_len > 1e-6 {
        pole_perp * (1.0 / pole_len)
    } else {
        // No valid pole direction -- fall back to the original bend.
        let original_bend = mid_pos - root_pos;
        let on_target = target_dir * original_bend.dot(target_dir);
        let perp = original_bend - on_target;
        let perp_len = perp.length();
        if perp_len > 1e-6 {
            perp * (1.0 / perp_len)
        } else {
            arbitrary_perpendicular(target_dir)
        }
    };

    // Compute the new mid position.
    let new_mid = root_pos
        + target_dir * (root_angle.cos() * upper_len)
        + bend_dir * (root_angle.sin() * upper_len);

    // Root rotation: rotate the old upper-bone direction to the new one.
    let old_upper_dir = (mid_pos - root_pos).normalize();
    let new_upper_dir = (new_mid - root_pos).normalize();
    let root_rot = rotation_arc(old_upper_dir, new_upper_dir);

    // Mid rotation: rotate the old lower-bone direction to the new one.
    let old_lower_dir = (tip_pos - mid_pos).normalize();
    let new_tip = root_pos + target_dir * clamped_dist;
    let new_lower_dir = (new_tip - new_mid).normalize();
    // The old lower direction must first be transformed by the root rotation.
    let old_lower_rotated = root_rot * old_lower_dir;
    let mid_rot = rotation_arc(old_lower_rotated, new_lower_dir);

    (root_rot, mid_rot)
}

/// FABRIK (Forward And Backward Reaching Inverse Kinematics) solver.
///
/// Iteratively adjusts the positions in `chain` (root-to-tip, length = bones + 1)
/// so the tip reaches `target`. Segment lengths are preserved.
///
/// Returns `true` if the solver converged within `tolerance`, `false` otherwise.
pub fn fabrik_solve(chain: &mut [Vec3], target: Vec3, tolerance: f32, max_iters: u32) -> bool {
    let n = chain.len();
    if n < 2 {
        return true;
    }

    // Pre-compute segment lengths.
    let lengths: Vec<f32> = (0..n - 1)
        .map(|i| (chain[i + 1] - chain[i]).length())
        .collect();
    let total_len: f32 = lengths.iter().sum();
    let root = chain[0];

    // If the target is unreachable, extend the chain straight toward it.
    let root_to_target = (target - root).length();
    if root_to_target > total_len {
        let dir = (target - root).normalize();
        let mut accumulated = 0.0_f32;
        for i in 1..n {
            accumulated += lengths[i - 1];
            chain[i] = root + dir * accumulated;
        }
        return false;
    }

    for _iteration in 0..max_iters {
        // Check convergence.
        let tip_dist = (chain[n - 1] - target).length();
        if tip_dist < tolerance {
            return true;
        }

        // Forward reaching: move tip to target, propagate backward.
        chain[n - 1] = target;
        for i in (0..n - 1).rev() {
            let dir = (chain[i] - chain[i + 1]).normalize();
            chain[i] = chain[i + 1] + dir * lengths[i];
        }

        // Backward reaching: pin root, propagate forward.
        chain[0] = root;
        for i in 0..n - 1 {
            let dir = (chain[i + 1] - chain[i]).normalize();
            chain[i + 1] = chain[i] + dir * lengths[i];
        }
    }

    // Final convergence check after all iterations.
    (chain[n - 1] - target).length() < tolerance
}

// ── ECS Components ───────────────────────────────────────────────────────────

/// ECS component: defines an IK bone chain with a world-space target.
///
/// The `bone_indices` list the joints from root to tip of the chain
/// (e.g. `[shoulder, elbow, wrist]` for an arm). The solver is chosen
/// automatically: two-bone analytical for exactly 3 joints, FABRIK otherwise.
#[derive(Clone, Debug)]
pub struct IkChain {
    /// Joint indices into the skeleton, ordered root-to-tip.
    pub bone_indices: Vec<usize>,
    /// World-space target position the tip should reach.
    pub target: Vec3,
    /// Optional pole target controlling the bend-plane direction (e.g. knee
    /// or elbow direction). Only used by the two-bone solver.
    pub pole: Option<Vec3>,
    /// Blend weight: 0.0 = pure animation, 1.0 = pure IK.
    pub weight: f32,
}

/// ECS component: rotates a single bone to face a target direction.
///
/// Commonly used for head tracking, turret aiming, or eye look-at.
#[derive(Clone, Debug)]
pub struct LookAtConstraint {
    /// Index of the bone to rotate.
    pub bone_index: usize,
    /// World-space position to look at.
    pub target: Vec3,
    /// The up-vector used to construct the look-at orientation.
    pub up: Vec3,
}

// ── IK Solve System ──────────────────────────────────────────────────────────

/// ECS system: processes all [`IkChain`] and [`LookAtConstraint`] components.
///
/// For each entity that has an [`Animator`] and an [`IkChain`], this system:
/// 1. Retrieves the current animation pose
/// 2. Computes world-space joint positions
/// 3. Runs the appropriate IK solver (two-bone or FABRIK)
/// 4. Converts solved positions back to local-space rotations
/// 5. Blends the result with the animation pose by `IkChain::weight`
///
/// For each entity with an [`Animator`] and a [`LookAtConstraint`], it
/// rotates the specified bone so its forward direction points at the target.
///
/// This should run **after** `animation_evaluate_system`.
pub fn ik_solve_system(world: &mut World, _dt: f32) {
    // Snapshot the skeleton library (needed for world-space transforms).
    let skeletons: Vec<Skeleton> = match world
        .resource::<AnimationLibrary>()
        .map(|lib| lib.skeletons.clone())
    {
        Some(s) => s,
        None => return,
    };

    // ── Process IkChain components ──────────────────────────────────────
    solve_ik_chains(world, &skeletons);

    // ── Process LookAtConstraint components ─────────────────────────────
    solve_look_at_constraints(world, &skeletons);
}

/// Solve all IK chains in the world.
fn solve_ik_chains(world: &mut World, skeletons: &[Skeleton]) {
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &IkChain)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    for entity in entities {
        let chain = match world.get::<IkChain>(entity) {
            Some(c) => c.clone(),
            None => continue,
        };

        if chain.weight <= 0.0 || chain.bone_indices.is_empty() {
            continue;
        }

        let skeleton_index = match world.get::<Animator>(entity) {
            Some(a) => a.skeleton_index,
            None => continue,
        };
        let skeleton = match skeletons.get(skeleton_index) {
            Some(s) => s,
            None => continue,
        };

        let entity_transform = world
            .get::<Transform>(entity)
            .copied()
            .unwrap_or(Transform::IDENTITY);

        let pose = match world.get::<Animator>(entity) {
            Some(a) => match &a.previous_pose {
                Some(p) => p.clone(),
                None => AnimPose::from_skeleton(skeleton),
            },
            None => continue,
        };

        let mut new_pose = pose.clone();

        if chain.bone_indices.len() == 3 {
            // Two-bone solver.
            let world_positions = compute_world_positions(&pose, skeleton, &entity_transform);
            let root_idx = chain.bone_indices[0];
            let mid_idx = chain.bone_indices[1];
            let tip_idx = chain.bone_indices[2];

            if root_idx >= skeleton.joints.len()
                || mid_idx >= skeleton.joints.len()
                || tip_idx >= skeleton.joints.len()
            {
                continue;
            }

            let pole = chain.pole.unwrap_or(Vec3::Z);
            let (root_rot, mid_rot) = two_bone_ik(
                world_positions[root_idx],
                world_positions[mid_idx],
                world_positions[tip_idx],
                chain.target,
                pole,
            );

            // Blend with weight.
            new_pose.joints[root_idx].rotation = pose.joints[root_idx].rotation.slerp(
                (root_rot * pose.joints[root_idx].rotation).normalize(),
                chain.weight,
            );
            new_pose.joints[mid_idx].rotation = pose.joints[mid_idx].rotation.slerp(
                (mid_rot * pose.joints[mid_idx].rotation).normalize(),
                chain.weight,
            );
        } else if chain.bone_indices.len() >= 2 {
            // FABRIK solver.
            let world_positions = compute_world_positions(&pose, skeleton, &entity_transform);

            let mut chain_positions: Vec<Vec3> = chain
                .bone_indices
                .iter()
                .map(|&idx| {
                    if idx < world_positions.len() {
                        world_positions[idx]
                    } else {
                        Vec3::ZERO
                    }
                })
                .collect();

            fabrik_solve(&mut chain_positions, chain.target, 0.001, 16);

            apply_fabrik_result(
                &mut new_pose,
                skeleton,
                &chain.bone_indices,
                &chain_positions,
                &entity_transform,
                chain.weight,
                &pose,
            );
        }

        // Write the modified pose back.
        if let Some(animator) = world.get_mut::<Animator>(entity) {
            animator.previous_pose = Some(new_pose);
        }
    }
}

/// Solve all look-at constraints in the world.
fn solve_look_at_constraints(world: &mut World, skeletons: &[Skeleton]) {
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &LookAtConstraint)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    for entity in entities {
        let constraint = match world.get::<LookAtConstraint>(entity) {
            Some(c) => c.clone(),
            None => continue,
        };

        let skeleton_index = match world.get::<Animator>(entity) {
            Some(a) => a.skeleton_index,
            None => continue,
        };
        let skeleton = match skeletons.get(skeleton_index) {
            Some(s) => s,
            None => continue,
        };

        if constraint.bone_index >= skeleton.joints.len() {
            continue;
        }

        let entity_transform = world
            .get::<Transform>(entity)
            .copied()
            .unwrap_or(Transform::IDENTITY);

        let pose = match world.get::<Animator>(entity) {
            Some(a) => match &a.previous_pose {
                Some(p) => p.clone(),
                None => AnimPose::from_skeleton(skeleton),
            },
            None => continue,
        };

        let mut new_pose = pose.clone();

        let world_positions = compute_world_positions(&pose, skeleton, &entity_transform);
        let bone_pos = world_positions[constraint.bone_index];
        let to_target = constraint.target - bone_pos;
        let to_target_len = to_target.length();

        if to_target_len < 1e-6 {
            continue;
        }

        let to_target_dir = to_target * (1.0 / to_target_len);

        // Compute the bone's current forward direction in world space.
        // We define "forward" as the direction from this bone toward its child
        // (or +Y if it is a leaf bone).
        let bone_world_rot =
            compute_world_rotation(&pose, skeleton, constraint.bone_index, &entity_transform);
        let current_forward = (bone_world_rot * Vec3::Y).normalize();

        // Compute the rotation from current forward to target direction.
        let world_rot = rotation_arc(current_forward, to_target_dir);

        // Convert to local space: new_local = inv(parent_world) * world_rot * parent_world * local
        let parent_world_rot = if let Some(parent) = skeleton.joints[constraint.bone_index].parent {
            compute_world_rotation(&pose, skeleton, parent, &entity_transform)
        } else {
            entity_transform.rotation
        };

        let local_rot_delta = parent_world_rot.inverse() * world_rot * parent_world_rot;
        new_pose.joints[constraint.bone_index].rotation =
            (local_rot_delta * pose.joints[constraint.bone_index].rotation).normalize();

        if let Some(animator) = world.get_mut::<Animator>(entity) {
            animator.previous_pose = Some(new_pose);
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Compute world-space positions for every joint in the skeleton.
fn compute_world_positions(
    pose: &AnimPose,
    skeleton: &Skeleton,
    entity_transform: &Transform,
) -> Vec<Vec3> {
    let n = skeleton.joints.len();
    let mut world_transforms = vec![Transform::IDENTITY; n];

    for i in 0..n {
        let local = if i < pose.joints.len() {
            pose.joints[i]
        } else {
            skeleton.joints[i].local_transform
        };

        let parent_transform = match skeleton.joints[i].parent {
            Some(parent) => world_transforms[parent],
            None => *entity_transform,
        };

        world_transforms[i] = parent_transform.mul(local);
    }

    world_transforms.iter().map(|t| t.translation).collect()
}

/// Compute the world-space rotation of a specific bone.
fn compute_world_rotation(
    pose: &AnimPose,
    skeleton: &Skeleton,
    bone_idx: usize,
    entity_transform: &Transform,
) -> Quat {
    let mut chain_indices = Vec::new();
    let mut current = Some(bone_idx);
    while let Some(idx) = current {
        chain_indices.push(idx);
        current = skeleton.joints[idx].parent;
    }

    let mut world_rot = entity_transform.rotation;
    for &idx in chain_indices.iter().rev() {
        let local_rot = if idx < pose.joints.len() {
            pose.joints[idx].rotation
        } else {
            skeleton.joints[idx].local_transform.rotation
        };
        world_rot = (world_rot * local_rot).normalize();
    }

    world_rot
}

/// Convert FABRIK solved world positions back to local-space rotations.
fn apply_fabrik_result(
    new_pose: &mut AnimPose,
    skeleton: &Skeleton,
    bone_indices: &[usize],
    solved_positions: &[Vec3],
    entity_transform: &Transform,
    weight: f32,
    original_pose: &AnimPose,
) {
    if bone_indices.len() < 2 || solved_positions.len() < 2 {
        return;
    }

    let world_positions = compute_world_positions(original_pose, skeleton, entity_transform);

    for i in 0..bone_indices.len() - 1 {
        let bone_idx = bone_indices[i];
        let next_idx = bone_indices[i + 1];

        if bone_idx >= skeleton.joints.len() || next_idx >= skeleton.joints.len() {
            continue;
        }

        let old_dir = (world_positions[next_idx] - world_positions[bone_idx]).normalize();
        let new_dir = (solved_positions[i + 1] - solved_positions[i]).normalize();

        if old_dir.length_squared() < 1e-6 || new_dir.length_squared() < 1e-6 {
            continue;
        }

        let world_rot = rotation_arc(old_dir, new_dir);

        // Convert to local space.
        let parent_world_rot = if let Some(parent) = skeleton.joints[bone_idx].parent {
            compute_world_rotation(original_pose, skeleton, parent, entity_transform)
        } else {
            entity_transform.rotation
        };

        let local_rot_delta = parent_world_rot.inverse() * world_rot * parent_world_rot;
        let ik_rotation = (local_rot_delta * original_pose.joints[bone_idx].rotation).normalize();

        new_pose.joints[bone_idx].rotation = original_pose.joints[bone_idx]
            .rotation
            .slerp(ik_rotation, weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_asset::skeleton::Joint;
    use euca_math::Mat4;

    fn test_skeleton(joint_count: usize) -> Skeleton {
        Skeleton {
            joints: (0..joint_count)
                .map(|i| Joint {
                    name: format!("joint_{i}"),
                    parent: if i == 0 { None } else { Some(i - 1) },
                    local_transform: Transform {
                        translation: if i == 0 {
                            Vec3::ZERO
                        } else {
                            Vec3::new(0.0, 1.0, 0.0)
                        },
                        rotation: Quat::IDENTITY,
                        scale: Vec3::ONE,
                    },
                })
                .collect(),
            inverse_bind_matrices: vec![Mat4::IDENTITY; joint_count],
            joint_node_indices: (0..joint_count).collect(),
        }
    }

    // ── Two-bone IK tests ──

    #[test]
    fn two_bone_reaches_target_within_range() {
        // Straight chain along Y: root(0,0,0), mid(0,1,0), tip(0,2,0).
        let root = Vec3::ZERO;
        let mid = Vec3::new(0.0, 1.0, 0.0);
        let tip = Vec3::new(0.0, 2.0, 0.0);
        let target = Vec3::new(1.0, 1.0, 0.0);
        let pole = Vec3::new(0.0, 0.0, 1.0);

        let (root_rot, mid_rot) = two_bone_ik(root, mid, tip, target, pole);

        // Apply rotations and verify the tip reaches the target.
        let new_mid = root + root_rot * (mid - root);
        let new_tip = new_mid + (mid_rot * root_rot) * (tip - mid);

        let dist = (new_tip - target).length();
        assert!(
            dist < 0.15,
            "Two-bone tip should reach target. Distance: {dist}"
        );
    }

    #[test]
    fn two_bone_unreachable_target_clamps() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(0.0, 1.0, 0.0);
        let tip = Vec3::new(0.0, 2.0, 0.0);
        // Target is at distance 5, chain is only 2 long.
        let target = Vec3::new(5.0, 0.0, 0.0);
        let pole = Vec3::new(0.0, 0.0, 1.0);

        let (root_rot, mid_rot) = two_bone_ik(root, mid, tip, target, pole);

        // Chain should extend toward target; the new tip should be near
        // distance 2 from root and in the +X direction.
        let new_mid = root + root_rot * (mid - root);
        let new_tip = new_mid + (mid_rot * root_rot) * (tip - mid);

        assert!(
            new_tip.x > 0.0,
            "Chain should extend toward +X. new_tip = {new_tip:?}"
        );
        let tip_dist_from_root = (new_tip - root).length();
        assert!(
            (tip_dist_from_root - 2.0).abs() < 0.2,
            "Tip should be near chain length from root. dist = {tip_dist_from_root}"
        );
    }

    // ── FABRIK tests ──

    #[test]
    fn fabrik_converges_for_reachable_target() {
        let mut chain = vec![
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let target = Vec3::new(1.5, 1.5, 0.0);

        let converged = fabrik_solve(&mut chain, target, 0.01, 20);

        assert!(converged, "FABRIK should converge for reachable target");

        let tip_dist = (chain[3] - target).length();
        assert!(
            tip_dist < 0.02,
            "Tip should reach target. Distance: {tip_dist}"
        );

        // Root must stay pinned.
        assert!(
            (chain[0] - Vec3::ZERO).length() < 1e-5,
            "FABRIK root should remain pinned"
        );
    }

    #[test]
    fn fabrik_unreachable_returns_false() {
        let mut chain = vec![
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        // Target is at distance 10, chain total length is 2.
        let target = Vec3::new(10.0, 0.0, 0.0);

        let converged = fabrik_solve(&mut chain, target, 0.01, 20);

        assert!(
            !converged,
            "FABRIK should not converge for unreachable target"
        );

        // All joints should be stretched toward the target.
        assert!(chain[2].x > chain[1].x);
        assert!(chain[1].x > chain[0].x);
    }

    #[test]
    fn fabrik_preserves_segment_lengths() {
        let mut chain = vec![
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let original_lengths: Vec<f32> = (0..chain.len() - 1)
            .map(|i| (chain[i + 1] - chain[i]).length())
            .collect();

        let target = Vec3::new(1.0, 1.0, 0.0);
        fabrik_solve(&mut chain, target, 0.01, 20);

        for i in 0..chain.len() - 1 {
            let len = (chain[i + 1] - chain[i]).length();
            assert!(
                (len - original_lengths[i]).abs() < 0.02,
                "Segment {i} length changed: {len} vs {}",
                original_lengths[i]
            );
        }
    }

    // ── Look-at constraint test ──

    #[test]
    fn look_at_faces_target() {
        let skeleton = test_skeleton(3);
        let mut pose = AnimPose::from_skeleton(&skeleton);
        let entity_transform = Transform::IDENTITY;

        let constraint = LookAtConstraint {
            bone_index: 2,
            target: Vec3::new(5.0, 2.0, 0.0),
            up: Vec3::Y,
        };

        // Apply the look-at manually (same logic the system uses).
        let world_positions = compute_world_positions(&pose, &skeleton, &entity_transform);
        let bone_pos = world_positions[constraint.bone_index];
        let to_target = (constraint.target - bone_pos).normalize();

        let bone_world_rot =
            compute_world_rotation(&pose, &skeleton, constraint.bone_index, &entity_transform);
        let current_forward = (bone_world_rot * Vec3::Y).normalize();

        let world_rot = rotation_arc(current_forward, to_target);
        let parent_world_rot = if let Some(parent) = skeleton.joints[constraint.bone_index].parent {
            compute_world_rotation(&pose, &skeleton, parent, &entity_transform)
        } else {
            entity_transform.rotation
        };
        let local_delta = parent_world_rot.inverse() * world_rot * parent_world_rot;
        pose.joints[constraint.bone_index].rotation =
            (local_delta * pose.joints[constraint.bone_index].rotation).normalize();

        // Verify the bone's forward now points roughly toward the target.
        let new_world_rot =
            compute_world_rotation(&pose, &skeleton, constraint.bone_index, &entity_transform);
        let new_forward = (new_world_rot * Vec3::Y).normalize();
        let dot = new_forward.dot(to_target);
        assert!(
            dot > 0.8,
            "Look-at bone should face the target. dot = {dot}"
        );
    }

    // ── Component construction tests ──

    #[test]
    fn ik_chain_component_fields() {
        let chain = IkChain {
            bone_indices: vec![0, 1, 2],
            target: Vec3::new(1.0, 0.0, 0.0),
            pole: Some(Vec3::Z),
            weight: 0.75,
        };
        assert_eq!(chain.bone_indices.len(), 3);
        assert!((chain.weight - 0.75).abs() < 1e-6);
        assert!(chain.pole.is_some());
    }

    #[test]
    fn look_at_constraint_component_fields() {
        let constraint = LookAtConstraint {
            bone_index: 2,
            target: Vec3::new(0.0, 0.0, 5.0),
            up: Vec3::Y,
        };
        assert_eq!(constraint.bone_index, 2);
        assert!((constraint.up.y - 1.0).abs() < 1e-6);
    }
}
