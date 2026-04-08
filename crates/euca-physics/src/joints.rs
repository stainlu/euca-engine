//! Joint constraints connecting two physics bodies.
//!
//! Joints constrain the relative motion between two entities.
//! Resolved during the constraint solver iterations alongside contact constraints.

use euca_ecs::Entity;
use euca_math::{Quat, Vec3};

/// Motor that drives a revolute joint toward a target angular velocity.
#[derive(Clone, Copy, Debug)]
pub struct JointMotor {
    /// Target angular velocity in radians per second.
    pub target_velocity: f32,
    /// Maximum torque the motor can exert (N-m).
    pub max_torque: f32,
}

/// A joint connecting two entities.
#[derive(Clone, Debug)]
pub struct Joint {
    /// First body (must have PhysicsBody + LocalTransform).
    pub entity_a: Entity,
    /// Second body (must have PhysicsBody + LocalTransform).
    pub entity_b: Entity,
    /// Type of constraint.
    pub kind: JointKind,
    /// How strongly to enforce the constraint (0.0 = soft, 1.0 = rigid).
    pub stiffness: f32,
}

/// The type of joint constraint.
#[derive(Clone, Debug)]
pub enum JointKind {
    /// Maintains a fixed distance between two anchor points.
    /// `anchor_a` and `anchor_b` are local-space offsets from each entity's origin.
    Distance {
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest_length: f32,
        /// Minimum allowed distance (`None` = can compress to zero).
        min_length: Option<f32>,
        /// Maximum allowed distance (`None` = can stretch infinitely).
        max_length: Option<f32>,
    },
    /// Ball-and-socket: constrains position but allows free rotation.
    /// Anchors are local-space attachment points.
    BallAndSocket { anchor_a: Vec3, anchor_b: Vec3 },
    /// Revolute (hinge): constrains to rotation around a single axis.
    Revolute {
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
        /// Lower angle limit in radians (`None` = no limit).
        min_angle: Option<f32>,
        /// Upper angle limit in radians (`None` = no limit).
        max_angle: Option<f32>,
        /// Optional motor driving the joint.
        motor: Option<JointMotor>,
    },
}

impl Joint {
    /// Create a distance joint between two entities.
    pub fn distance(
        entity_a: Entity,
        entity_b: Entity,
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest_length: f32,
    ) -> Self {
        Self {
            entity_a,
            entity_b,
            kind: JointKind::Distance {
                anchor_a,
                anchor_b,
                rest_length,
                min_length: None,
                max_length: None,
            },
            stiffness: 1.0,
        }
    }

    /// Create a ball-and-socket joint.
    pub fn ball_and_socket(
        entity_a: Entity,
        entity_b: Entity,
        anchor_a: Vec3,
        anchor_b: Vec3,
    ) -> Self {
        Self {
            entity_a,
            entity_b,
            kind: JointKind::BallAndSocket { anchor_a, anchor_b },
            stiffness: 1.0,
        }
    }

    /// Create a revolute (hinge) joint.
    pub fn revolute(
        entity_a: Entity,
        entity_b: Entity,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
    ) -> Self {
        Self {
            entity_a,
            entity_b,
            kind: JointKind::Revolute {
                anchor_a,
                anchor_b,
                axis,
                min_angle: None,
                max_angle: None,
                motor: None,
            },
            stiffness: 1.0,
        }
    }

    /// Set the stiffness (0.0 = soft spring, 1.0 = rigid).
    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness.clamp(0.0, 1.0);
        self
    }

    /// Set angle limits on a revolute joint (radians).
    ///
    /// # Panics
    /// Panics if this joint is not `JointKind::Revolute`.
    pub fn with_angle_limits(mut self, min: f32, max: f32) -> Self {
        match &mut self.kind {
            JointKind::Revolute {
                min_angle,
                max_angle,
                ..
            } => {
                *min_angle = Some(min);
                *max_angle = Some(max);
            }
            _ => panic!("with_angle_limits called on non-Revolute joint"),
        }
        self
    }

    /// Attach a motor to a revolute joint.
    ///
    /// # Panics
    /// Panics if this joint is not `JointKind::Revolute`.
    pub fn with_motor(mut self, target_velocity: f32, max_torque: f32) -> Self {
        match &mut self.kind {
            JointKind::Revolute { motor, .. } => {
                *motor = Some(JointMotor {
                    target_velocity,
                    max_torque,
                });
            }
            _ => panic!("with_motor called on non-Revolute joint"),
        }
        self
    }

    /// Set distance limits on a distance joint.
    ///
    /// # Panics
    /// Panics if this joint is not `JointKind::Distance`.
    pub fn with_distance_limits(mut self, min: f32, max: f32) -> Self {
        match &mut self.kind {
            JointKind::Distance {
                min_length,
                max_length,
                ..
            } => {
                *min_length = Some(min);
                *max_length = Some(max);
            }
            _ => panic!("with_distance_limits called on non-Distance joint"),
        }
        self
    }

    /// Returns `true` if this joint has a motor attached.
    pub fn has_motor(&self) -> bool {
        matches!(&self.kind, JointKind::Revolute { motor: Some(_), .. })
    }

    /// Compute the position correction for this joint given current positions
    /// and rotations.
    ///
    /// Returns `(correction_a, correction_b)` -- position deltas to apply.
    pub fn solve(
        &self,
        pos_a: Vec3,
        pos_b: Vec3,
        rot_a: Quat,
        rot_b: Quat,
        is_a_dynamic: bool,
        is_b_dynamic: bool,
    ) -> (Vec3, Vec3) {
        match &self.kind {
            JointKind::Distance {
                anchor_a,
                anchor_b,
                rest_length,
                min_length,
                max_length,
            } => {
                let world_a = pos_a + *anchor_a;
                let world_b = pos_b + *anchor_b;
                let delta = world_b - world_a;
                let dist = delta.length();
                if dist < 1e-8 {
                    return (Vec3::ZERO, Vec3::ZERO);
                }

                // Determine target distance based on limits.
                let target = if min_length.is_some() || max_length.is_some() {
                    // With limits: free range between min and max.
                    let lo = min_length.unwrap_or(0.0);
                    let hi = max_length.unwrap_or(f32::MAX);
                    if dist < lo {
                        lo
                    } else if dist > hi {
                        hi
                    } else {
                        // Within free range -- no correction.
                        return (Vec3::ZERO, Vec3::ZERO);
                    }
                } else {
                    // No limits: maintain rest length.
                    *rest_length
                };

                let error = dist - target;
                let dir = delta * (1.0 / dist);
                let correction = dir * error * self.stiffness;

                distribute_correction(correction, is_a_dynamic, is_b_dynamic)
            }
            JointKind::BallAndSocket { anchor_a, anchor_b } => {
                let world_a = pos_a + *anchor_a;
                let world_b = pos_b + *anchor_b;
                let delta = world_b - world_a;
                let correction = delta * self.stiffness;

                distribute_correction(correction, is_a_dynamic, is_b_dynamic)
            }
            JointKind::Revolute {
                anchor_a,
                anchor_b,
                axis,
                min_angle,
                max_angle,
                ..
            } => {
                // Position constraint: same as ball-and-socket
                let world_a = pos_a + *anchor_a;
                let world_b = pos_b + *anchor_b;
                let delta = world_b - world_a;

                // Project delta onto the plane perpendicular to the hinge axis
                // (allow movement along the axis, correct perpendicular error)
                let axis_component = *axis * delta.dot(*axis);
                let perp_error = delta - axis_component;
                let mut correction = perp_error * self.stiffness;

                // Angle limit enforcement via position-level correction.
                if min_angle.is_some() || max_angle.is_some() {
                    let angle = relative_hinge_angle(rot_a, rot_b, *axis);
                    let lo = min_angle.unwrap_or(-std::f32::consts::PI);
                    let hi = max_angle.unwrap_or(std::f32::consts::PI);
                    let angle_error = if angle < lo {
                        lo - angle
                    } else if angle > hi {
                        hi - angle
                    } else {
                        0.0
                    };
                    if angle_error.abs() > 1e-6 {
                        // Convert angular error into a positional correction
                        // using the anchor offset as the lever arm.
                        let arm = world_b - pos_b;
                        let arm_len = arm.length();
                        let lever = arm_len.max(0.01);
                        let linear_error = angle_error * lever;
                        let lever_dir = if arm_len > 1e-6 {
                            arm.normalize()
                        } else {
                            perpendicular_to(*axis)
                        };
                        let cross = axis.cross(lever_dir);
                        let correction_dir = if cross.length() > 1e-6 {
                            cross.normalize()
                        } else {
                            Vec3::ZERO
                        };
                        correction = correction + correction_dir * linear_error * self.stiffness;
                    }
                }

                distribute_correction(correction, is_a_dynamic, is_b_dynamic)
            }
        }
    }

    /// Compute velocity-level corrections for motor-driven joints.
    ///
    /// For revolute joints with a motor, returns the angular impulses to apply to
    /// body A and body B. Returns `(Vec3::ZERO, Vec3::ZERO)` if no motor is set.
    pub fn solve_velocity(
        &self,
        angular_vel_a: Vec3,
        angular_vel_b: Vec3,
        is_a_dynamic: bool,
        is_b_dynamic: bool,
        dt: f32,
    ) -> (Vec3, Vec3) {
        match &self.kind {
            JointKind::Revolute { axis, motor, .. } => {
                let motor = match motor {
                    Some(m) => m,
                    None => return (Vec3::ZERO, Vec3::ZERO),
                };
                let current_rel_vel = (angular_vel_b - angular_vel_a).dot(*axis);
                let vel_error = motor.target_velocity - current_rel_vel;
                let impulse_magnitude = (vel_error * self.stiffness)
                    .clamp(-motor.max_torque * dt, motor.max_torque * dt);
                let impulse = *axis * impulse_magnitude;
                // Motor drives B toward target relative to A: B gets positive
                // impulse along axis, A gets the reaction (opposite sign).
                distribute_angular_impulse(impulse, is_a_dynamic, is_b_dynamic)
            }
            _ => (Vec3::ZERO, Vec3::ZERO),
        }
    }
}

/// Compute the relative rotation angle between two bodies projected onto a hinge axis.
///
/// Returns the signed angle (in radians) of `rot_b` relative to `rot_a` around `axis`.
fn relative_hinge_angle(rot_a: Quat, rot_b: Quat, axis: Vec3) -> f32 {
    let rel = rot_a.inverse() * rot_b;
    // Extract the angle around the hinge axis from the quaternion's vector part.
    let proj = Vec3::new(rel.x, rel.y, rel.z).dot(axis);
    proj.atan2(rel.w) * 2.0
}

/// Return a unit vector perpendicular to the given axis.
fn perpendicular_to(axis: Vec3) -> Vec3 {
    let candidate = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    axis.cross(candidate).normalize()
}

/// Distribute an angular impulse between two bodies.
///
/// Body B receives the impulse (driven), body A receives the reaction (negated).
/// When both are dynamic, the impulse is split equally.
fn distribute_angular_impulse(
    impulse: Vec3,
    is_a_dynamic: bool,
    is_b_dynamic: bool,
) -> (Vec3, Vec3) {
    match (is_a_dynamic, is_b_dynamic) {
        (true, true) => (impulse * (-0.5), impulse * 0.5),
        (true, false) => (impulse * (-1.0), Vec3::ZERO),
        (false, true) => (Vec3::ZERO, impulse),
        (false, false) => (Vec3::ZERO, Vec3::ZERO),
    }
}

/// Distribute a correction between two bodies based on their dynamic status.
fn distribute_correction(correction: Vec3, is_a_dynamic: bool, is_b_dynamic: bool) -> (Vec3, Vec3) {
    match (is_a_dynamic, is_b_dynamic) {
        (true, true) => (correction * 0.5, correction * (-0.5)),
        (true, false) => (correction, Vec3::ZERO),
        (false, true) => (Vec3::ZERO, correction * (-1.0)),
        (false, false) => (Vec3::ZERO, Vec3::ZERO),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_4, PI};

    fn make_entity(index: u32) -> Entity {
        Entity::from_raw(index, 0)
    }

    // Identity rotation shorthand for tests that don't involve rotation.
    const ID: Quat = Quat::IDENTITY;

    // ── Existing tests (unchanged behavior) ────────────────────────────────

    #[test]
    fn distance_joint_at_rest() {
        let joint = Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0);

        // Bodies are exactly rest_length apart -- no correction
        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0), ID, ID, true, true);
        assert!(ca.length() < 1e-5);
        assert!(cb.length() < 1e-5);
    }

    #[test]
    fn distance_joint_stretched() {
        let joint = Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0);

        // Bodies are 10 apart but rest length is 5 -- should pull together
        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), ID, ID, true, true);
        assert!(ca.x > 0.0, "A should move toward B");
        assert!(cb.x < 0.0, "B should move toward A");
    }

    #[test]
    fn ball_and_socket_offset() {
        let joint = Joint::ball_and_socket(
            make_entity(0),
            make_entity(1),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
        );

        // Anchors should meet: A at (0,0,0)+anchor = (1,0,0), B at (3,0,0)+anchor = (2,0,0)
        // Error = B_anchor - A_anchor = (2,0,0)-(1,0,0) = (1,0,0)
        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0), ID, ID, true, true);
        assert!(ca.x > 0.0);
        assert!(cb.x < 0.0);
    }

    #[test]
    fn static_body_doesnt_move() {
        let joint = Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0);

        // A is static -- only B should receive correction
        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), ID, ID, false, true);
        assert!(ca.length() < 1e-5, "Static body shouldn't move");
        assert!(cb.x < 0.0, "Dynamic body should be pulled");
    }

    // ── Distance joint with limits ─────────────────────────────────────────

    #[test]
    fn distance_joint_with_limits_at_rest() {
        // Within the [3, 7] free range at distance 5 -- no correction.
        let joint = Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0)
            .with_distance_limits(3.0, 7.0);

        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0), ID, ID, true, true);
        assert!(ca.length() < 1e-5);
        assert!(cb.length() < 1e-5);
    }

    #[test]
    fn distance_joint_stretched_beyond_max() {
        // Distance 10.0 exceeds max_length 7.0 -- should correct toward 7.0.
        let joint = Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0)
            .with_distance_limits(3.0, 7.0);

        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), ID, ID, true, true);
        assert!(ca.x > 0.0, "A should move toward B");
        assert!(cb.x < 0.0, "B should move toward A");
        // Correction magnitude should correspond to error of 3.0 (10-7), not 5.0 (10-5).
        let total = (ca - cb).length();
        assert!(
            (total - 3.0).abs() < 1e-4,
            "total correction should be 3.0, got {total}"
        );
    }

    #[test]
    fn distance_joint_compressed_below_min() {
        // Distance 2.0 is below min_length 3.0 -- should push apart.
        let joint = Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0)
            .with_distance_limits(3.0, 7.0);

        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), ID, ID, true, true);
        assert!(ca.x < 0.0, "A should move away from B");
        assert!(cb.x > 0.0, "B should move away from A");
    }

    #[test]
    fn distance_joint_without_limits_preserves_behavior() {
        // No limits set -- behaves exactly as rest_length constraint.
        let joint = Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0);

        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), ID, ID, true, true);
        let total = (ca - cb).length();
        assert!(
            (total - 5.0).abs() < 1e-4,
            "should correct full error to rest_length"
        );
    }

    // ── Revolute joint with angle limits ───────────────────────────────────

    #[test]
    fn revolute_angle_within_limits_no_correction() {
        // Both anchors at origin, bodies co-located -- positional error is zero.
        let joint = Joint::revolute(
            make_entity(0),
            make_entity(1),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
        )
        .with_angle_limits(-FRAC_PI_4, FRAC_PI_4);

        // Both at identity -- relative angle is 0, well within limits.
        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::ZERO, ID, ID, true, true);
        // No positional error and angle is within limits -- zero correction.
        assert!(ca.length() < 1e-4);
        assert!(cb.length() < 1e-4);
    }

    #[test]
    fn revolute_angle_past_limit_gets_corrected() {
        // B is rotated 90 degrees around Y, but limit is [-pi/4, pi/4].
        let joint = Joint::revolute(
            make_entity(0),
            make_entity(1),
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::Y,
        )
        .with_angle_limits(-FRAC_PI_4, FRAC_PI_4);

        let rot_b = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), ID, rot_b, true, true);
        // Some angular correction should be produced.
        let total = (ca - cb).length();
        assert!(
            total > 1e-3,
            "should produce correction for angle past limit"
        );
    }

    // ── Revolute motor ─────────────────────────────────────────────────────

    #[test]
    fn revolute_motor_applies_impulse() {
        let joint = Joint::revolute(
            make_entity(0),
            make_entity(1),
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::Y,
        )
        .with_motor(5.0, 100.0);

        // Both bodies at rest -- motor drives toward target_velocity 5.0 rad/s.
        let dt = 1.0 / 60.0;
        let (imp_a, imp_b) = joint.solve_velocity(Vec3::ZERO, Vec3::ZERO, true, true, dt);
        // Impulse should be along the hinge axis (Y).
        assert!(
            imp_a.y.abs() > 1e-5 || imp_b.y.abs() > 1e-5,
            "motor should produce Y-axis impulse"
        );
        // A gets negative (reaction), B gets positive (driven).
        assert!(imp_a.y < 0.0, "body A receives reaction impulse");
        assert!(imp_b.y > 0.0, "body B receives drive impulse");
    }

    #[test]
    fn motor_respects_max_torque() {
        let joint = Joint::revolute(
            make_entity(0),
            make_entity(1),
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::Y,
        )
        .with_motor(1000.0, 2.0); // Huge velocity difference, tiny max torque.

        let dt = 1.0 / 60.0;
        let (imp_a, imp_b) = joint.solve_velocity(Vec3::ZERO, Vec3::ZERO, true, true, dt);
        let total = (imp_a - imp_b).length();
        // The total impulse magnitude must be <= max_torque * dt = 2.0 / 60.0.
        let max_impulse = 2.0 * dt;
        assert!(
            total <= max_impulse + 1e-5,
            "impulse {total} should not exceed max_torque * dt = {max_impulse}"
        );
    }

    #[test]
    fn motor_converges_over_time() {
        let joint = Joint::revolute(
            make_entity(0),
            make_entity(1),
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::Y,
        )
        .with_motor(5.0, 100.0);

        let dt = 1.0 / 60.0;
        let mut ang_vel_b = Vec3::ZERO;
        for _ in 0..120 {
            let (_, imp_b) = joint.solve_velocity(Vec3::ZERO, ang_vel_b, false, true, dt);
            ang_vel_b = ang_vel_b + imp_b;
        }
        // After 2 seconds at 60 Hz with plenty of torque, should approach target.
        let rel_vel = ang_vel_b.dot(Vec3::Y);
        assert!(
            (rel_vel - 5.0).abs() < 0.5,
            "relative velocity {rel_vel} should approach target 5.0"
        );
    }

    // ── Builder panic tests ────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "with_angle_limits called on non-Revolute joint")]
    fn angle_limits_on_distance_panics() {
        Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0)
            .with_angle_limits(-PI, PI);
    }

    #[test]
    #[should_panic(expected = "with_motor called on non-Revolute joint")]
    fn motor_on_distance_panics() {
        Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0)
            .with_motor(1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "with_distance_limits called on non-Distance joint")]
    fn distance_limits_on_revolute_panics() {
        Joint::revolute(
            make_entity(0),
            make_entity(1),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
        )
        .with_distance_limits(1.0, 5.0);
    }
}
