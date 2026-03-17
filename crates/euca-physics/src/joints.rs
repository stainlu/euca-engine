//! Joint constraints connecting two physics bodies.
//!
//! Joints constrain the relative motion between two entities.
//! Resolved during the constraint solver iterations alongside contact constraints.

use euca_ecs::Entity;
use euca_math::Vec3;

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
    },
    /// Ball-and-socket: constrains position but allows free rotation.
    /// Anchors are local-space attachment points.
    BallAndSocket { anchor_a: Vec3, anchor_b: Vec3 },
    /// Revolute (hinge): constrains to rotation around a single axis.
    Revolute {
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
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
            },
            stiffness: 1.0,
        }
    }

    /// Set the stiffness (0.0 = soft spring, 1.0 = rigid).
    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness.clamp(0.0, 1.0);
        self
    }

    /// Compute the position correction for this joint given current positions.
    /// Returns (correction_a, correction_b) — position deltas to apply.
    pub fn solve(
        &self,
        pos_a: Vec3,
        pos_b: Vec3,
        is_a_dynamic: bool,
        is_b_dynamic: bool,
    ) -> (Vec3, Vec3) {
        match &self.kind {
            JointKind::Distance {
                anchor_a,
                anchor_b,
                rest_length,
            } => {
                let world_a = pos_a + *anchor_a;
                let world_b = pos_b + *anchor_b;
                let delta = world_b - world_a;
                let dist = delta.length();
                if dist < 1e-8 {
                    return (Vec3::ZERO, Vec3::ZERO);
                }
                let error = dist - rest_length;
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
            } => {
                // Position constraint: same as ball-and-socket
                let world_a = pos_a + *anchor_a;
                let world_b = pos_b + *anchor_b;
                let delta = world_b - world_a;

                // Project delta onto the plane perpendicular to the hinge axis
                // (allow movement along the axis, correct perpendicular error)
                let axis_component = *axis * delta.dot(*axis);
                let perp_error = delta - axis_component;
                let correction = perp_error * self.stiffness;

                distribute_correction(correction, is_a_dynamic, is_b_dynamic)
            }
        }
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

    fn make_entity(index: u32) -> Entity {
        Entity::from_raw(index, 0)
    }

    #[test]
    fn distance_joint_at_rest() {
        let joint = Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0);

        // Bodies are exactly rest_length apart — no correction
        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0), true, true);
        assert!(ca.length() < 1e-5);
        assert!(cb.length() < 1e-5);
    }

    #[test]
    fn distance_joint_stretched() {
        let joint = Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0);

        // Bodies are 10 apart but rest length is 5 — should pull together
        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), true, true);
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
        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0), true, true);
        assert!(ca.x > 0.0);
        assert!(cb.x < 0.0);
    }

    #[test]
    fn static_body_doesnt_move() {
        let joint = Joint::distance(make_entity(0), make_entity(1), Vec3::ZERO, Vec3::ZERO, 5.0);

        // A is static — only B should receive correction
        let (ca, cb) = joint.solve(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), false, true);
        assert!(ca.length() < 1e-5, "Static body shouldn't move");
        assert!(cb.x < 0.0, "Dynamic body should be pulled");
    }
}
