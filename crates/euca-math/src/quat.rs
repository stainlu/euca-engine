use crate::Vec3;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut, Mul};

/// Quaternion rotation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Quat(pub glam::Quat);

impl Default for Quat {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Self = Self(glam::Quat::IDENTITY);

    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        Self(glam::Quat::from_axis_angle(axis.0, angle))
    }

    #[inline]
    pub fn from_euler(yaw: f32, pitch: f32, roll: f32) -> Self {
        Self(glam::Quat::from_euler(
            glam::EulerRot::YXZ,
            yaw,
            pitch,
            roll,
        ))
    }

    #[inline]
    pub fn inverse(self) -> Self {
        Self(self.0.inverse())
    }

    #[inline]
    pub fn normalize(self) -> Self {
        Self(self.0.normalize())
    }

    #[inline]
    pub fn slerp(self, end: Self, t: f32) -> Self {
        Self(self.0.slerp(end.0, t))
    }
}

impl Deref for Quat {
    type Target = glam::Quat;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Quat {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Compose rotations: `a * b` applies `b` first, then `a`.
impl Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

/// Rotate a vector by this quaternion.
impl Mul<Vec3> for Quat {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3(self.0 * rhs.0)
    }
}

impl From<glam::Quat> for Quat {
    #[inline]
    fn from(q: glam::Quat) -> Self {
        Self(q)
    }
}

impl From<Quat> for glam::Quat {
    #[inline]
    fn from(q: Quat) -> Self {
        q.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn identity_rotation() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rotated = Quat::IDENTITY * v;
        assert_eq!(rotated, v);
    }

    #[test]
    fn rotate_90_degrees_around_z() {
        let q = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q * v;
        // X axis rotated 90° around Z → Y axis
        assert!((rotated.x).abs() < 1e-6);
        assert!((rotated.y - 1.0).abs() < 1e-6);
        assert!((rotated.z).abs() < 1e-6);
    }

    #[test]
    fn inverse_undoes_rotation() {
        let q = Quat::from_axis_angle(Vec3::Y, 1.0);
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rotated = q * v;
        let back = q.inverse() * rotated;
        assert!((back.x - v.x).abs() < 1e-5);
        assert!((back.y - v.y).abs() < 1e-5);
        assert!((back.z - v.z).abs() < 1e-5);
    }

    #[test]
    fn slerp_halfway() {
        let a = Quat::IDENTITY;
        let b = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
        let mid = a.slerp(b, 0.5);
        let v = mid * Vec3::X;
        // Should be ~45° rotation
        let expected_angle = FRAC_PI_2 / 2.0;
        assert!((v.x - expected_angle.cos()).abs() < 1e-5);
        assert!((v.y - expected_angle.sin()).abs() < 1e-5);
    }
}
