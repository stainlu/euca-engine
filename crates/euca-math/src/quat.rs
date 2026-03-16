use crate::Vec3;
use serde::{Deserialize, Serialize};
use std::ops::Mul;

/// Quaternion (xyzw layout, unit quaternion for rotations).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Default for Quat {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    /// The identity quaternion (no rotation).
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    /// Creates a quaternion from raw x, y, z, w components.
    #[inline(always)]
    pub const fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Create a quaternion from axis-angle rotation.
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let half = angle * 0.5;
        let s = half.sin();
        let c = half.cos();
        let a = axis.normalize();
        Self {
            x: a.x * s,
            y: a.y * s,
            z: a.z * s,
            w: c,
        }
    }

    /// Create from Euler angles (yaw, pitch, roll) in YXZ order.
    #[inline]
    pub fn from_euler(yaw: f32, pitch: f32, roll: f32) -> Self {
        let (sy, cy) = (yaw * 0.5).sin_cos();
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sr, cr) = (roll * 0.5).sin_cos();

        Self {
            x: cy * sp * cr + sy * cp * sr,
            y: sy * cp * cr - cy * sp * sr,
            z: cy * cp * sr - sy * sp * cr,
            w: cy * cp * cr + sy * sp * sr,
        }
    }

    /// Returns the length (norm) of the quaternion.
    #[inline(always)]
    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// Returns a unit-length quaternion in the same direction.
    #[inline]
    pub fn normalize(self) -> Self {
        let inv = 1.0 / self.length();
        Self {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
            w: self.w * inv,
        }
    }

    /// Returns the conjugate (inverse for unit quaternions).
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Spherical linear interpolation.
    #[inline]
    pub fn slerp(self, mut end: Self, t: f32) -> Self {
        let mut dot = self.x * end.x + self.y * end.y + self.z * end.z + self.w * end.w;

        if dot < 0.0 {
            end = Self {
                x: -end.x,
                y: -end.y,
                z: -end.z,
                w: -end.w,
            };
            dot = -dot;
        }

        if dot > 0.9995 {
            return Self {
                x: self.x + (end.x - self.x) * t,
                y: self.y + (end.y - self.y) * t,
                z: self.z + (end.z - self.z) * t,
                w: self.w + (end.w - self.w) * t,
            }
            .normalize();
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let s0 = ((1.0 - t) * theta).sin() / sin_theta;
        let s1 = (t * theta).sin() / sin_theta;

        Self {
            x: self.x * s0 + end.x * s1,
            y: self.y * s0 + end.y * s1,
            z: self.z * s0 + end.z * s1,
            w: self.w * s0 + end.w * s1,
        }
    }
}

/// Compose rotations: `a * b` applies `b` first, then `a`.
impl Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        }
    }
}

/// Rotate a Vec3 by this quaternion.
impl Mul<Vec3> for Quat {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        u * (2.0 * u.dot(v)) + v * (s * s - u.dot(u)) + u.cross(v) * (2.0 * s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn identity_rotation() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = Quat::IDENTITY * v;
        assert!((r.x - v.x).abs() < 1e-6);
        assert!((r.y - v.y).abs() < 1e-6);
        assert!((r.z - v.z).abs() < 1e-6);
    }

    #[test]
    fn rotate_90_around_z() {
        let q = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
        let r = q * Vec3::X;
        assert!(r.x.abs() < 1e-5);
        assert!((r.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn inverse_undoes_rotation() {
        let q = Quat::from_axis_angle(Vec3::Y, 1.0);
        let v = Vec3::new(1.0, 2.0, 3.0);
        let back = q.inverse() * (q * v);
        assert!((back.x - v.x).abs() < 1e-4);
        assert!((back.y - v.y).abs() < 1e-4);
        assert!((back.z - v.z).abs() < 1e-4);
    }

    #[test]
    fn slerp_halfway() {
        let a = Quat::IDENTITY;
        let b = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
        let mid = a.slerp(b, 0.5);
        let v = mid * Vec3::X;
        let expected = FRAC_PI_2 / 2.0;
        assert!((v.x - expected.cos()).abs() < 1e-4);
        assert!((v.y - expected.sin()).abs() < 1e-4);
    }
}
