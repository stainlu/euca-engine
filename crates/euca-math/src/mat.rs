use std::ops::{Mul, Deref, DerefMut};
use serde::{Serialize, Deserialize};
use crate::{Vec3, Vec4, Quat};

/// 4x4 matrix (column-major, matching GPU conventions).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Mat4(pub glam::Mat4);

impl Default for Mat4 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat4 {
    pub const IDENTITY: Self = Self(glam::Mat4::IDENTITY);
    pub const ZERO: Self = Self(glam::Mat4::ZERO);

    #[inline]
    pub fn from_translation(translation: Vec3) -> Self {
        Self(glam::Mat4::from_translation(translation.0))
    }

    #[inline]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self(glam::Mat4::from_quat(rotation.0))
    }

    #[inline]
    pub fn from_scale(scale: Vec3) -> Self {
        Self(glam::Mat4::from_scale(scale.0))
    }

    #[inline]
    pub fn from_scale_rotation_translation(scale: Vec3, rotation: Quat, translation: Vec3) -> Self {
        Self(glam::Mat4::from_scale_rotation_translation(scale.0, rotation.0, translation.0))
    }

    #[inline]
    pub fn perspective_lh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        Self(glam::Mat4::perspective_lh(fov_y_radians, aspect_ratio, z_near, z_far))
    }

    #[inline]
    pub fn look_at_lh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        Self(glam::Mat4::look_at_lh(eye.0, center.0, up.0))
    }

    #[inline]
    pub fn inverse(self) -> Self {
        Self(self.0.inverse())
    }

    #[inline]
    pub fn transpose(self) -> Self {
        Self(self.0.transpose())
    }

    #[inline]
    pub fn determinant(self) -> f32 {
        self.0.determinant()
    }
}

impl Deref for Mat4 {
    type Target = glam::Mat4;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Mat4 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Mul for Mat4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, rhs: Vec4) -> Vec4 {
        Vec4(self.0 * rhs.0)
    }
}

impl From<glam::Mat4> for Mat4 {
    #[inline]
    fn from(m: glam::Mat4) -> Self {
        Self(m)
    }
}

impl From<Mat4> for glam::Mat4 {
    #[inline]
    fn from(m: Mat4) -> Self {
        m.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_mul() {
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let result = Mat4::IDENTITY * m;
        assert_eq!(result, m);
    }

    #[test]
    fn inverse_identity() {
        let m = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 2.0, 2.0),
            Quat::from_axis_angle(Vec3::Z, 0.5),
            Vec3::new(10.0, 20.0, 30.0),
        );
        let inv = m.inverse();
        let result = m * inv;
        // Should be approximately identity
        let id = Mat4::IDENTITY;
        for i in 0..4 {
            for j in 0..4 {
                assert!((result.0.col(i)[j] - id.0.col(i)[j]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn transform_point_with_matrix() {
        let m = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let p = Vec4::new(1.0, 2.0, 3.0, 1.0); // w=1 for point
        let result = m * p;
        assert_eq!(result, Vec4::new(11.0, 2.0, 3.0, 1.0));
    }
}
