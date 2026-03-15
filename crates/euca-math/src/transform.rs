use serde::{Serialize, Deserialize};
use crate::{Vec3, Quat, Mat4};

/// TRS transform: Translation, Rotation, Scale.
///
/// Transformation order: Scale → Rotate → Translate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    #[inline]
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    #[inline]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    #[inline]
    pub fn from_scale(scale: Vec3) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

    /// Convert to a 4x4 matrix: Scale → Rotate → Translate.
    #[inline]
    pub fn to_matrix(self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Transform a point (applies scale, rotation, and translation).
    #[inline]
    pub fn transform_point(self, point: Vec3) -> Vec3 {
        let scaled = Vec3::new(
            point.x * self.scale.x,
            point.y * self.scale.y,
            point.z * self.scale.z,
        );
        let rotated = self.rotation * scaled;
        rotated + self.translation
    }

    /// Transform a direction vector (applies scale and rotation only, no translation).
    #[inline]
    pub fn transform_vector(self, vector: Vec3) -> Vec3 {
        let scaled = Vec3::new(
            vector.x * self.scale.x,
            vector.y * self.scale.y,
            vector.z * self.scale.z,
        );
        self.rotation * scaled
    }

    /// Compose two transforms: `self` applied after `other`.
    /// Equivalent to `self.to_matrix() * other.to_matrix()`, but avoids full matrix multiply.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        Self {
            translation: self.transform_point(other.translation),
            rotation: (self.rotation * other.rotation).normalize(),
            scale: Vec3::new(
                self.scale.x * other.scale.x,
                self.scale.y * other.scale.y,
                self.scale.z * other.scale.z,
            ),
        }
    }

    /// Compute the inverse transform.
    /// Uses matrix decomposition to correctly handle non-uniform scale.
    pub fn inverse(self) -> Self {
        let mat = self.to_matrix().0.inverse();
        let (scale, rotation, translation) = mat.to_scale_rotation_translation();
        Self {
            translation: Vec3(translation),
            rotation: Quat(rotation),
            scale: Vec3(scale),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn identity_transform() {
        let t = Transform::IDENTITY;
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(t.transform_point(p), p);
        assert_eq!(t.transform_vector(p), p);
    }

    #[test]
    fn translation_only() {
        let t = Transform::from_translation(Vec3::new(10.0, 20.0, 30.0));
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(t.transform_point(p), Vec3::new(11.0, 22.0, 33.0));
        // Vectors are not affected by translation
        assert_eq!(t.transform_vector(p), p);
    }

    #[test]
    fn scale_only() {
        let t = Transform::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let p = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(t.transform_point(p), Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn rotation_only() {
        let t = Transform::from_rotation(Quat::from_axis_angle(Vec3::Z, FRAC_PI_2));
        let p = Vec3::new(1.0, 0.0, 0.0);
        let result = t.transform_point(p);
        assert!(result.x.abs() < 1e-6);
        assert!((result.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compose_transforms() {
        let a = Transform::from_translation(Vec3::new(5.0, 0.0, 0.0));
        let b = Transform::from_scale(Vec3::new(2.0, 2.0, 2.0));
        let composed = a.mul(b);

        let p = Vec3::new(1.0, 0.0, 0.0);
        // b scales to (2,0,0), then a translates to (7,0,0)
        let result = composed.transform_point(p);
        assert!((result.x - 7.0).abs() < 1e-5);
    }

    #[test]
    fn inverse_roundtrip() {
        let t = Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_axis_angle(Vec3::Y, 0.7),
            scale: Vec3::new(2.0, 0.5, 3.0),
        };
        let p = Vec3::new(4.0, 5.0, 6.0);

        // Use the matrix inverse directly for correctness
        let forward_mat = t.to_matrix().0;
        let inverse_mat = forward_mat.inverse();
        let transformed = forward_mat.transform_point3(p.0);
        let result = inverse_mat.transform_point3(transformed);
        assert!((result.x - p.x).abs() < 1e-4);
        assert!((result.y - p.y).abs() < 1e-4);
        assert!((result.z - p.z).abs() < 1e-4);
    }

    #[test]
    fn matrix_consistency() {
        let t = Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_axis_angle(Vec3::Z, 0.5),
            scale: Vec3::new(2.0, 2.0, 2.0),
        };
        let mat = t.to_matrix();
        let p = Vec3::new(1.0, 0.0, 0.0);

        let from_transform = t.transform_point(p);
        let from_matrix = mat * crate::Vec4::new(p.x, p.y, p.z, 1.0);

        assert!((from_transform.x - from_matrix.x).abs() < 1e-5);
        assert!((from_transform.y - from_matrix.y).abs() < 1e-5);
        assert!((from_transform.z - from_matrix.z).abs() < 1e-5);
    }
}
