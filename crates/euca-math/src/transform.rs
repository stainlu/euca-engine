use crate::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// TRS transform: Translation, Rotation, Scale.
/// Transformation order: Scale -> Rotate -> Translate.
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
    /// The identity transform (no translation, rotation, or scale).
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Creates a transform with only a translation.
    #[inline]
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    /// Creates a transform with only a rotation.
    #[inline]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    /// Creates a transform with only a scale.
    #[inline]
    pub fn from_scale(scale: Vec3) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

    /// Convert to a 4x4 matrix (Scale -> Rotate -> Translate).
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

    /// Transform a direction vector (scale + rotation only, no translation).
    #[inline]
    pub fn transform_vector(self, vector: Vec3) -> Vec3 {
        let scaled = Vec3::new(
            vector.x * self.scale.x,
            vector.y * self.scale.y,
            vector.z * self.scale.z,
        );
        self.rotation * scaled
    }

    /// Compose two transforms.
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

    /// Compute inverse via matrix decomposition.
    pub fn inverse(self) -> Self {
        let mat = self.to_matrix().inverse();
        // Extract translation from column 3
        let translation = Vec3::new(mat.cols[3][0], mat.cols[3][1], mat.cols[3][2]);
        // Extract scale from column lengths
        let sx = Vec3::new(mat.cols[0][0], mat.cols[0][1], mat.cols[0][2]).length();
        let sy = Vec3::new(mat.cols[1][0], mat.cols[1][1], mat.cols[1][2]).length();
        let sz = Vec3::new(mat.cols[2][0], mat.cols[2][1], mat.cols[2][2]).length();
        let scale = Vec3::new(sx, sy, sz);
        // Extract rotation (divide columns by scale)
        let rot_mat = Mat4 {
            cols: [
                [
                    mat.cols[0][0] / sx,
                    mat.cols[0][1] / sx,
                    mat.cols[0][2] / sx,
                    0.0,
                ],
                [
                    mat.cols[1][0] / sy,
                    mat.cols[1][1] / sy,
                    mat.cols[1][2] / sy,
                    0.0,
                ],
                [
                    mat.cols[2][0] / sz,
                    mat.cols[2][1] / sz,
                    mat.cols[2][2] / sz,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        // Extract quaternion from rotation matrix
        let trace = rot_mat.cols[0][0] + rot_mat.cols[1][1] + rot_mat.cols[2][2];
        let rotation = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Quat::from_xyzw(
                (rot_mat.cols[1][2] - rot_mat.cols[2][1]) / s,
                (rot_mat.cols[2][0] - rot_mat.cols[0][2]) / s,
                (rot_mat.cols[0][1] - rot_mat.cols[1][0]) / s,
                0.25 * s,
            )
        } else {
            Quat::IDENTITY
        };

        Self {
            translation,
            rotation: rotation.normalize(),
            scale,
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
        let r = t.transform_point(p);
        assert!((r.x - p.x).abs() < 1e-6);
        assert!((r.y - p.y).abs() < 1e-6);
        assert!((r.z - p.z).abs() < 1e-6);
    }

    #[test]
    fn translation_only() {
        let t = Transform::from_translation(Vec3::new(10.0, 20.0, 30.0));
        let p = Vec3::new(1.0, 2.0, 3.0);
        let r = t.transform_point(p);
        assert_eq!(r, Vec3::new(11.0, 22.0, 33.0));
        // Vectors unaffected by translation
        let v = t.transform_vector(p);
        assert_eq!(v, p);
    }

    #[test]
    fn scale_only() {
        let t = Transform::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let r = t.transform_point(Vec3::ONE);
        assert_eq!(r, Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn rotation_only() {
        let t = Transform::from_rotation(Quat::from_axis_angle(Vec3::Z, FRAC_PI_2));
        let r = t.transform_point(Vec3::X);
        assert!(r.x.abs() < 1e-5);
        assert!((r.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn compose_transforms() {
        let a = Transform::from_translation(Vec3::new(5.0, 0.0, 0.0));
        let b = Transform::from_scale(Vec3::new(2.0, 2.0, 2.0));
        let composed = a.mul(b);
        let r = composed.transform_point(Vec3::X);
        assert!((r.x - 7.0).abs() < 1e-5);
    }

    #[test]
    fn matrix_consistency() {
        let t = Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_axis_angle(Vec3::Z, 0.5),
            scale: Vec3::new(2.0, 2.0, 2.0),
        };
        let mat = t.to_matrix();
        let p = Vec3::X;

        let from_transform = t.transform_point(p);
        let from_matrix = mat.transform_point3(p);

        assert!((from_transform.x - from_matrix.x).abs() < 1e-5);
        assert!((from_transform.y - from_matrix.y).abs() < 1e-5);
        assert!((from_transform.z - from_matrix.z).abs() < 1e-5);
    }

    #[test]
    fn inverse_roundtrip() {
        let t = Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_axis_angle(Vec3::Y, 0.7),
            scale: Vec3::new(2.0, 0.5, 3.0),
        };
        let p = Vec3::new(4.0, 5.0, 6.0);
        let forward = t.to_matrix();
        let inv = forward.inverse();
        let transformed = forward.transform_point3(p);
        let result = inv.transform_point3(transformed);
        assert!((result.x - p.x).abs() < 1e-4);
        assert!((result.y - p.y).abs() < 1e-4);
        assert!((result.z - p.z).abs() < 1e-4);
    }
}
