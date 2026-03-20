use crate::{Quat, Vec3, Vec4, cfg_scalar, cfg_simd};
use serde::{Deserialize, Serialize};
use std::ops::Mul;

/// 4x4 column-major matrix.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct Mat4 {
    /// Column 0 (x-axis)
    pub cols: [[f32; 4]; 4],
}

impl Default for Mat4 {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ── SIMD helpers ────────────────────────────────────────────────────────────

cfg_simd! {
    use crate::simd::f32x4;

    impl Mat4 {
        /// Load column `i` as a SIMD register.
        #[inline(always)]
        fn load_col(&self, i: usize) -> f32x4 {
            let c = &self.cols[i];
            f32x4::new(c[0], c[1], c[2], c[3])
        }

        /// Store a SIMD register into column array.
        #[inline(always)]
        fn store_col(v: f32x4) -> [f32; 4] {
            [v.x(), v.y(), v.z(), v.w()]
        }

        /// Multiply a column vector by this matrix using SIMD: result = M * v.
        /// Computes: col0 * v.x + col1 * v.y + col2 * v.z + col3 * v.w
        #[inline(always)]
        fn mul_col(&self, v: f32x4) -> f32x4 {
            let c0 = self.load_col(0);
            let c1 = self.load_col(1);
            let c2 = self.load_col(2);
            let c3 = self.load_col(3);
            c0.mul(v.splat_x())
                .add(c1.mul(v.splat_y()))
                .add(c2.mul(v.splat_z()))
                .add(c3.mul(v.splat_w()))
        }
    }
}

impl Mat4 {
    /// The identity matrix.
    pub const IDENTITY: Self = Self {
        cols: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// All zeros.
    pub const ZERO: Self = Self {
        cols: [[0.0; 4]; 4],
    };

    /// Returns column `i` as a 4-element array.
    #[inline(always)]
    pub fn col(&self, i: usize) -> [f32; 4] {
        self.cols[i]
    }

    /// Column-major element access: col i, row j.
    #[inline(always)]
    pub fn get(&self, col: usize, row: usize) -> f32 {
        self.cols[col][row]
    }

    /// Creates a translation matrix from a 3D offset.
    pub fn from_translation(t: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        m.cols[3][0] = t.x;
        m.cols[3][1] = t.y;
        m.cols[3][2] = t.z;
        m
    }

    /// Creates a non-uniform scale matrix.
    pub fn from_scale(s: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        m.cols[0][0] = s.x;
        m.cols[1][1] = s.y;
        m.cols[2][2] = s.z;
        m
    }

    /// Creates a rotation matrix from a unit quaternion.
    pub fn from_rotation(q: Quat) -> Self {
        let x2 = q.x + q.x;
        let y2 = q.y + q.y;
        let z2 = q.z + q.z;
        let xx = q.x * x2;
        let xy = q.x * y2;
        let xz = q.x * z2;
        let yy = q.y * y2;
        let yz = q.y * z2;
        let zz = q.z * z2;
        let wx = q.w * x2;
        let wy = q.w * y2;
        let wz = q.w * z2;

        Self {
            cols: [
                [1.0 - (yy + zz), xy + wz, xz - wy, 0.0],
                [xy - wz, 1.0 - (xx + zz), yz + wx, 0.0],
                [xz + wy, yz - wx, 1.0 - (xx + yy), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Creates a combined scale-rotation-translation matrix.
    pub fn from_scale_rotation_translation(s: Vec3, r: Quat, t: Vec3) -> Self {
        let rot = Self::from_rotation(r);
        Self {
            cols: [
                [
                    rot.cols[0][0] * s.x,
                    rot.cols[0][1] * s.x,
                    rot.cols[0][2] * s.x,
                    0.0,
                ],
                [
                    rot.cols[1][0] * s.y,
                    rot.cols[1][1] * s.y,
                    rot.cols[1][2] * s.y,
                    0.0,
                ],
                [
                    rot.cols[2][0] * s.z,
                    rot.cols[2][1] * s.z,
                    rot.cols[2][2] * s.z,
                    0.0,
                ],
                [t.x, t.y, t.z, 1.0],
            ],
        }
    }

    /// Orthographic projection (left-handed, depth 0..1).
    pub fn orthographic_lh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        let rml = right - left;
        let tmb = top - bottom;
        let fmn = z_far - z_near;
        Self {
            cols: [
                [2.0 / rml, 0.0, 0.0, 0.0],
                [0.0, 2.0 / tmb, 0.0, 0.0],
                [0.0, 0.0, 1.0 / fmn, 0.0],
                [
                    -(right + left) / rml,
                    -(top + bottom) / tmb,
                    -z_near / fmn,
                    1.0,
                ],
            ],
        }
    }

    /// Perspective projection (left-handed, depth 0..1).
    pub fn perspective_lh(fov_y_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Self {
        let h = 1.0 / (fov_y_radians * 0.5).tan();
        let w = h / aspect;
        let r = z_far / (z_far - z_near);

        Self {
            cols: [
                [w, 0.0, 0.0, 0.0],
                [0.0, h, 0.0, 0.0],
                [0.0, 0.0, r, 1.0],
                [0.0, 0.0, -r * z_near, 0.0],
            ],
        }
    }

    /// Left-handed look-at view matrix.
    pub fn look_at_lh(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = (target - eye).normalize();
        let s = up.cross(f).normalize();
        let u = f.cross(s);

        Self {
            cols: [
                [s.x, u.x, f.x, 0.0],
                [s.y, u.y, f.y, 0.0],
                [s.z, u.z, f.z, 0.0],
                [-s.dot(eye), -u.dot(eye), -f.dot(eye), 1.0],
            ],
        }
    }

    /// Computes the matrix inverse via cofactor expansion.
    pub fn inverse(self) -> Self {
        let m = &self.cols;
        let a2323 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        let a1323 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
        let a1223 = m[1][2] * m[2][3] - m[2][2] * m[1][3];
        let a0323 = m[0][2] * m[3][3] - m[3][2] * m[0][3];
        let a0223 = m[0][2] * m[2][3] - m[2][2] * m[0][3];
        let a0123 = m[0][2] * m[1][3] - m[1][2] * m[0][3];
        let a2313 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        let a1313 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
        let a1213 = m[1][1] * m[2][3] - m[2][1] * m[1][3];
        let a2312 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        let a1312 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
        let a1212 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
        let a0313 = m[0][1] * m[3][3] - m[3][1] * m[0][3];
        let a0213 = m[0][1] * m[2][3] - m[2][1] * m[0][3];
        let a0312 = m[0][1] * m[3][2] - m[3][1] * m[0][2];
        let a0212 = m[0][1] * m[2][2] - m[2][1] * m[0][2];
        let a0113 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
        let a0112 = m[0][1] * m[1][2] - m[1][1] * m[0][2];

        let det = m[0][0] * (m[1][1] * a2323 - m[2][1] * a1323 + m[3][1] * a1223)
            - m[1][0] * (m[0][1] * a2323 - m[2][1] * a0323 + m[3][1] * a0223)
            + m[2][0] * (m[0][1] * a1323 - m[1][1] * a0323 + m[3][1] * a0123)
            - m[3][0] * (m[0][1] * a1223 - m[1][1] * a0223 + m[2][1] * a0123);

        let inv_det = 1.0 / det;

        Self {
            cols: [
                [
                    inv_det * (m[1][1] * a2323 - m[2][1] * a1323 + m[3][1] * a1223),
                    inv_det * -(m[0][1] * a2323 - m[2][1] * a0323 + m[3][1] * a0223),
                    inv_det * (m[0][1] * a1323 - m[1][1] * a0323 + m[3][1] * a0123),
                    inv_det * -(m[0][1] * a1223 - m[1][1] * a0223 + m[2][1] * a0123),
                ],
                [
                    inv_det * -(m[1][0] * a2323 - m[2][0] * a1323 + m[3][0] * a1223),
                    inv_det * (m[0][0] * a2323 - m[2][0] * a0323 + m[3][0] * a0223),
                    inv_det * -(m[0][0] * a1323 - m[1][0] * a0323 + m[3][0] * a0123),
                    inv_det * (m[0][0] * a1223 - m[1][0] * a0223 + m[2][0] * a0123),
                ],
                [
                    inv_det * (m[1][0] * a2313 - m[2][0] * a1313 + m[3][0] * a1213),
                    inv_det * -(m[0][0] * a2313 - m[2][0] * a0313 + m[3][0] * a0213),
                    inv_det * (m[0][0] * a1313 - m[1][0] * a0313 + m[3][0] * a0113),
                    inv_det * -(m[0][0] * a1213 - m[1][0] * a0213 + m[2][0] * a0113),
                ],
                [
                    inv_det * -(m[1][0] * a2312 - m[2][0] * a1312 + m[3][0] * a1212),
                    inv_det * (m[0][0] * a2312 - m[2][0] * a0312 + m[3][0] * a0212),
                    inv_det * -(m[0][0] * a1312 - m[1][0] * a0312 + m[3][0] * a0112),
                    inv_det * (m[0][0] * a1212 - m[1][0] * a0212 + m[2][0] * a0112),
                ],
            ],
        }
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(self) -> Self {
        let m = &self.cols;
        Self {
            cols: [
                [m[0][0], m[1][0], m[2][0], m[3][0]],
                [m[0][1], m[1][1], m[2][1], m[3][1]],
                [m[0][2], m[1][2], m[2][2], m[3][2]],
                [m[0][3], m[1][3], m[2][3], m[3][3]],
            ],
        }
    }

    /// Convert to column-major 2D array (for GPU upload).
    #[inline(always)]
    pub fn to_cols_array_2d(&self) -> [[f32; 4]; 4] {
        self.cols
    }

    /// Create from a column-major 2D array.
    #[inline(always)]
    pub fn from_cols_array_2d(cols: &[[f32; 4]; 4]) -> Self {
        Self { cols: *cols }
    }
}

// ── SIMD: transform_point3, Mat4*Mat4, Mat4*Vec4 ────────────────────────────

cfg_simd! {
    impl Mat4 {
        /// Transform a point (w=1, applies translation).
        pub fn transform_point3(&self, p: Vec3) -> Vec3 {
            let v = f32x4::new(p.x, p.y, p.z, 1.0);
            let result = self.mul_col(v);
            Vec3::new(result.x(), result.y(), result.z())
        }
    }

    impl Mul for Mat4 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self {
                cols: [
                    Self::store_col(self.mul_col(rhs.load_col(0))),
                    Self::store_col(self.mul_col(rhs.load_col(1))),
                    Self::store_col(self.mul_col(rhs.load_col(2))),
                    Self::store_col(self.mul_col(rhs.load_col(3))),
                ],
            }
        }
    }

    impl Mul<Vec4> for Mat4 {
        type Output = Vec4;
        fn mul(self, v: Vec4) -> Vec4 {
            Vec4::from_simd(self.mul_col(v.load()))
        }
    }
}

// ── Scalar fallback: transform_point3, Mat4*Mat4, Mat4*Vec4 ─────────────────

cfg_scalar! {
    impl Mat4 {
        /// Transform a point (w=1, applies translation).
        pub fn transform_point3(&self, p: Vec3) -> Vec3 {
            let m = &self.cols;
            Vec3::new(
                m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0],
                m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1],
                m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z + m[3][2],
            )
        }
    }

    impl Mul for Mat4 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            let a = &self.cols;
            let b = &rhs.cols;
            let mut out = [[0.0f32; 4]; 4];
            for c in 0..4 {
                for r in 0..4 {
                    out[c][r] = a[0][r] * b[c][0] + a[1][r] * b[c][1]
                        + a[2][r] * b[c][2] + a[3][r] * b[c][3];
                }
            }
            Self { cols: out }
        }
    }

    impl Mul<Vec4> for Mat4 {
        type Output = Vec4;
        fn mul(self, v: Vec4) -> Vec4 {
            let m = &self.cols;
            Vec4::new(
                m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w,
                m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w,
                m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w,
                m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3] * v.w,
            )
        }
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
        for c in 0..4 {
            for r in 0..4 {
                let expected = if c == r { 1.0 } else { 0.0 };
                assert!(
                    (result.cols[c][r] - expected).abs() < 1e-4,
                    "M*M^-1 [{c}][{r}] = {} (expected {expected})",
                    result.cols[c][r]
                );
            }
        }
    }

    #[test]
    fn transform_point() {
        let m = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let p = Vec3::new(1.0, 2.0, 3.0);
        let result = m.transform_point3(p);
        assert_eq!(result, Vec3::new(11.0, 2.0, 3.0));
    }

    #[test]
    fn mat4_mul_vec4() {
        let m = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
        let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let result = m * v;
        assert!((result.x - 11.0).abs() < 1e-6);
        assert!((result.y - 22.0).abs() < 1e-6);
        assert!((result.z - 33.0).abs() < 1e-6);
        assert!((result.w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mat4_mul_associative() {
        let a = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let b = Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let c = Mat4::from_rotation(Quat::from_axis_angle(Vec3::Y, 0.5));

        let ab_c = (a * b) * c;
        let a_bc = a * (b * c);

        for col in 0..4 {
            for row in 0..4 {
                assert!(
                    (ab_c.cols[col][row] - a_bc.cols[col][row]).abs() < 1e-4,
                    "Associativity failed at [{col}][{row}]"
                );
            }
        }
    }
}
