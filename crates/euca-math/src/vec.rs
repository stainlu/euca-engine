use crate::{cfg_scalar, cfg_simd};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// 2D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

/// 3D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 4D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

// ── SIMD load/store helpers ─────────────────────────────────────────────────
//
// Vec3 and Vec4 are `#[repr(C, align(16))]`, so they can be loaded directly
// into a 128-bit SIMD register. We load Vec3 with w=0.0 to avoid polluting
// the dot product.

cfg_simd! {
    use crate::simd::f32x4;
}

cfg_simd! {
    impl Vec3 {
        /// Load into a SIMD register with w=0.
        #[inline(always)]
        fn load(self) -> f32x4 {
            f32x4::new(self.x, self.y, self.z, 0.0)
        }

        /// Store the xyz lanes of a SIMD register back into a Vec3.
        #[inline(always)]
        fn from_simd(v: f32x4) -> Self {
            Self { x: v.x(), y: v.y(), z: v.z() }
        }
    }

    impl Vec4 {
        /// Load into a SIMD register.
        #[inline(always)]
        pub(crate) fn load(self) -> f32x4 {
            f32x4::new(self.x, self.y, self.z, self.w)
        }

        /// Store all four lanes of a SIMD register back into a Vec4.
        #[inline(always)]
        pub(crate) fn from_simd(v: f32x4) -> Self {
            Self { x: v.x(), y: v.y(), z: v.z(), w: v.w() }
        }
    }
}

// ── Vec2 ──

impl Vec2 {
    /// All zeros.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    /// All ones.
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };
    /// Unit vector along the X axis.
    pub const X: Self = Self { x: 1.0, y: 0.0 };
    /// Unit vector along the Y axis.
    pub const Y: Self = Self { x: 0.0, y: 1.0 };

    /// Creates a new `Vec2` from x and y components.
    #[inline(always)]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Returns the dot product of two vectors.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// Returns the Euclidean length of the vector.
    #[inline(always)]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns a unit-length vector in the same direction.
    #[inline(always)]
    pub fn normalize(self) -> Self {
        let inv = 1.0 / self.length();
        Self {
            x: self.x * inv,
            y: self.y * inv,
        }
    }

    /// Returns the Euclidean distance between two points.
    #[inline(always)]
    pub fn distance(self, rhs: Self) -> f32 {
        (self - rhs).length()
    }
}

impl Add for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Neg for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

// ── Vec3 ──

impl Vec3 {
    /// All zeros.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    /// All ones.
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    /// Unit vector along the X axis.
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    /// Unit vector along the Y axis.
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    /// Unit vector along the Z axis.
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// Creates a new `Vec3` from x, y, and z components.
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Returns the Euclidean length of the vector.
    #[inline(always)]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns the squared length (avoids a square root).
    #[inline(always)]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Returns the Euclidean distance between two points.
    #[inline(always)]
    pub fn distance(self, rhs: Self) -> f32 {
        (self - rhs).length()
    }

    /// Linearly interpolates between `self` and `rhs` by factor `t`.
    #[inline(always)]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        self + (rhs - self) * t
    }

    /// Find the parameter t on a line (line_origin + line_dir * t) at the point
    /// closest to a ray (ray_origin + ray_dir * s). Returns t.
    /// Used for projecting mouse rays onto gizmo axes.
    pub fn closest_line_param(
        line_origin: Vec3,
        line_dir: Vec3,
        ray_origin: Vec3,
        ray_dir: Vec3,
    ) -> f32 {
        let w0 = line_origin - ray_origin;
        let a = line_dir.dot(line_dir);
        let b = line_dir.dot(ray_dir);
        let c = ray_dir.dot(ray_dir);
        let d = line_dir.dot(w0);
        let e = ray_dir.dot(w0);
        let denom = a * c - b * b;
        if denom.abs() < 1e-10 {
            return 0.0; // parallel lines
        }
        (b * e - c * d) / denom
    }
}

// ── Vec3: SIMD-accelerated methods ──────────────────────────────────────────

cfg_simd! {
    impl Vec3 {
        /// Returns the dot product of two vectors.
        #[inline(always)]
        pub fn dot(self, rhs: Self) -> f32 {
            self.load().mul(rhs.load()).horizontal_sum()
        }

        /// Returns the cross product of two vectors.
        #[inline(always)]
        pub fn cross(self, rhs: Self) -> Self {
            // cross(a, b) = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
            let a_yzx = f32x4::new(self.y, self.z, self.x, 0.0);
            let b_zxy = f32x4::new(rhs.z, rhs.x, rhs.y, 0.0);
            let a_zxy = f32x4::new(self.z, self.x, self.y, 0.0);
            let b_yzx = f32x4::new(rhs.y, rhs.z, rhs.x, 0.0);
            Self::from_simd(a_yzx.mul(b_zxy).sub(a_zxy.mul(b_yzx)))
        }

        /// Returns a unit-length vector in the same direction.
        #[inline(always)]
        pub fn normalize(self) -> Self {
            let v = self.load();
            let inv_len = f32x4::splat(1.0 / v.mul(v).horizontal_sum().sqrt());
            Self::from_simd(v.mul(inv_len))
        }

        /// Returns the component-wise minimum of two vectors.
        #[inline(always)]
        pub fn min(self, rhs: Self) -> Self {
            Self::from_simd(self.load().min(rhs.load()))
        }

        /// Returns the component-wise maximum of two vectors.
        #[inline(always)]
        pub fn max(self, rhs: Self) -> Self {
            Self::from_simd(self.load().max(rhs.load()))
        }
    }

    impl Add for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self { Self::from_simd(self.load().add(rhs.load())) }
    }

    impl Sub for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self { Self::from_simd(self.load().sub(rhs.load())) }
    }

    impl Mul<f32> for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: f32) -> Self { Self::from_simd(self.load().mul(f32x4::splat(rhs))) }
    }

    impl Div<f32> for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: f32) -> Self {
            Self::from_simd(self.load().mul(f32x4::splat(1.0 / rhs)))
        }
    }

    impl Neg for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self { Self::from_simd(self.load().neg()) }
    }
}

// ── Vec3: scalar fallback methods ───────────────────────────────────────────

cfg_scalar! {
    impl Vec3 {
        /// Returns the dot product of two vectors.
        #[inline(always)]
        pub fn dot(self, rhs: Self) -> f32 {
            self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
        }

        /// Returns the cross product of two vectors.
        #[inline(always)]
        pub fn cross(self, rhs: Self) -> Self {
            Self {
                x: self.y * rhs.z - self.z * rhs.y,
                y: self.z * rhs.x - self.x * rhs.z,
                z: self.x * rhs.y - self.y * rhs.x,
            }
        }

        /// Returns a unit-length vector in the same direction.
        #[inline(always)]
        pub fn normalize(self) -> Self {
            let inv = 1.0 / self.length();
            Self { x: self.x * inv, y: self.y * inv, z: self.z * inv }
        }

        /// Returns the component-wise minimum of two vectors.
        #[inline(always)]
        pub fn min(self, rhs: Self) -> Self {
            Self { x: self.x.min(rhs.x), y: self.y.min(rhs.y), z: self.z.min(rhs.z) }
        }

        /// Returns the component-wise maximum of two vectors.
        #[inline(always)]
        pub fn max(self, rhs: Self) -> Self {
            Self { x: self.x.max(rhs.x), y: self.y.max(rhs.y), z: self.z.max(rhs.z) }
        }
    }

    impl Add for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
        }
    }

    impl Sub for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
        }
    }

    impl Mul<f32> for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: f32) -> Self {
            Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs }
        }
    }

    impl Div<f32> for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: f32) -> Self {
            let inv = 1.0 / rhs;
            Self { x: self.x * inv, y: self.y * inv, z: self.z * inv }
        }
    }

    impl Neg for Vec3 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
    }
}

// ── Vec4 ──

impl Vec4 {
    /// All zeros.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };
    /// All ones.
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
        w: 1.0,
    };

    /// Creates a new `Vec4` from x, y, z, and w components.
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Returns the Euclidean length of the vector.
    #[inline(always)]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
}

// ── Vec4: SIMD-accelerated methods ──────────────────────────────────────────

cfg_simd! {
    impl Vec4 {
        /// Returns the dot product of two vectors.
        #[inline(always)]
        pub fn dot(self, rhs: Self) -> f32 {
            self.load().mul(rhs.load()).horizontal_sum()
        }

        /// Returns a unit-length vector in the same direction.
        #[inline(always)]
        pub fn normalize(self) -> Self {
            let v = self.load();
            let inv_len = f32x4::splat(1.0 / v.mul(v).horizontal_sum().sqrt());
            Self::from_simd(v.mul(inv_len))
        }
    }

    impl Add for Vec4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self { Self::from_simd(self.load().add(rhs.load())) }
    }

    impl Sub for Vec4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self { Self::from_simd(self.load().sub(rhs.load())) }
    }

    impl Mul<f32> for Vec4 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: f32) -> Self { Self::from_simd(self.load().mul(f32x4::splat(rhs))) }
    }

    impl Neg for Vec4 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self { Self::from_simd(self.load().neg()) }
    }
}

// ── Vec4: scalar fallback methods ───────────────────────────────────────────

cfg_scalar! {
    impl Vec4 {
        /// Returns the dot product of two vectors.
        #[inline(always)]
        pub fn dot(self, rhs: Self) -> f32 {
            self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
        }

        /// Returns a unit-length vector in the same direction.
        #[inline(always)]
        pub fn normalize(self) -> Self {
            let inv = 1.0 / self.length();
            Self { x: self.x * inv, y: self.y * inv, z: self.z * inv, w: self.w * inv }
        }
    }

    impl Add for Vec4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, w: self.w + rhs.w }
        }
    }

    impl Sub for Vec4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, w: self.w - rhs.w }
        }
    }

    impl Mul<f32> for Vec4 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: f32) -> Self {
            Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs, w: self.w * rhs }
        }
    }

    impl Neg for Vec4 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z, w: -self.w } }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_basic_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b - a, Vec3::new(3.0, 3.0, 3.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn vec3_dot_cross() {
        let a = Vec3::X;
        let b = Vec3::Y;
        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.cross(b), Vec3::Z);
    }

    #[test]
    fn vec3_length_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < 1e-6);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec2_distance() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);
        assert!((a.distance(b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn vec3_lerp() {
        let a = Vec3::ZERO;
        let b = Vec3::new(10.0, 20.0, 30.0);
        let mid = a.lerp(b, 0.5);
        assert_eq!(mid, Vec3::new(5.0, 10.0, 15.0));
    }

    #[test]
    fn vec3_div() {
        let v = Vec3::new(10.0, 20.0, 30.0);
        let result = v / 10.0;
        assert!((result.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec4_dot() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a.dot(b), 70.0);
    }

    #[test]
    fn closest_line_param_perpendicular() {
        let t = Vec3::closest_line_param(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::X,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::Y,
        );
        assert!((t - 2.0).abs() < 1e-6, "Expected t=2.0, got {t}");
    }

    #[test]
    fn closest_line_param_parallel() {
        let t = Vec3::closest_line_param(Vec3::ZERO, Vec3::X, Vec3::new(0.0, 1.0, 0.0), Vec3::X);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn vec4_normalize() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec4_add_sub() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a + b, Vec4::new(6.0, 8.0, 10.0, 12.0));
        assert_eq!(b - a, Vec4::new(4.0, 4.0, 4.0, 4.0));
    }

    #[test]
    fn vec4_mul_neg() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a * 2.0, Vec4::new(2.0, 4.0, 6.0, 8.0));
        assert_eq!(-a, Vec4::new(-1.0, -2.0, -3.0, -4.0));
    }

    #[test]
    fn vec3_min_max() {
        let a = Vec3::new(1.0, 5.0, 3.0);
        let b = Vec3::new(4.0, 2.0, 6.0);
        assert_eq!(a.min(b), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(a.max(b), Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn vec3_cross_identity() {
        assert_eq!(Vec3::X.cross(Vec3::Y), Vec3::Z);
        assert_eq!(Vec3::Y.cross(Vec3::Z), Vec3::X);
        assert_eq!(Vec3::Z.cross(Vec3::X), Vec3::Y);
    }

    #[test]
    fn vec3_cross_anticommutative() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let ab = a.cross(b);
        let ba = b.cross(a);
        assert!((ab.x + ba.x).abs() < 1e-6);
        assert!((ab.y + ba.y).abs() < 1e-6);
        assert!((ab.z + ba.z).abs() < 1e-6);
    }
}
