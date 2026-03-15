use serde::{Deserialize, Serialize};
use std::ops::{Add, Deref, DerefMut, Mul, Neg, Sub};

/// 2D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Vec2(pub glam::Vec2);

/// 3D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Vec3(pub glam::Vec3);

/// 4D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Vec4(pub glam::Vec4);

// ── Vec2 ──

impl Vec2 {
    pub const ZERO: Self = Self(glam::Vec2::ZERO);
    pub const ONE: Self = Self(glam::Vec2::ONE);
    pub const X: Self = Self(glam::Vec2::X);
    pub const Y: Self = Self(glam::Vec2::Y);

    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self(glam::Vec2::new(x, y))
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.0.dot(rhs.0)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.0.length()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        Self(self.0.normalize())
    }

    #[inline]
    pub fn distance(self, rhs: Self) -> f32 {
        self.0.distance(rhs.0)
    }
}

impl Deref for Vec2 {
    type Target = glam::Vec2;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Vec2 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self(self.0 * rhs)
    }
}

impl Neg for Vec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl From<glam::Vec2> for Vec2 {
    #[inline]
    fn from(v: glam::Vec2) -> Self {
        Self(v)
    }
}

impl From<Vec2> for glam::Vec2 {
    #[inline]
    fn from(v: Vec2) -> Self {
        v.0
    }
}

// ── Vec3 ──

impl Vec3 {
    pub const ZERO: Self = Self(glam::Vec3::ZERO);
    pub const ONE: Self = Self(glam::Vec3::ONE);
    pub const X: Self = Self(glam::Vec3::X);
    pub const Y: Self = Self(glam::Vec3::Y);
    pub const Z: Self = Self(glam::Vec3::Z);
    pub const UP: Self = Self(glam::Vec3::Z); // Z-up like UE
    pub const FORWARD: Self = Self(glam::Vec3::X);
    pub const RIGHT: Self = Self(glam::Vec3::Y);

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self(glam::Vec3::new(x, y, z))
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.0.dot(rhs.0)
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self(self.0.cross(rhs.0))
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.0.length()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        Self(self.0.normalize())
    }

    #[inline]
    pub fn distance(self, rhs: Self) -> f32 {
        self.0.distance(rhs.0)
    }
}

impl Deref for Vec3 {
    type Target = glam::Vec3;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Vec3 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self(self.0 * rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl From<glam::Vec3> for Vec3 {
    #[inline]
    fn from(v: glam::Vec3) -> Self {
        Self(v)
    }
}

impl From<Vec3> for glam::Vec3 {
    #[inline]
    fn from(v: Vec3) -> Self {
        v.0
    }
}

// ── Vec4 ──

impl Vec4 {
    pub const ZERO: Self = Self(glam::Vec4::ZERO);
    pub const ONE: Self = Self(glam::Vec4::ONE);

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(glam::Vec4::new(x, y, z, w))
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.0.dot(rhs.0)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.0.length()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        Self(self.0.normalize())
    }
}

impl Deref for Vec4 {
    type Target = glam::Vec4;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Vec4 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Add for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self(self.0 * rhs)
    }
}

impl Neg for Vec4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl From<glam::Vec4> for Vec4 {
    #[inline]
    fn from(v: glam::Vec4) -> Self {
        Self(v)
    }
}

impl From<Vec4> for glam::Vec4 {
    #[inline]
    fn from(v: Vec4) -> Self {
        v.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_basic_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);

        let sum = a + b;
        assert_eq!(sum, Vec3::new(5.0, 7.0, 9.0));

        let diff = b - a;
        assert_eq!(diff, Vec3::new(3.0, 3.0, 3.0));

        let scaled = a * 2.0;
        assert_eq!(scaled, Vec3::new(2.0, 4.0, 6.0));

        assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn vec3_dot_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);

        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.cross(b), Vec3::new(0.0, 0.0, 1.0));
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
}
