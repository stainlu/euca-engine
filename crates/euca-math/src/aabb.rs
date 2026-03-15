use serde::{Serialize, Deserialize};
use crate::Vec3;

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Create an AABB from min and max corners.
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Compute the AABB enclosing all given points.
    /// Returns `None` if the iterator is empty.
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Option<Self> {
        let mut iter = points.into_iter();
        let first = iter.next()?;
        let mut min = first;
        let mut max = first;
        for p in iter {
            min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
            max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
        }
        Some(Self { min, max })
    }

    /// Center point of the AABB.
    #[inline]
    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-extents (size from center to each face).
    #[inline]
    pub fn extents(self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Full size of the AABB.
    #[inline]
    pub fn size(self) -> Vec3 {
        self.max - self.min
    }

    /// Check if a point is inside the AABB.
    #[inline]
    pub fn contains_point(self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Check if two AABBs overlap.
    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Return the union of two AABBs (smallest AABB enclosing both).
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_point() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(aabb.contains_point(Vec3::ZERO));
        assert!(aabb.contains_point(Vec3::new(1.0, 1.0, 1.0)));
        assert!(!aabb.contains_point(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn intersects() {
        let a = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let b = Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        let c = Aabb::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));

        assert!(a.intersects(b));
        assert!(!a.intersects(c));
    }

    #[test]
    fn from_points() {
        let points = vec![
            Vec3::new(1.0, -2.0, 3.0),
            Vec3::new(-1.0, 4.0, 0.0),
            Vec3::new(2.0, 0.0, -1.0),
        ];
        let aabb = Aabb::from_points(points).unwrap();
        assert_eq!(aabb.min, Vec3::new(-1.0, -2.0, -1.0));
        assert_eq!(aabb.max, Vec3::new(2.0, 4.0, 3.0));
    }

    #[test]
    fn center_and_extents() {
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 6.0, 8.0));
        assert_eq!(aabb.center(), Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(aabb.extents(), Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(aabb.size(), Vec3::new(4.0, 6.0, 8.0));
    }

    #[test]
    fn from_empty_points() {
        let result = Aabb::from_points(std::iter::empty());
        assert!(result.is_none());
    }
}
