use euca_math::Vec3;

use crate::components::ColliderShape;

/// A pair of colliding entities with contact info.
#[derive(Clone, Debug)]
pub struct CollisionPair {
    pub entity_a: euca_ecs::Entity,
    pub entity_b: euca_ecs::Entity,
    /// Contact normal (from A to B).
    pub normal: Vec3,
    /// Penetration depth (positive = overlapping).
    pub depth: f32,
}

/// Test if two AABBs overlap given their centers and half-extents.
#[allow(clippy::too_many_arguments)]
pub fn intersect_aabb(
    pos_a: Vec3,
    hx_a: f32,
    hy_a: f32,
    hz_a: f32,
    pos_b: Vec3,
    hx_b: f32,
    hy_b: f32,
    hz_b: f32,
) -> Option<(Vec3, f32)> {
    let dx = (pos_b.x - pos_a.x).abs() - (hx_a + hx_b);
    let dy = (pos_b.y - pos_a.y).abs() - (hy_a + hy_b);
    let dz = (pos_b.z - pos_a.z).abs() - (hz_a + hz_b);

    if dx > 0.0 || dy > 0.0 || dz > 0.0 {
        return None; // No overlap
    }

    // Find the axis of minimum penetration
    let overlap_x = -dx;
    let overlap_y = -dy;
    let overlap_z = -dz;

    if overlap_x <= overlap_y && overlap_x <= overlap_z {
        let sign = if pos_b.x > pos_a.x { 1.0 } else { -1.0 };
        Some((Vec3::new(sign, 0.0, 0.0), overlap_x))
    } else if overlap_y <= overlap_z {
        let sign = if pos_b.y > pos_a.y { 1.0 } else { -1.0 };
        Some((Vec3::new(0.0, sign, 0.0), overlap_y))
    } else {
        let sign = if pos_b.z > pos_a.z { 1.0 } else { -1.0 };
        Some((Vec3::new(0.0, 0.0, sign), overlap_z))
    }
}

/// Test if two spheres overlap.
pub fn intersect_spheres(
    pos_a: Vec3,
    radius_a: f32,
    pos_b: Vec3,
    radius_b: f32,
) -> Option<(Vec3, f32)> {
    let diff = pos_b - pos_a;
    let dist_sq = diff.length_squared();
    let sum_r = radius_a + radius_b;

    if dist_sq >= sum_r * sum_r {
        return None;
    }

    let dist = dist_sq.sqrt();
    if dist < 1e-6 {
        return Some((Vec3::Y, sum_r)); // Degenerate: same position
    }

    let normal = diff * (1.0 / dist);
    let depth = sum_r - dist;
    Some((normal, depth))
}

/// Test AABB vs Sphere overlap.
pub fn intersect_aabb_sphere(
    aabb_pos: Vec3,
    hx: f32,
    hy: f32,
    hz: f32,
    sphere_pos: Vec3,
    radius: f32,
) -> Option<(Vec3, f32)> {
    // Find closest point on AABB to sphere center
    let closest = Vec3::new(
        sphere_pos.x.clamp(aabb_pos.x - hx, aabb_pos.x + hx),
        sphere_pos.y.clamp(aabb_pos.y - hy, aabb_pos.y + hy),
        sphere_pos.z.clamp(aabb_pos.z - hz, aabb_pos.z + hz),
    );

    let diff = sphere_pos - closest;
    let dist_sq = diff.length_squared();

    if dist_sq >= radius * radius {
        return None;
    }

    let dist = dist_sq.sqrt();
    if dist < 1e-6 {
        // Sphere center is inside AABB
        return Some((Vec3::Y, radius));
    }

    let normal = diff * (1.0 / dist);
    let depth = radius - dist;
    Some((normal, depth))
}

/// Test two collider shapes for overlap.
pub fn intersect_shapes(
    pos_a: Vec3,
    shape_a: &ColliderShape,
    pos_b: Vec3,
    shape_b: &ColliderShape,
) -> Option<(Vec3, f32)> {
    match (shape_a, shape_b) {
        (
            ColliderShape::Aabb {
                hx: hxa,
                hy: hya,
                hz: hza,
            },
            ColliderShape::Aabb {
                hx: hxb,
                hy: hyb,
                hz: hzb,
            },
        ) => intersect_aabb(pos_a, *hxa, *hya, *hza, pos_b, *hxb, *hyb, *hzb),
        (ColliderShape::Sphere { radius: ra }, ColliderShape::Sphere { radius: rb }) => {
            intersect_spheres(pos_a, *ra, pos_b, *rb)
        }
        (ColliderShape::Aabb { hx, hy, hz }, ColliderShape::Sphere { radius }) => {
            intersect_aabb_sphere(pos_a, *hx, *hy, *hz, pos_b, *radius)
        }
        (ColliderShape::Sphere { radius }, ColliderShape::Aabb { hx, hy, hz }) => {
            intersect_aabb_sphere(pos_b, *hx, *hy, *hz, pos_a, *radius).map(|(n, d)| (-n, d))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_overlap() {
        let r = intersect_aabb(
            Vec3::ZERO,
            1.0,
            1.0,
            1.0,
            Vec3::new(1.5, 0.0, 0.0),
            1.0,
            1.0,
            1.0,
        );
        assert!(r.is_some());
        let (normal, depth) = r.unwrap();
        assert!((normal.x - 1.0).abs() < 1e-6);
        assert!(depth > 0.0);
    }

    #[test]
    fn aabb_no_overlap() {
        let r = intersect_aabb(
            Vec3::ZERO,
            1.0,
            1.0,
            1.0,
            Vec3::new(5.0, 0.0, 0.0),
            1.0,
            1.0,
            1.0,
        );
        assert!(r.is_none());
    }

    #[test]
    fn sphere_overlap() {
        let r = intersect_spheres(Vec3::ZERO, 1.0, Vec3::new(1.5, 0.0, 0.0), 1.0);
        assert!(r.is_some());
        let (_, depth) = r.unwrap();
        assert!((depth - 0.5).abs() < 1e-5);
    }

    #[test]
    fn sphere_no_overlap() {
        let r = intersect_spheres(Vec3::ZERO, 1.0, Vec3::new(5.0, 0.0, 0.0), 1.0);
        assert!(r.is_none());
    }

    #[test]
    fn aabb_sphere_overlap() {
        let r = intersect_aabb_sphere(Vec3::ZERO, 1.0, 1.0, 1.0, Vec3::new(1.5, 0.0, 0.0), 1.0);
        assert!(r.is_some());
    }
}
