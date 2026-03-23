use euca_math::Vec3;

use crate::components::ColliderShape;

/// A pair of colliding entities with contact info.
#[derive(Clone, Debug)]
pub struct CollisionPair {
    /// First colliding entity.
    pub entity_a: euca_ecs::Entity,
    /// Second colliding entity.
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

// ── Capsule collision helpers ──

/// A capsule's line segment endpoints (the spine, excluding hemisphere radius).
fn capsule_segment(pos: Vec3, half_height: f32) -> (Vec3, Vec3) {
    (
        Vec3::new(pos.x, pos.y - half_height, pos.z),
        Vec3::new(pos.x, pos.y + half_height, pos.z),
    )
}

/// Closest point on line segment AB to point P.
fn closest_point_on_segment(a: Vec3, b: Vec3, p: Vec3) -> Vec3 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-12 {
        return a; // Degenerate segment
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + ab * t
}

/// Closest points between two line segments AB and CD.
/// Returns (closest_on_AB, closest_on_CD).
fn closest_points_segments(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> (Vec3, Vec3) {
    let ab = b - a;
    let cd = d - c;
    let ac = c - a;

    let d1 = ab.dot(ab);
    let d2 = ab.dot(cd);
    let d3 = cd.dot(cd);
    let d4 = ab.dot(ac);
    let d5 = cd.dot(ac);

    let denom = d1 * d3 - d2 * d2;

    let (s, t);
    if denom.abs() < 1e-12 {
        // Parallel segments
        s = 0.0;
        t = (d5 / d3).clamp(0.0, 1.0);
    } else {
        s = ((d2 * d5 - d3 * d4) / denom).clamp(0.0, 1.0);
        t = ((d1 * d5 - d2 * d4) / denom).clamp(0.0, 1.0);
    }

    // Re-clamp since clamping one parameter may invalidate the other
    #[allow(unused_variables)]
    let closest_a = a + ab * s;

    // Refine: project back onto each segment for better accuracy
    let t_refined = if d3 > 1e-12 {
        ((closest_a - c).dot(cd) / d3).clamp(0.0, 1.0)
    } else {
        t
    };
    let closest_c = c + cd * t_refined;

    let s_refined = if d1 > 1e-12 {
        ((closest_c - a).dot(ab) / d1).clamp(0.0, 1.0)
    } else {
        s
    };
    let closest_a = a + ab * s_refined;

    (closest_a, closest_c)
}

/// Test two capsules for overlap.
pub fn intersect_capsules(
    pos_a: Vec3,
    radius_a: f32,
    half_height_a: f32,
    pos_b: Vec3,
    radius_b: f32,
    half_height_b: f32,
) -> Option<(Vec3, f32)> {
    let (a0, a1) = capsule_segment(pos_a, half_height_a);
    let (b0, b1) = capsule_segment(pos_b, half_height_b);

    let (ca, cb) = closest_points_segments(a0, a1, b0, b1);
    intersect_spheres(ca, radius_a, cb, radius_b)
}

/// Test capsule vs sphere overlap.
pub fn intersect_capsule_sphere(
    cap_pos: Vec3,
    cap_radius: f32,
    cap_half_height: f32,
    sphere_pos: Vec3,
    sphere_radius: f32,
) -> Option<(Vec3, f32)> {
    let (a, b) = capsule_segment(cap_pos, cap_half_height);
    let closest = closest_point_on_segment(a, b, sphere_pos);
    intersect_spheres(closest, cap_radius, sphere_pos, sphere_radius)
}

/// Test capsule vs AABB overlap (approximate: finds closest point on capsule
/// spine to AABB, then does sphere-AABB test at that point).
pub fn intersect_capsule_aabb(
    cap_pos: Vec3,
    cap_radius: f32,
    cap_half_height: f32,
    aabb_pos: Vec3,
    hx: f32,
    hy: f32,
    hz: f32,
) -> Option<(Vec3, f32)> {
    let (a, b) = capsule_segment(cap_pos, cap_half_height);
    // Find the point on the capsule spine closest to the AABB center
    let closest_on_spine = closest_point_on_segment(a, b, aabb_pos);
    // Now test that sphere against the AABB
    intersect_aabb_sphere(aabb_pos, hx, hy, hz, closest_on_spine, cap_radius)
}

/// Test two collider shapes for overlap. Dispatches to the correct narrow-phase
/// test based on shape types. Returns `(normal_A_to_B, penetration_depth)` on overlap.
pub fn intersect_shapes(
    pos_a: Vec3,
    shape_a: &ColliderShape,
    pos_b: Vec3,
    shape_b: &ColliderShape,
) -> Option<(Vec3, f32)> {
    match (shape_a, shape_b) {
        // AABB vs AABB
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
        // Sphere vs Sphere
        (ColliderShape::Sphere { radius: ra }, ColliderShape::Sphere { radius: rb }) => {
            intersect_spheres(pos_a, *ra, pos_b, *rb)
        }
        // AABB vs Sphere
        (ColliderShape::Aabb { hx, hy, hz }, ColliderShape::Sphere { radius }) => {
            intersect_aabb_sphere(pos_a, *hx, *hy, *hz, pos_b, *radius)
        }
        (ColliderShape::Sphere { radius }, ColliderShape::Aabb { hx, hy, hz }) => {
            intersect_aabb_sphere(pos_b, *hx, *hy, *hz, pos_a, *radius).map(|(n, d)| (-n, d))
        }
        // Capsule vs Capsule
        (
            ColliderShape::Capsule {
                radius: ra,
                half_height: ha,
            },
            ColliderShape::Capsule {
                radius: rb,
                half_height: hb,
            },
        ) => intersect_capsules(pos_a, *ra, *ha, pos_b, *rb, *hb),
        // Capsule vs Sphere
        (
            ColliderShape::Capsule {
                radius,
                half_height,
            },
            ColliderShape::Sphere { radius: sr },
        ) => intersect_capsule_sphere(pos_a, *radius, *half_height, pos_b, *sr),
        (
            ColliderShape::Sphere { radius: sr },
            ColliderShape::Capsule {
                radius,
                half_height,
            },
        ) => {
            intersect_capsule_sphere(pos_b, *radius, *half_height, pos_a, *sr).map(|(n, d)| (-n, d))
        }
        // Capsule vs AABB
        (
            ColliderShape::Capsule {
                radius,
                half_height,
            },
            ColliderShape::Aabb { hx, hy, hz },
        ) => intersect_capsule_aabb(pos_a, *radius, *half_height, pos_b, *hx, *hy, *hz),
        (
            ColliderShape::Aabb { hx, hy, hz },
            ColliderShape::Capsule {
                radius,
                half_height,
            },
        ) => intersect_capsule_aabb(pos_b, *radius, *half_height, pos_a, *hx, *hy, *hz)
            .map(|(n, d)| (-n, d)),
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

    // ── Capsule tests ──

    #[test]
    fn capsule_capsule_overlap() {
        // Two vertical capsules side by side
        let r = intersect_capsules(Vec3::ZERO, 0.5, 1.0, Vec3::new(0.8, 0.0, 0.0), 0.5, 1.0);
        assert!(r.is_some());
        let (_, depth) = r.unwrap();
        assert!(depth > 0.0);
    }

    #[test]
    fn capsule_capsule_no_overlap() {
        let r = intersect_capsules(Vec3::ZERO, 0.5, 1.0, Vec3::new(5.0, 0.0, 0.0), 0.5, 1.0);
        assert!(r.is_none());
    }

    #[test]
    fn capsule_sphere_overlap() {
        let r = intersect_capsule_sphere(Vec3::ZERO, 0.5, 1.0, Vec3::new(0.8, 0.0, 0.0), 0.5);
        assert!(r.is_some());
    }

    #[test]
    fn capsule_sphere_end_overlap() {
        // Sphere near the top hemisphere of capsule
        let r = intersect_capsule_sphere(Vec3::ZERO, 0.5, 1.0, Vec3::new(0.0, 1.3, 0.0), 0.5);
        assert!(r.is_some());
    }

    #[test]
    fn capsule_aabb_overlap() {
        let r = intersect_capsule_aabb(
            Vec3::ZERO,
            0.5,
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            1.0,
            1.0,
        );
        assert!(r.is_some());
    }

    #[test]
    fn capsule_via_shapes_dispatcher() {
        let cap = ColliderShape::Capsule {
            radius: 0.5,
            half_height: 1.0,
        };
        let sphere = ColliderShape::Sphere { radius: 0.5 };
        let aabb = ColliderShape::Aabb {
            hx: 1.0,
            hy: 1.0,
            hz: 1.0,
        };

        // Capsule vs Sphere
        assert!(intersect_shapes(Vec3::ZERO, &cap, Vec3::new(0.8, 0.0, 0.0), &sphere).is_some());
        // Sphere vs Capsule (reversed)
        assert!(intersect_shapes(Vec3::new(0.8, 0.0, 0.0), &sphere, Vec3::ZERO, &cap).is_some());
        // Capsule vs AABB
        assert!(intersect_shapes(Vec3::ZERO, &cap, Vec3::new(1.0, 0.0, 0.0), &aabb).is_some());
        // AABB vs Capsule (reversed)
        assert!(intersect_shapes(Vec3::new(1.0, 0.0, 0.0), &aabb, Vec3::ZERO, &cap).is_some());
        // Capsule vs Capsule
        assert!(intersect_shapes(Vec3::ZERO, &cap, Vec3::new(0.8, 0.0, 0.0), &cap).is_some());
    }
}
