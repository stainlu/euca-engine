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
///
/// Returns `(normal_A_to_B, penetration_depth, contact_point)` on overlap.
/// The contact point is the midpoint of the two centers, offset to lie on the
/// overlap face along the minimum penetration axis.
// clippy::too_many_arguments — separating position and per-axis half-extents
// avoids allocating intermediate AABB structs in the hot collision loop.
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
) -> Option<(Vec3, f32, Vec3)> {
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

    // Contact point: midpoint of the overlap region along the penetration axis.
    // On the minimum-penetration axis, the contact sits at the boundary between
    // the two AABBs. On the other two axes, use the midpoint of the centers.
    let mid = (pos_a + pos_b) * 0.5;

    if overlap_x <= overlap_y && overlap_x <= overlap_z {
        let sign = if pos_b.x > pos_a.x { 1.0 } else { -1.0 };
        let contact_x = pos_a.x + sign * hx_a;
        let contact = Vec3::new(contact_x, mid.y, mid.z);
        Some((Vec3::new(sign, 0.0, 0.0), overlap_x, contact))
    } else if overlap_y <= overlap_z {
        let sign = if pos_b.y > pos_a.y { 1.0 } else { -1.0 };
        let contact_y = pos_a.y + sign * hy_a;
        let contact = Vec3::new(mid.x, contact_y, mid.z);
        Some((Vec3::new(0.0, sign, 0.0), overlap_y, contact))
    } else {
        let sign = if pos_b.z > pos_a.z { 1.0 } else { -1.0 };
        let contact_z = pos_a.z + sign * hz_a;
        let contact = Vec3::new(mid.x, mid.y, contact_z);
        Some((Vec3::new(0.0, 0.0, sign), overlap_z, contact))
    }
}

/// Test if two spheres overlap.
///
/// Returns `(normal_A_to_B, penetration_depth, contact_point)` on overlap.
/// The contact point lies on the line between centers at A's surface.
pub fn intersect_spheres(
    pos_a: Vec3,
    radius_a: f32,
    pos_b: Vec3,
    radius_b: f32,
) -> Option<(Vec3, f32, Vec3)> {
    let diff = pos_b - pos_a;
    let dist_sq = diff.length_squared();
    let sum_r = radius_a + radius_b;

    if dist_sq >= sum_r * sum_r {
        return None;
    }

    let dist = dist_sq.sqrt();
    if dist < 1e-6 {
        // Degenerate: same position. Pick arbitrary normal, contact at center.
        return Some((Vec3::Y, sum_r, pos_a));
    }

    let normal = diff * (1.0 / dist);
    let depth = sum_r - dist;
    let contact = pos_a + normal * radius_a;
    Some((normal, depth, contact))
}

/// Test AABB vs Sphere overlap.
///
/// Returns `(normal_AABB_to_Sphere, penetration_depth, contact_point)` on overlap.
/// The contact point is the closest point on the AABB surface to the sphere center.
pub fn intersect_aabb_sphere(
    aabb_pos: Vec3,
    hx: f32,
    hy: f32,
    hz: f32,
    sphere_pos: Vec3,
    radius: f32,
) -> Option<(Vec3, f32, Vec3)> {
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
        return Some((Vec3::Y, radius, closest));
    }

    let normal = diff * (1.0 / dist);
    let depth = radius - dist;
    Some((normal, depth, closest))
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

/// Shift a sphere-vs-sphere contact point from A's surface to the midpoint
/// of the overlap region. Used by capsule tests that delegate to `intersect_spheres`.
fn midpoint_contact(result: (Vec3, f32, Vec3)) -> (Vec3, f32, Vec3) {
    let (normal, depth, surface_a) = result;
    let contact = surface_a + normal * (depth * 0.5);
    (normal, depth, contact)
}

/// Test two capsules for overlap.
///
/// Returns `(normal_A_to_B, penetration_depth, contact_point)` on overlap.
/// The contact point is the midpoint of the overlap between the two capsule surfaces.
pub fn intersect_capsules(
    pos_a: Vec3,
    radius_a: f32,
    half_height_a: f32,
    pos_b: Vec3,
    radius_b: f32,
    half_height_b: f32,
) -> Option<(Vec3, f32, Vec3)> {
    let (a0, a1) = capsule_segment(pos_a, half_height_a);
    let (b0, b1) = capsule_segment(pos_b, half_height_b);

    let (ca, cb) = closest_points_segments(a0, a1, b0, b1);
    intersect_spheres(ca, radius_a, cb, radius_b).map(midpoint_contact)
}

/// Test capsule vs sphere overlap.
///
/// Returns `(normal_capsule_to_sphere, penetration_depth, contact_point)`.
/// The contact point is the midpoint of the overlap between the capsule
/// surface and the sphere surface.
pub fn intersect_capsule_sphere(
    cap_pos: Vec3,
    cap_radius: f32,
    cap_half_height: f32,
    sphere_pos: Vec3,
    sphere_radius: f32,
) -> Option<(Vec3, f32, Vec3)> {
    let (a, b) = capsule_segment(cap_pos, cap_half_height);
    let closest = closest_point_on_segment(a, b, sphere_pos);
    intersect_spheres(closest, cap_radius, sphere_pos, sphere_radius).map(midpoint_contact)
}

/// Test capsule vs AABB overlap (approximate: finds closest point on capsule
/// spine to AABB, then does sphere-AABB test at that point).
///
/// Returns `(normal_AABB_to_capsule, penetration_depth, contact_point)`.
pub fn intersect_capsule_aabb(
    cap_pos: Vec3,
    cap_radius: f32,
    cap_half_height: f32,
    aabb_pos: Vec3,
    hx: f32,
    hy: f32,
    hz: f32,
) -> Option<(Vec3, f32, Vec3)> {
    let (a, b) = capsule_segment(cap_pos, cap_half_height);
    // Find the point on the capsule spine closest to the AABB center
    let closest_on_spine = closest_point_on_segment(a, b, aabb_pos);
    // Now test that sphere against the AABB — contact point is already the
    // closest point on the AABB surface from intersect_aabb_sphere.
    intersect_aabb_sphere(aabb_pos, hx, hy, hz, closest_on_spine, cap_radius)
}

/// Test two collider shapes for overlap. Dispatches to the correct narrow-phase
/// test based on shape types.
///
/// Returns `(normal_A_to_B, penetration_depth, contact_point)` on overlap.
pub fn intersect_shapes(
    pos_a: Vec3,
    shape_a: &ColliderShape,
    pos_b: Vec3,
    shape_b: &ColliderShape,
) -> Option<(Vec3, f32, Vec3)> {
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
            intersect_aabb_sphere(pos_b, *hx, *hy, *hz, pos_a, *radius)
                .map(|(n, d, cp)| (-n, d, cp))
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
        ) => intersect_capsule_sphere(pos_b, *radius, *half_height, pos_a, *sr)
            .map(|(n, d, cp)| (-n, d, cp)),
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
            .map(|(n, d, cp)| (-n, d, cp)),
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
        let (normal, depth, contact) = r.unwrap();
        assert!((normal.x - 1.0).abs() < 1e-6);
        assert!(depth > 0.0);
        // Contact should be on A's +X face (x = 1.0), centered on YZ.
        assert!((contact.x - 1.0).abs() < 1e-6, "contact.x = {}", contact.x);
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
        let (_, depth, contact) = r.unwrap();
        assert!((depth - 0.5).abs() < 1e-5);
        // Contact is on A's surface toward B: x = radius_a = 1.0.
        assert!((contact.x - 1.0).abs() < 1e-5, "contact.x = {}", contact.x);
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
        let (_, _, contact) = r.unwrap();
        // Contact is closest AABB point to sphere center = (1.0, 0.0, 0.0).
        assert!((contact.x - 1.0).abs() < 1e-5, "contact.x = {}", contact.x);
    }

    // ── Capsule tests ──

    #[test]
    fn capsule_capsule_overlap() {
        // Two vertical capsules side by side
        let r = intersect_capsules(Vec3::ZERO, 0.5, 1.0, Vec3::new(0.8, 0.0, 0.0), 0.5, 1.0);
        assert!(r.is_some());
        let (_, depth, _contact) = r.unwrap();
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
