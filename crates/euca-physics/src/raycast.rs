use euca_math::Vec3;

/// A ray defined by origin and direction.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Get the point at parameter t along the ray.
    pub fn at(self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Result of a raycast hit.
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// Distance along the ray to the hit point.
    pub t: f32,
    /// Hit point in world space.
    pub point: Vec3,
    /// Surface normal at the hit point.
    pub normal: Vec3,
}

/// Raycast against an AABB. Returns the distance t if hit.
pub fn raycast_aabb(ray: &Ray, aabb_center: Vec3, hx: f32, hy: f32, hz: f32) -> Option<RayHit> {
    let min = Vec3::new(aabb_center.x - hx, aabb_center.y - hy, aabb_center.z - hz);
    let max = Vec3::new(aabb_center.x + hx, aabb_center.y + hy, aabb_center.z + hz);

    let inv_dir = Vec3::new(
        1.0 / ray.direction.x,
        1.0 / ray.direction.y,
        1.0 / ray.direction.z,
    );

    let t1 = (min.x - ray.origin.x) * inv_dir.x;
    let t2 = (max.x - ray.origin.x) * inv_dir.x;
    let t3 = (min.y - ray.origin.y) * inv_dir.y;
    let t4 = (max.y - ray.origin.y) * inv_dir.y;
    let t5 = (min.z - ray.origin.z) * inv_dir.z;
    let t6 = (max.z - ray.origin.z) * inv_dir.z;

    let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
    let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

    if tmax < 0.0 || tmin > tmax {
        return None;
    }

    let t = if tmin >= 0.0 { tmin } else { tmax };
    let point = ray.at(t);

    // Determine which face was hit (normal)
    let normal = if (point.x - min.x).abs() < 1e-4 {
        Vec3::new(-1.0, 0.0, 0.0)
    } else if (point.x - max.x).abs() < 1e-4 {
        Vec3::X
    } else if (point.y - min.y).abs() < 1e-4 {
        Vec3::new(0.0, -1.0, 0.0)
    } else if (point.y - max.y).abs() < 1e-4 {
        Vec3::Y
    } else if (point.z - min.z).abs() < 1e-4 {
        Vec3::new(0.0, 0.0, -1.0)
    } else {
        Vec3::Z
    };

    Some(RayHit { t, point, normal })
}

/// Raycast against a sphere. Returns the distance t if hit.
pub fn raycast_sphere(ray: &Ray, center: Vec3, radius: f32) -> Option<RayHit> {
    let oc = ray.origin - center;
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.dot(oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None;
    }

    let sqrt_d = discriminant.sqrt();
    let t1 = (-b - sqrt_d) / (2.0 * a);
    let t2 = (-b + sqrt_d) / (2.0 * a);

    let t = if t1 >= 0.0 {
        t1
    } else if t2 >= 0.0 {
        t2
    } else {
        return None;
    };

    let point = ray.at(t);
    let normal = (point - center).normalize();

    Some(RayHit { t, point, normal })
}

/// Raycast against a capsule (Y-axis aligned).
/// Tests the cylinder body and two hemisphere endcaps.
pub fn raycast_capsule(ray: &Ray, center: Vec3, radius: f32, half_height: f32) -> Option<RayHit> {
    let top = Vec3::new(center.x, center.y + half_height, center.z);
    let bottom = Vec3::new(center.x, center.y - half_height, center.z);

    // Test against top and bottom hemispheres
    let hit_top = raycast_sphere(ray, top, radius);
    let hit_bottom = raycast_sphere(ray, bottom, radius);

    // Test against the infinite cylinder (XZ plane), then clamp to segment
    let oc = Vec3::new(ray.origin.x - center.x, 0.0, ray.origin.z - center.z);
    let dir_xz = Vec3::new(ray.direction.x, 0.0, ray.direction.z);
    let a = dir_xz.dot(dir_xz);
    let b = 2.0 * oc.dot(dir_xz);
    let c = oc.dot(oc) - radius * radius;

    let hit_cyl = if a > 1e-12 {
        let discriminant = b * b - 4.0 * a * c;
        if discriminant >= 0.0 {
            let sqrt_d = discriminant.sqrt();
            let t1 = (-b - sqrt_d) / (2.0 * a);
            let t2 = (-b + sqrt_d) / (2.0 * a);
            let t = if t1 >= 0.0 { t1 } else { t2 };
            if t >= 0.0 {
                let point = ray.at(t);
                // Check if hit is within the cylinder segment (between bottom and top Y)
                if point.y >= center.y - half_height && point.y <= center.y + half_height {
                    let normal = Vec3::new(point.x - center.x, 0.0, point.z - center.z).normalize();
                    Some(RayHit { t, point, normal })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Return the closest hit among cylinder and hemispheres
    [hit_cyl, hit_top, hit_bottom]
        .into_iter()
        .flatten()
        .min_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal))
}

/// Raycast against a collider at a given position. Dispatches by shape.
pub fn raycast_collider(
    ray: &Ray,
    pos: euca_math::Vec3,
    collider: &crate::components::Collider,
) -> Option<RayHit> {
    match &collider.shape {
        crate::components::ColliderShape::Aabb { hx, hy, hz } => {
            raycast_aabb(ray, pos, *hx, *hy, *hz)
        }
        crate::components::ColliderShape::Sphere { radius } => raycast_sphere(ray, pos, *radius),
        crate::components::ColliderShape::Capsule {
            radius,
            half_height,
        } => raycast_capsule(ray, pos, *radius, *half_height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_hits_aabb() {
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
        let hit = raycast_aabb(&ray, Vec3::ZERO, 1.0, 1.0, 1.0);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.t - 4.0).abs() < 1e-4); // hits at x=-1, distance 4 from origin at -5
    }

    #[test]
    fn ray_misses_aabb() {
        let ray = Ray::new(Vec3::new(-5.0, 5.0, 0.0), Vec3::X);
        let hit = raycast_aabb(&ray, Vec3::ZERO, 1.0, 1.0, 1.0);
        assert!(hit.is_none());
    }

    #[test]
    fn ray_hits_sphere() {
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
        let hit = raycast_sphere(&ray, Vec3::ZERO, 1.0);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.t - 4.0).abs() < 1e-4);
        assert!((h.normal.x - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn ray_misses_sphere() {
        let ray = Ray::new(Vec3::new(-5.0, 5.0, 0.0), Vec3::X);
        let hit = raycast_sphere(&ray, Vec3::ZERO, 1.0);
        assert!(hit.is_none());
    }

    #[test]
    fn ray_hits_capsule_body() {
        // Ray hitting the cylinder body of a capsule
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
        let hit = raycast_capsule(&ray, Vec3::ZERO, 1.0, 1.0);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.t - 4.0).abs() < 1e-3);
    }

    #[test]
    fn ray_hits_capsule_hemisphere() {
        // Ray hitting the top hemisphere
        let ray = Ray::new(Vec3::new(-5.0, 1.5, 0.0), Vec3::X);
        let hit = raycast_capsule(&ray, Vec3::ZERO, 1.0, 1.0);
        assert!(hit.is_some());
    }

    #[test]
    fn ray_misses_capsule() {
        let ray = Ray::new(Vec3::new(-5.0, 5.0, 0.0), Vec3::X);
        let hit = raycast_capsule(&ray, Vec3::ZERO, 0.5, 1.0);
        assert!(hit.is_none());
    }
}
