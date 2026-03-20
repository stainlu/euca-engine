use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;

use crate::collision::intersect_shapes;
use crate::components::{Collider, ColliderShape};

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
pub fn raycast_collider(ray: &Ray, pos: euca_math::Vec3, collider: &Collider) -> Option<RayHit> {
    match &collider.shape {
        ColliderShape::Aabb { hx, hy, hz } => raycast_aabb(ray, pos, *hx, *hy, *hz),
        ColliderShape::Sphere { radius } => raycast_sphere(ray, pos, *radius),
        ColliderShape::Capsule {
            radius,
            half_height,
        } => raycast_capsule(ray, pos, *radius, *half_height),
    }
}

// ── Scene query types ──

/// A world-space raycast hit that includes the entity.
#[derive(Clone, Copy, Debug)]
pub struct WorldRayHit {
    pub entity: Entity,
    /// Distance along the ray.
    pub t: f32,
    /// Hit point in world space.
    pub point: Vec3,
    /// Surface normal at the hit point.
    pub normal: Vec3,
}

/// An entity found by an overlap query.
#[derive(Clone, Copy, Debug)]
pub struct OverlapHit {
    pub entity: Entity,
}

/// A hit from a sweep (shape-cast) query.
#[derive(Clone, Copy, Debug)]
pub struct SweepHit {
    pub entity: Entity,
    /// Parameter along the sweep direction (0.0 = start, 1.0 = end of sweep distance).
    pub t: f32,
    /// Point of first contact in world space.
    pub point: Vec3,
    /// Contact normal.
    pub normal: Vec3,
}

// ── Scene query functions ──

/// Cast a ray against all colliders in the world and return all hits, sorted
/// by distance. Optionally filter by a layer mask (only colliders whose
/// `layer & query_mask != 0` are tested). Pass `u32::MAX` for no filtering.
///
/// `max_distance`: maximum ray distance to consider. Pass `f32::INFINITY` for unlimited.
pub fn raycast_world(
    world: &World,
    ray: &Ray,
    max_distance: f32,
    query_mask: u32,
) -> Vec<WorldRayHit> {
    let mut hits = Vec::new();

    let query = Query::<(Entity, &LocalTransform, &Collider)>::new(world);
    for (entity, lt, collider) in query.iter() {
        // Layer filter: the query acts as if it has layer=query_mask, mask=query_mask
        if (collider.layer & query_mask) == 0 {
            continue;
        }

        let pos = lt.0.translation;
        if let Some(hit) = raycast_collider(ray, pos, collider)
            && hit.t >= 0.0
            && hit.t <= max_distance
        {
            hits.push(WorldRayHit {
                entity,
                t: hit.t,
                point: hit.point,
                normal: hit.normal,
            });
        }
    }

    hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

/// Find all entities whose colliders overlap a sphere at `center` with `radius`.
/// Filter by `query_mask` (only colliders with `layer & query_mask != 0` are tested).
pub fn overlap_sphere(
    world: &World,
    center: Vec3,
    radius: f32,
    query_mask: u32,
) -> Vec<OverlapHit> {
    let probe = ColliderShape::Sphere { radius };
    let mut results = Vec::new();

    let query = Query::<(Entity, &LocalTransform, &Collider)>::new(world);
    for (entity, lt, collider) in query.iter() {
        if (collider.layer & query_mask) == 0 {
            continue;
        }

        let pos = lt.0.translation;
        if intersect_shapes(center, &probe, pos, &collider.shape).is_some() {
            results.push(OverlapHit { entity });
        }
    }

    results
}

/// Sweep (shape-cast) a sphere from `origin` along `direction` for `max_distance`.
/// Returns all entities hit, sorted by distance.
///
/// Implemented by inflating each collider by the sweep radius, then raycasting
/// against the inflated shape. This is exact for sphere-vs-sphere, and a
/// conservative approximation for sphere-vs-AABB and sphere-vs-capsule.
pub fn sweep_sphere(
    world: &World,
    origin: Vec3,
    direction: Vec3,
    radius: f32,
    max_distance: f32,
    query_mask: u32,
) -> Vec<SweepHit> {
    let ray = Ray::new(origin, direction);
    let mut hits = Vec::new();

    let query = Query::<(Entity, &LocalTransform, &Collider)>::new(world);
    for (entity, lt, collider) in query.iter() {
        if (collider.layer & query_mask) == 0 {
            continue;
        }

        let pos = lt.0.translation;

        // Inflate the target shape by the sweep radius (Minkowski sum approximation)
        let hit = match &collider.shape {
            ColliderShape::Sphere {
                radius: target_radius,
            } => raycast_sphere(&ray, pos, target_radius + radius),
            ColliderShape::Aabb { hx, hy, hz } => {
                raycast_aabb(&ray, pos, hx + radius, hy + radius, hz + radius)
            }
            ColliderShape::Capsule {
                radius: cap_radius,
                half_height,
            } => raycast_capsule(&ray, pos, cap_radius + radius, *half_height),
        };

        if let Some(h) = hit
            && h.t >= 0.0
            && h.t <= max_distance
        {
            hits.push(SweepHit {
                entity,
                t: h.t,
                point: h.point,
                normal: h.normal,
            });
        }
    }

    hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::PhysicsBody;
    use euca_math::Transform;
    use euca_scene::GlobalTransform;

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

    // ── Scene query tests ──

    fn setup_scene_world() -> (World, Entity, Entity, Entity) {
        let mut world = World::new();

        // Sphere at origin, layer 1
        let a = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(a, GlobalTransform::default());
        world.insert(a, PhysicsBody::dynamic());
        world.insert(a, Collider::sphere(1.0).with_layer(1));

        // Sphere at (5,0,0), layer 2
        let b = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            5.0, 0.0, 0.0,
        ))));
        world.insert(b, GlobalTransform::default());
        world.insert(b, PhysicsBody::dynamic());
        world.insert(b, Collider::sphere(1.0).with_layer(2));

        // AABB at (10,0,0), layer 1
        let c = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            10.0, 0.0, 0.0,
        ))));
        world.insert(c, GlobalTransform::default());
        world.insert(c, PhysicsBody::fixed());
        world.insert(c, Collider::aabb(1.0, 1.0, 1.0).with_layer(1));

        (world, a, b, c)
    }

    #[test]
    fn raycast_world_multi_hit() {
        let (world, _a, _b, _c) = setup_scene_world();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);

        // Query all layers
        let hits = raycast_world(&world, &ray, f32::INFINITY, u32::MAX);
        assert_eq!(hits.len(), 3, "Should hit all 3 colliders");
        // Should be sorted by distance
        assert!(hits[0].t <= hits[1].t);
        assert!(hits[1].t <= hits[2].t);
    }

    #[test]
    fn raycast_world_layer_filter() {
        let (world, _a, _b, _c) = setup_scene_world();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);

        // Query only layer 1
        let hits = raycast_world(&world, &ray, f32::INFINITY, 1);
        assert_eq!(hits.len(), 2, "Should hit only layer-1 colliders");
    }

    #[test]
    fn raycast_world_max_distance() {
        let (world, _a, _b, _c) = setup_scene_world();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);

        // Only close hits
        let hits = raycast_world(&world, &ray, 6.0, u32::MAX);
        assert_eq!(hits.len(), 1, "Should hit only the first sphere");
    }

    #[test]
    fn overlap_sphere_finds_nearby() {
        let (world, a, _b, _c) = setup_scene_world();

        // Overlap at origin with radius 2 should hit entity A (sphere at origin, r=1)
        let hits = overlap_sphere(&world, Vec3::ZERO, 2.0, u32::MAX);
        assert!(!hits.is_empty());
        assert!(hits.iter().any(|h| h.entity == a));
    }

    #[test]
    fn overlap_sphere_layer_filter() {
        let (world, _a, _b, _c) = setup_scene_world();

        // Overlap at origin with a large radius but only layer 2
        let hits = overlap_sphere(&world, Vec3::ZERO, 100.0, 2);
        // Should only find entity B (layer 2)
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn sweep_sphere_hits() {
        let (world, _a, _b, _c) = setup_scene_world();

        // Sweep a small sphere from far left toward +X
        let hits = sweep_sphere(
            &world,
            Vec3::new(-10.0, 0.0, 0.0),
            Vec3::X,
            0.5,
            100.0,
            u32::MAX,
        );
        assert!(hits.len() >= 2, "Should hit multiple colliders");
        assert!(hits[0].t <= hits[1].t, "Should be sorted by distance");
    }

    #[test]
    fn sweep_sphere_layer_filter() {
        let (world, _a, _b, _c) = setup_scene_world();

        let hits = sweep_sphere(
            &world,
            Vec3::new(-10.0, 0.0, 0.0),
            Vec3::X,
            0.5,
            100.0,
            2, // only layer 2
        );
        assert_eq!(hits.len(), 1, "Should only hit layer-2 entity");
    }
}
