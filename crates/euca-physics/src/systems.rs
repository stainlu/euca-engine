use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;

use crate::collision::intersect_shapes;
use crate::components::*;
use crate::world::PhysicsConfig;

/// Main physics system: apply gravity, integrate velocity, detect collisions, resolve.
pub fn physics_step_system(world: &mut World) {
    let config = world
        .resource::<PhysicsConfig>()
        .cloned()
        .unwrap_or_default();
    let dt = config.fixed_dt;
    let gravity = config.gravity;

    // Step 1: Apply gravity to dynamic bodies
    apply_gravity(world, gravity, dt);

    // Step 2: Integrate velocity → position
    integrate_positions(world, dt);

    // Step 3: Detect and resolve collisions
    resolve_collisions(world);
}

fn apply_gravity(world: &mut World, gravity: Vec3, dt: f32) {
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &PhysicsBody, &Velocity)>::new(world);
        query
            .iter()
            .filter(|(_, body, _)| body.body_type == RigidBodyType::Dynamic)
            .map(|(e, _, _)| e)
            .collect()
    };

    for entity in entities {
        let g = world.get::<Gravity>(entity).map(|g| g.0).unwrap_or(gravity);
        if let Some(vel) = world.get_mut::<Velocity>(entity) {
            vel.linear = vel.linear + g * dt;
        }
    }
}

fn integrate_positions(world: &mut World, dt: f32) {
    let updates: Vec<(Entity, Vec3)> = {
        let query = Query::<(Entity, &PhysicsBody, &Velocity)>::new(world);
        query
            .iter()
            .filter(|(_, body, _)| body.body_type == RigidBodyType::Dynamic)
            .map(|(e, _, vel)| (e, vel.linear))
            .collect()
    };

    for (entity, linear_vel) in updates {
        if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
            lt.0.translation = lt.0.translation + linear_vel * dt;
        }
    }
}

/// Minimum approach speed for a bounce to occur. Below this, the object comes to rest.
const REST_VELOCITY_THRESHOLD: f32 = 0.5;

/// Spatial hash cell size. Bodies are inserted into all cells their AABB overlaps.
const BROADPHASE_CELL_SIZE: f32 = 4.0;

/// Compute the AABB extents for any collider shape.
fn shape_extent(shape: &ColliderShape) -> f32 {
    match shape {
        ColliderShape::Aabb { hx, hy, hz } => hx.max(*hy).max(*hz),
        ColliderShape::Sphere { radius } => *radius,
        ColliderShape::Capsule {
            radius,
            half_height,
        } => radius + half_height,
    }
}

/// Collectable body data for broadphase + narrowphase.
struct Body {
    entity: Entity,
    pos: Vec3,
    shape: ColliderShape,
    body_type: RigidBodyType,
    restitution: f32,
    friction: f32,
}

/// Spatial hash broadphase: returns candidate pairs (indices into bodies slice).
/// Only pairs sharing at least one grid cell are returned. Eliminates most
/// non-colliding pairs for O(n * avg_neighbors) instead of O(n²).
fn broadphase_spatial_hash(bodies: &[Body]) -> Vec<(usize, usize)> {
    use std::collections::{HashMap, HashSet};

    if bodies.len() < 20 {
        // For small body counts, O(n²) is faster than hashing overhead
        let mut pairs = Vec::new();
        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                pairs.push((i, j));
            }
        }
        return pairs;
    }

    let inv_cell = 1.0 / BROADPHASE_CELL_SIZE;

    // Map: cell_key → list of body indices
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

    for (idx, body) in bodies.iter().enumerate() {
        let ext = shape_extent(&body.shape);
        let min_x = ((body.pos.x - ext) * inv_cell).floor() as i32;
        let max_x = ((body.pos.x + ext) * inv_cell).floor() as i32;
        let min_y = ((body.pos.y - ext) * inv_cell).floor() as i32;
        let max_y = ((body.pos.y + ext) * inv_cell).floor() as i32;
        let min_z = ((body.pos.z - ext) * inv_cell).floor() as i32;
        let max_z = ((body.pos.z + ext) * inv_cell).floor() as i32;

        for cx in min_x..=max_x {
            for cy in min_y..=max_y {
                for cz in min_z..=max_z {
                    grid.entry((cx, cy, cz)).or_default().push(idx);
                }
            }
        }
    }

    // Collect unique pairs from cells
    let mut pair_set: HashSet<(usize, usize)> = HashSet::new();
    for cell_bodies in grid.values() {
        for i in 0..cell_bodies.len() {
            for j in (i + 1)..cell_bodies.len() {
                let a = cell_bodies[i];
                let b = cell_bodies[j];
                let pair = if a < b { (a, b) } else { (b, a) };
                pair_set.insert(pair);
            }
        }
    }

    pair_set.into_iter().collect()
}

fn resolve_collisions(world: &mut World) {
    let bodies: Vec<Body> = {
        let query = Query::<(Entity, &LocalTransform, &Collider)>::new(world);
        query
            .iter()
            .filter_map(|(e, lt, col)| {
                let body = world.get::<PhysicsBody>(e)?;
                Some(Body {
                    entity: e,
                    pos: lt.0.translation,
                    shape: col.shape.clone(),
                    body_type: body.body_type,
                    restitution: col.restitution,
                    friction: col.friction,
                })
            })
            .collect()
    };

    // Spatial hash broadphase: insert bodies into grid cells, test only within same cells
    let candidate_pairs = broadphase_spatial_hash(&bodies);
    let mut corrections: Vec<(Entity, Vec3, Vec3)> = Vec::with_capacity(bodies.len());

    for (i, j) in candidate_pairs {
        let a = &bodies[i];
        let b = &bodies[j];

        if a.body_type == RigidBodyType::Static && b.body_type == RigidBodyType::Static {
            continue;
        }

        if let Some((normal, depth)) = intersect_shapes(a.pos, &a.shape, b.pos, &b.shape) {
            let restitution = a.restitution * b.restitution;
            let friction = (a.friction * b.friction).sqrt();

            // Push-out resolution
            match (a.body_type, b.body_type) {
                (RigidBodyType::Dynamic, RigidBodyType::Static) => {
                    corrections.push((a.entity, normal * (-depth), Vec3::ZERO));
                }
                (RigidBodyType::Static, RigidBodyType::Dynamic) => {
                    corrections.push((b.entity, normal * depth, Vec3::ZERO));
                }
                (RigidBodyType::Dynamic, RigidBodyType::Dynamic) => {
                    corrections.push((a.entity, normal * (-depth * 0.5), Vec3::ZERO));
                    corrections.push((b.entity, normal * (depth * 0.5), Vec3::ZERO));
                }
                _ => {}
            }

            // Velocity response for dynamic body A
            if a.body_type == RigidBodyType::Dynamic
                && let Some(vel) = world.get::<Velocity>(a.entity)
            {
                let n = normal * (-1.0);
                let vn = vel.linear.dot(n);
                if vn < 0.0 {
                    let approach_speed = -vn;
                    let new_normal_vel = if approach_speed < REST_VELOCITY_THRESHOLD {
                        0.0
                    } else {
                        approach_speed * restitution
                    };
                    let normal_correction = n * (new_normal_vel - vn);
                    let tangent_vel = vel.linear - n * vn;
                    let friction_correction = tangent_vel * (-friction);
                    corrections.push((
                        a.entity,
                        Vec3::ZERO,
                        normal_correction + friction_correction,
                    ));
                }
            }

            // Velocity response for dynamic body B
            if b.body_type == RigidBodyType::Dynamic
                && let Some(vel) = world.get::<Velocity>(b.entity)
            {
                let n = normal;
                let vn = vel.linear.dot(n);
                if vn < 0.0 {
                    let approach_speed = -vn;
                    let new_normal_vel = if approach_speed < REST_VELOCITY_THRESHOLD {
                        0.0
                    } else {
                        approach_speed * restitution
                    };
                    let normal_correction = n * (new_normal_vel - vn);
                    let tangent_vel = vel.linear - n * vn;
                    let friction_correction = tangent_vel * (-friction);
                    corrections.push((
                        b.entity,
                        Vec3::ZERO,
                        normal_correction + friction_correction,
                    ));
                }
            }
        }
    }

    // Apply corrections
    for (entity, pos_correction, vel_correction) in corrections {
        if pos_correction.length_squared() > 0.0
            && let Some(lt) = world.get_mut::<LocalTransform>(entity)
        {
            lt.0.translation = lt.0.translation + pos_correction;
        }
        if vel_correction.length_squared() > 0.0
            && let Some(vel) = world.get_mut::<Velocity>(entity)
        {
            vel.linear = vel.linear + vel_correction;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;
    use euca_scene::GlobalTransform;

    #[test]
    fn gravity_moves_dynamic_body() {
        let mut world = World::new();
        world.insert_resource(PhysicsConfig::new());

        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 10.0, 0.0,
        ))));
        world.insert(entity, GlobalTransform::default());
        world.insert(entity, PhysicsBody::dynamic());
        world.insert(entity, Velocity::default());
        world.insert(entity, Collider::aabb(0.5, 0.5, 0.5));

        for _ in 0..120 {
            physics_step_system(&mut world);
        }

        let lt = world.get::<LocalTransform>(entity).unwrap();
        assert!(
            lt.0.translation.y < 0.0,
            "Body should have fallen past origin, y={}",
            lt.0.translation.y
        );
    }

    #[test]
    fn static_body_does_not_move() {
        let mut world = World::new();
        world.insert_resource(PhysicsConfig::new());

        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(entity, GlobalTransform::default());
        world.insert(entity, PhysicsBody::fixed());
        world.insert(entity, Collider::aabb(10.0, 0.5, 10.0));

        for _ in 0..60 {
            physics_step_system(&mut world);
        }

        let lt = world.get::<LocalTransform>(entity).unwrap();
        assert!((lt.0.translation.y).abs() < 0.01);
    }

    #[test]
    fn dynamic_body_lands_on_static() {
        let mut world = World::new();
        world.insert_resource(PhysicsConfig::new());

        // Ground at y=0
        let ground = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(ground, GlobalTransform::default());
        world.insert(ground, PhysicsBody::fixed());
        world.insert(ground, Collider::aabb(10.0, 0.5, 10.0));

        // Cube at y=5
        let cube = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 5.0, 0.0,
        ))));
        world.insert(cube, GlobalTransform::default());
        world.insert(cube, PhysicsBody::dynamic());
        world.insert(cube, Velocity::default());
        world.insert(cube, Collider::aabb(0.5, 0.5, 0.5));

        for _ in 0..300 {
            physics_step_system(&mut world);
        }

        let lt = world.get::<LocalTransform>(cube).unwrap();
        assert!(
            lt.0.translation.y > -1.0,
            "Cube should be near ground, y={}",
            lt.0.translation.y
        );
        assert!(
            lt.0.translation.y < 5.0,
            "Cube should have fallen, y={}",
            lt.0.translation.y
        );
    }
}
