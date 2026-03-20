use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;

use crate::collision::intersect_shapes;
use crate::components::*;
use crate::world::PhysicsConfig;

/// Physics system with fixed-timestep accumulation.
///
/// Call with your frame's delta time. Accumulates time and runs fixed-dt
/// substeps as needed. Insert a `PhysicsAccumulator` resource to use this.
/// Falls back to single-step if accumulator is not present.
pub fn physics_step_with_dt(world: &mut World, frame_dt: f32) {
    let config = world
        .resource::<PhysicsConfig>()
        .cloned()
        .unwrap_or_default();

    let accumulator = world
        .resource::<crate::world::PhysicsAccumulator>()
        .map(|a| a.accumulator)
        .unwrap_or(0.0)
        + frame_dt;

    let mut remaining = accumulator;
    let mut steps = 0u32;
    while remaining >= config.fixed_dt && steps < config.max_substeps {
        physics_step_single(world, config.fixed_dt, config.gravity);
        remaining -= config.fixed_dt;
        steps += 1;
    }

    if let Some(acc) = world.resource_mut::<crate::world::PhysicsAccumulator>() {
        acc.accumulator = remaining;
    }
}

/// Main physics system: single fixed-dt step. Use `physics_step_with_dt` for accumulation.
pub fn physics_step_system(world: &mut World) {
    let config = world
        .resource::<PhysicsConfig>()
        .cloned()
        .unwrap_or_default();
    physics_step_single(world, config.fixed_dt, config.gravity);
}

fn physics_step_single(world: &mut World, dt: f32, gravity: Vec3) {
    apply_gravity(world, gravity, dt);
    integrate_positions(world, dt);
    resolve_collisions_and_joints(world);
    update_sleep_states(world);
}

fn resolve_collisions_and_joints(world: &mut World) {
    // Collect joints (if any)
    let joints = world
        .resource::<crate::world::Joints>()
        .map(|j| j.joints.clone())
        .unwrap_or_default();

    resolve_collisions_with_joints(world, &joints);
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
        // Skip sleeping bodies
        if world.get::<Sleeping>(entity).is_some() {
            continue;
        }
        let g = world.get::<Gravity>(entity).map(|g| g.0).unwrap_or(gravity);
        if let Some(vel) = world.get_mut::<Velocity>(entity) {
            vel.linear = vel.linear + g * dt;
        }
    }
}

/// Put slow bodies to sleep, wake bodies involved in collisions.
fn update_sleep_states(world: &mut World) {
    let candidates: Vec<(Entity, f32)> = {
        let query = Query::<(Entity, &PhysicsBody, &Velocity)>::new(world);
        query
            .iter()
            .filter(|(_, body, _)| body.body_type == RigidBodyType::Dynamic)
            .map(|(e, _, vel)| {
                (
                    e,
                    vel.linear.length_squared() + vel.angular.length_squared(),
                )
            })
            .collect()
    };

    for (entity, speed_sq) in candidates {
        if speed_sq < SLEEP_THRESHOLD * SLEEP_THRESHOLD {
            // Put to sleep if not already
            if world.get::<Sleeping>(entity).is_none() {
                world.insert(entity, Sleeping);
                // Zero out velocity to prevent drift
                if let Some(vel) = world.get_mut::<Velocity>(entity) {
                    vel.linear = Vec3::ZERO;
                    vel.angular = Vec3::ZERO;
                }
            }
        } else {
            // Wake up if sleeping
            world.remove::<Sleeping>(entity);
        }
    }
}

fn integrate_positions(world: &mut World, dt: f32) {
    use crate::raycast::{Ray, raycast_collider};

    // Collect movers: entity, old position, linear vel, angular vel, collider extent
    // Collider is optional — entities without colliders still move, just skip CCD.
    let movers: Vec<(Entity, Vec3, Vec3, Vec3, f32)> = {
        let query = Query::<(Entity, &PhysicsBody, &Velocity, &LocalTransform)>::new(world);
        query
            .iter()
            .filter(|(_, body, _, _)| body.body_type != RigidBodyType::Static)
            .map(|(e, _, vel, lt)| {
                let extent = world
                    .get::<Collider>(e)
                    .map(|c| shape_extent(&c.shape))
                    .unwrap_or(0.0);
                (e, lt.0.translation, vel.linear, vel.angular, extent)
            })
            .collect()
    };

    // Collect static/kinematic colliders for CCD raycasting
    let statics: Vec<(Entity, Vec3, Collider)> = {
        let query = Query::<(Entity, &LocalTransform, &Collider, &PhysicsBody)>::new(world);
        query
            .iter()
            .filter(|(_, _, _, body)| body.body_type != RigidBodyType::Dynamic)
            .map(|(e, lt, col, _)| (e, lt.0.translation, col.clone()))
            .collect()
    };

    for (entity, old_pos, linear_vel, angular_vel, extent) in movers {
        let displacement = linear_vel * dt;
        let mut new_pos = old_pos + displacement;

        // Apply angular velocity to rotation
        if angular_vel.length_squared() > 1e-8 {
            let angle = angular_vel.length() * dt;
            let axis = angular_vel * (1.0 / angular_vel.length());
            if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
                let delta_rot = euca_math::Quat::from_axis_angle(axis, angle);
                lt.0.rotation = delta_rot * lt.0.rotation;
            }
        }

        // CCD: only for Dynamic bodies (Kinematic skip collision entirely)
        let is_dynamic = world
            .get::<PhysicsBody>(entity)
            .is_some_and(|b| b.body_type == RigidBodyType::Dynamic);
        let speed = displacement.length();
        if is_dynamic && speed > extent * 0.5 && speed > 1e-6 {
            let ray = Ray::new(old_pos, displacement);
            let mut closest_t = 1.0_f32; // 1.0 = full displacement

            for (static_e, static_pos, static_col) in &statics {
                if *static_e == entity {
                    continue;
                }
                if let Some(hit) = raycast_collider(&ray, *static_pos, static_col) {
                    // hit.t is distance along ray; normalize to [0, 1] of displacement
                    let t_normalized = hit.t / speed;
                    if t_normalized < closest_t && t_normalized >= 0.0 {
                        closest_t = t_normalized;
                    }
                }
            }

            if closest_t < 1.0 {
                // Clamp position to just before the hit
                let safe_t = (closest_t - 0.01).max(0.0);
                new_pos = old_pos + displacement * safe_t;

                // Dampen velocity on impact
                if let Some(vel) = world.get_mut::<Velocity>(entity) {
                    vel.linear = vel.linear * 0.1;
                }
            }
        }

        if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
            lt.0.translation = new_pos;
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

/// Number of constraint solver iterations. More = more stable stacking.
const SOLVER_ITERATIONS: usize = 4;

fn resolve_collisions_with_joints(world: &mut World, joints: &[crate::joints::Joint]) {
    // ── Iterative constraint solver ──
    // Collect bodies once, iterate position corrections in-place,
    // write back to world at the end.

    let mut bodies: Vec<Body> = {
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

    for _iteration in 0..SOLVER_ITERATIONS {
        // Recompute broadphase each iteration (positions change)
        let candidate_pairs = broadphase_spatial_hash(&bodies);

        for (i, j) in candidate_pairs {
            if bodies[i].body_type == RigidBodyType::Static
                && bodies[j].body_type == RigidBodyType::Static
            {
                continue;
            }

            if let Some((normal, depth)) = intersect_shapes(
                bodies[i].pos,
                &bodies[i].shape,
                bodies[j].pos,
                &bodies[j].shape,
            ) {
                // Position correction in local bodies array (not world yet)
                match (bodies[i].body_type, bodies[j].body_type) {
                    (RigidBodyType::Dynamic, RigidBodyType::Static) => {
                        bodies[i].pos = bodies[i].pos + normal * (-depth);
                    }
                    (RigidBodyType::Static, RigidBodyType::Dynamic) => {
                        bodies[j].pos = bodies[j].pos + normal * depth;
                    }
                    (RigidBodyType::Dynamic, RigidBodyType::Dynamic) => {
                        bodies[i].pos = bodies[i].pos + normal * (-depth * 0.5);
                        bodies[j].pos = bodies[j].pos + normal * (depth * 0.5);
                    }
                    _ => {}
                }

                // Velocity correction (only on last iteration to avoid over-damping)
                if _iteration == SOLVER_ITERATIONS - 1 {
                    let restitution = bodies[i].restitution * bodies[j].restitution;
                    let friction = (bodies[i].friction * bodies[j].friction).sqrt();

                    if bodies[i].body_type == RigidBodyType::Dynamic
                        && let Some(vel) = world.get_mut::<Velocity>(bodies[i].entity)
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
                            vel.linear = vel.linear + n * (new_normal_vel - vn);
                            let tangent_vel = vel.linear - n * vel.linear.dot(n);
                            vel.linear = vel.linear - tangent_vel * friction;
                        }
                    }

                    if bodies[j].body_type == RigidBodyType::Dynamic
                        && let Some(vel) = world.get_mut::<Velocity>(bodies[j].entity)
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
                            vel.linear = vel.linear + n * (new_normal_vel - vn);
                            let tangent_vel = vel.linear - n * vel.linear.dot(n);
                            vel.linear = vel.linear - tangent_vel * friction;
                        }
                    }
                }
            }
        }
    }

    // ── Solve joint constraints (using body positions from the solver) ──
    if !joints.is_empty() {
        // Build entity → body index map for fast lookup
        let entity_to_idx: std::collections::HashMap<Entity, usize> = bodies
            .iter()
            .enumerate()
            .map(|(i, b)| (b.entity, i))
            .collect();

        for _iter in 0..SOLVER_ITERATIONS {
            for joint in joints {
                let idx_a = entity_to_idx.get(&joint.entity_a).copied();
                let idx_b = entity_to_idx.get(&joint.entity_b).copied();

                let (pos_a, is_a_dyn) = match idx_a {
                    Some(i) => (bodies[i].pos, bodies[i].body_type == RigidBodyType::Dynamic),
                    None => continue,
                };
                let (pos_b, is_b_dyn) = match idx_b {
                    Some(i) => (bodies[i].pos, bodies[i].body_type == RigidBodyType::Dynamic),
                    None => continue,
                };

                let (ca, cb) = joint.solve(pos_a, pos_b, is_a_dyn, is_b_dyn);

                if let Some(i) = idx_a {
                    bodies[i].pos = bodies[i].pos + ca;
                }
                if let Some(i) = idx_b {
                    bodies[i].pos = bodies[i].pos + cb;
                }
            }
        }
    }

    // Write solved positions back to world
    for body in &bodies {
        if body.body_type == RigidBodyType::Dynamic
            && let Some(lt) = world.get_mut::<LocalTransform>(body.entity)
        {
            lt.0.translation = body.pos;
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

    #[test]
    fn stacking_stability() {
        // Three cubes stacked on a static ground. With iterative solver,
        // they should settle without exploding or falling through.
        let mut world = World::new();
        world.insert_resource(PhysicsConfig::new());

        // Ground at y=0
        let ground = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(ground, GlobalTransform::default());
        world.insert(ground, PhysicsBody::fixed());
        world.insert(ground, Collider::aabb(10.0, 0.5, 10.0));

        // Stack: cube1 at y=1, cube2 at y=2, cube3 at y=3
        let mut cubes = Vec::new();
        for i in 1..=3 {
            let e = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
                0.0, i as f32, 0.0,
            ))));
            world.insert(e, GlobalTransform::default());
            world.insert(e, PhysicsBody::dynamic());
            world.insert(e, Velocity::default());
            world.insert(e, Collider::aabb(0.5, 0.5, 0.5));
            cubes.push(e);
        }

        // Run simulation for a while
        for _ in 0..300 {
            physics_step_system(&mut world);
        }

        // All cubes should be above ground (y > -0.5) and below starting height
        for (i, &cube) in cubes.iter().enumerate() {
            let y = world.get::<LocalTransform>(cube).unwrap().0.translation.y;
            assert!(y > -1.0, "Cube {} fell through ground, y={}", i, y);
            assert!(y < 5.0, "Cube {} exploded upward, y={}", i, y);
        }
    }

    #[test]
    fn ccd_prevents_tunneling() {
        // Fast bullet (speed >> collider size) aimed at a thin wall.
        // Without CCD, bullet would pass through. With CCD, it stops before.
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO, // no gravity for this test
            fixed_dt: 1.0 / 60.0,
            max_substeps: 8,
        });

        // Thin wall at x=10 (AABB half-extent 0.1 in X)
        let wall = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            10.0, 0.0, 0.0,
        ))));
        world.insert(wall, GlobalTransform::default());
        world.insert(wall, PhysicsBody::fixed());
        world.insert(wall, Collider::aabb(0.1, 2.0, 2.0));

        // Bullet at x=0, moving at 600 m/s (10 units per frame at 60fps)
        // Bullet size is 0.1 — displacement per frame (10) >> size (0.1)
        let bullet = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(bullet, GlobalTransform::default());
        world.insert(bullet, PhysicsBody::dynamic());
        world.insert(
            bullet,
            Velocity {
                linear: Vec3::new(600.0, 0.0, 0.0),
                angular: Vec3::ZERO,
            },
        );
        world.insert(bullet, Collider::sphere(0.1));

        // Run one physics step
        physics_step_system(&mut world);

        let lt = world.get::<LocalTransform>(bullet).unwrap();
        // Dynamic body with CCD should stop before the wall
        assert!(
            lt.0.translation.x < 10.0,
            "Bullet should not tunnel through wall, x={}",
            lt.0.translation.x
        );
    }
}
