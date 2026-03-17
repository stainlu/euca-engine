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
        if let Some(vel) = world.get_mut::<Velocity>(entity) {
            // Check for per-entity gravity override
            let g = gravity; // TODO: check Gravity component
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

fn resolve_collisions(world: &mut World) {
    // Collect all collidable entities
    let bodies: Vec<(Entity, Vec3, ColliderShape, RigidBodyType, f32, f32)> = {
        let query = Query::<(Entity, &LocalTransform, &Collider)>::new(world);
        query
            .iter()
            .filter_map(|(e, lt, col)| {
                let body = world.get::<PhysicsBody>(e)?;
                Some((
                    e,
                    lt.0.translation,
                    col.shape.clone(),
                    body.body_type,
                    col.restitution,
                    col.friction,
                ))
            })
            .collect()
    };

    // O(n²) broadphase
    let mut corrections: Vec<(Entity, Vec3, Vec3)> = Vec::new();

    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let (e_a, pos_a, shape_a, type_a, rest_a, fric_a) = &bodies[i];
            let (e_b, pos_b, shape_b, type_b, rest_b, fric_b) = &bodies[j];

            if *type_a == RigidBodyType::Static && *type_b == RigidBodyType::Static {
                continue;
            }

            if let Some((normal, depth)) = intersect_shapes(*pos_a, shape_a, *pos_b, shape_b) {
                let restitution = (rest_a + rest_b) * 0.5;
                let friction = (fric_a + fric_b) * 0.5;

                // Push-out resolution
                match (type_a, type_b) {
                    (RigidBodyType::Dynamic, RigidBodyType::Static) => {
                        corrections.push((*e_a, normal * (-depth), Vec3::ZERO));
                    }
                    (RigidBodyType::Static, RigidBodyType::Dynamic) => {
                        corrections.push((*e_b, normal * depth, Vec3::ZERO));
                    }
                    (RigidBodyType::Dynamic, RigidBodyType::Dynamic) => {
                        corrections.push((*e_a, normal * (-depth * 0.5), Vec3::ZERO));
                        corrections.push((*e_b, normal * (depth * 0.5), Vec3::ZERO));
                    }
                    _ => {}
                }

                // Velocity response for dynamic body A
                if *type_a == RigidBodyType::Dynamic
                    && let Some(vel) = world.get::<Velocity>(*e_a)
                {
                    let n = normal * (-1.0); // normal pointing away from A
                    let vn = vel.linear.dot(n); // normal component (negative = approaching)
                    if vn < 0.0 {
                        let approach_speed = -vn;
                        // Bounce or rest
                        let new_normal_vel = if approach_speed < REST_VELOCITY_THRESHOLD {
                            0.0 // come to rest
                        } else {
                            approach_speed * restitution // bounce with energy loss
                        };
                        let normal_correction = n * (new_normal_vel - vn);
                        // Friction: damp tangential velocity
                        let tangent_vel = vel.linear - n * vn;
                        let friction_correction = tangent_vel * (-friction);
                        corrections.push((
                            *e_a,
                            Vec3::ZERO,
                            normal_correction + friction_correction,
                        ));
                    }
                }

                // Velocity response for dynamic body B
                if *type_b == RigidBodyType::Dynamic
                    && let Some(vel) = world.get::<Velocity>(*e_b)
                {
                    let n = normal; // normal already points away from B toward A
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
                            *e_b,
                            Vec3::ZERO,
                            normal_correction + friction_correction,
                        ));
                    }
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
