use euca_ecs::{Entity, Query, Without, World};
use euca_math::{Quat, Vec3};
use euca_scene::LocalTransform;

use crate::components;
use crate::world::PhysicsWorld;

/// Main physics system: register new bodies, step simulation, write back transforms.
pub fn physics_step_system(world: &mut World) {
    register_new_bodies(world);
    step_simulation(world);
    write_back_transforms(world);
}

#[allow(clippy::type_complexity)]
fn register_new_bodies(world: &mut World) {
    let new_bodies: Vec<(
        Entity,
        components::RigidBodyType,
        Vec3,
        Option<(components::ColliderShape, f32, f32)>,
    )> = {
        let query = Query::<
            (Entity, &components::PhysicsBody, &LocalTransform),
            Without<components::PhysicsRegistered>,
        >::new(world);
        query
            .iter()
            .map(|(e, body, lt)| {
                let collider = world
                    .get::<components::PhysicsCollider>(e)
                    .map(|c| (c.shape.clone(), c.restitution, c.friction));
                (e, body.body_type, lt.0.translation, collider)
            })
            .collect()
    };

    if new_bodies.is_empty() {
        return;
    }

    {
        let physics = match world.resource_mut::<PhysicsWorld>() {
            Some(p) => p,
            None => return,
        };

        for (entity, body_type, pos, collider_info) in &new_bodies {
            let rb = match body_type {
                components::RigidBodyType::Dynamic => {
                    rapier3d::dynamics::RigidBodyBuilder::dynamic()
                }
                components::RigidBodyType::Static => rapier3d::dynamics::RigidBodyBuilder::fixed(),
                components::RigidBodyType::Kinematic => {
                    rapier3d::dynamics::RigidBodyBuilder::kinematic_position_based()
                }
            }
            .translation(rapier3d::math::Vec3::new(pos.x, pos.y, pos.z))
            .build();

            let body_handle = physics.bodies.insert(rb);
            physics.entity_to_body.insert(*entity, body_handle);

            if let Some((shape, restitution, friction)) = collider_info {
                let collider = match shape {
                    components::ColliderShape::Cuboid { hx, hy, hz } => {
                        rapier3d::geometry::ColliderBuilder::cuboid(*hx, *hy, *hz)
                    }
                    components::ColliderShape::Sphere { radius } => {
                        rapier3d::geometry::ColliderBuilder::ball(*radius)
                    }
                    components::ColliderShape::Capsule {
                        half_height,
                        radius,
                    } => rapier3d::geometry::ColliderBuilder::capsule_y(*half_height, *radius),
                }
                .restitution(*restitution)
                .friction(*friction)
                .build();

                physics
                    .colliders
                    .insert_with_parent(collider, body_handle, &mut physics.bodies);
            }
        }
    }

    for (entity, _, _, _) in new_bodies {
        world.insert(entity, components::PhysicsRegistered);
    }
}

fn step_simulation(world: &mut World) {
    if let Some(physics) = world.resource_mut::<PhysicsWorld>() {
        physics.step();
    }
}

fn write_back_transforms(world: &mut World) {
    let dynamic_entities: Vec<Entity> = {
        let query = Query::<(
            Entity,
            &components::PhysicsBody,
            &components::PhysicsRegistered,
        )>::new(world);
        query
            .iter()
            .filter(|(_, body, _)| body.body_type == components::RigidBodyType::Dynamic)
            .map(|(e, _, _)| e)
            .collect()
    };

    if dynamic_entities.is_empty() {
        return;
    }

    let updates: Vec<(Entity, Vec3, Quat)> = {
        let physics = match world.resource::<PhysicsWorld>() {
            Some(p) => p,
            None => return,
        };

        dynamic_entities
            .iter()
            .filter_map(|entity| {
                let handle = physics.entity_to_body.get(entity)?;
                let body = physics.bodies.get(*handle)?;
                let pos = body.translation();
                let rot = body.rotation();
                Some((
                    *entity,
                    Vec3::new(pos.x, pos.y, pos.z),
                    Quat(glam::Quat::from_xyzw(rot.x, rot.y, rot.z, rot.w)),
                ))
            })
            .collect()
    };

    for (entity, translation, rotation) in updates {
        if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
            lt.0.translation = translation;
            lt.0.rotation = rotation;
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
        world.insert_resource(PhysicsWorld::new());

        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 10.0, 0.0,
        ))));
        world.insert(entity, GlobalTransform::default());
        world.insert(entity, components::PhysicsBody::dynamic());
        world.insert(entity, components::PhysicsCollider::cuboid(0.5, 0.5, 0.5));

        for _ in 0..120 {
            physics_step_system(&mut world);
        }

        // After 2s of freefall from y=10, body should be well below origin
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
        world.insert_resource(PhysicsWorld::new());

        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 0.0, 0.0,
        ))));
        world.insert(entity, GlobalTransform::default());
        world.insert(entity, components::PhysicsBody::fixed());
        world.insert(entity, components::PhysicsCollider::cuboid(10.0, 0.5, 10.0));

        for _ in 0..60 {
            physics_step_system(&mut world);
        }

        let lt = world.get::<LocalTransform>(entity).unwrap();
        assert!(
            (lt.0.translation.y).abs() < 0.01,
            "Static body should not move"
        );
    }

    #[test]
    fn dynamic_body_lands_on_static() {
        let mut world = World::new();
        world.insert_resource(PhysicsWorld::new());

        // Ground
        let ground = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 0.0, 0.0,
        ))));
        world.insert(ground, GlobalTransform::default());
        world.insert(ground, components::PhysicsBody::fixed());
        world.insert(ground, components::PhysicsCollider::cuboid(10.0, 0.5, 10.0));

        // Falling cube
        let cube = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 5.0, 0.0,
        ))));
        world.insert(cube, GlobalTransform::default());
        world.insert(cube, components::PhysicsBody::dynamic());
        world.insert(cube, components::PhysicsCollider::cuboid(0.5, 0.5, 0.5));

        for _ in 0..180 {
            physics_step_system(&mut world);
        }

        let lt = world.get::<LocalTransform>(cube).unwrap();
        assert!(lt.0.translation.y > 0.0, "Cube should be above ground");
        assert!(
            lt.0.translation.y < 3.0,
            "Cube should have fallen, y={}",
            lt.0.translation.y
        );
    }
}
