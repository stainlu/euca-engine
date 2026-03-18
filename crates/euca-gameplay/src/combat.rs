//! Projectiles — moving damaging entities.
//!
//! Components: `Projectile`.
//! Systems: `projectile_system`.

use euca_ecs::{Entity, Events, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;

use crate::health::DamageEvent;

/// Entity that moves in a direction and damages what it hits.
#[derive(Clone, Debug)]
pub struct Projectile {
    pub direction: Vec3,
    pub speed: f32,
    pub damage: f32,
    pub lifetime: f32,
    pub owner: Entity,
    pub elapsed: f32,
}

impl Projectile {
    pub fn new(direction: Vec3, speed: f32, damage: f32, lifetime: f32, owner: Entity) -> Self {
        Self {
            direction: direction.normalize(),
            speed,
            damage,
            lifetime,
            owner,
            elapsed: 0.0,
        }
    }
}

/// Move projectiles, check lifetime, check collision with Health entities.
pub fn projectile_system(world: &mut World, dt: f32) {
    // Move projectiles and collect expired/hit
    let mut to_despawn: Vec<Entity> = Vec::new();
    let mut damage_events: Vec<DamageEvent> = Vec::new();

    // Collect projectile data
    let projectiles: Vec<(Entity, Vec3, f32, f32, Entity, Vec3)> = {
        let query = Query::<(Entity, &Projectile, &LocalTransform)>::new(world);
        query
            .iter()
            .map(|(e, p, lt)| (e, p.direction, p.speed, p.damage, p.owner, lt.0.translation))
            .collect()
    };

    // Collect potential targets (entities with Health and a position)
    let targets: Vec<(Entity, Vec3)> = {
        let query = Query::<(Entity, &LocalTransform)>::new(world);
        query
            .iter()
            .filter(|(e, _)| world.get::<crate::health::Health>(*e).is_some())
            .map(|(e, lt)| (e, lt.0.translation))
            .collect()
    };

    for (proj_entity, direction, speed, damage, owner, pos) in &projectiles {
        let new_pos = *pos + *direction * (*speed * dt);

        // Update position
        if let Some(lt) = world.get_mut::<LocalTransform>(*proj_entity) {
            lt.0.translation = new_pos;
        }

        // Tick lifetime
        if let Some(proj) = world.get_mut::<Projectile>(*proj_entity) {
            proj.elapsed += dt;
            if proj.elapsed >= proj.lifetime {
                to_despawn.push(*proj_entity);
                continue;
            }
        }

        // Simple sphere collision with targets (radius 0.5)
        let hit_radius = 0.5;
        for (target_entity, target_pos) in &targets {
            if *target_entity == *owner || *target_entity == *proj_entity {
                continue;
            }
            let dist = (new_pos - *target_pos).length();
            if dist < hit_radius {
                damage_events.push(DamageEvent {
                    target: *target_entity,
                    amount: *damage,
                    source: Some(*owner),
                });
                to_despawn.push(*proj_entity);
                break;
            }
        }
    }

    // Emit damage events
    if let Some(events) = world.resource_mut::<Events>() {
        for event in damage_events {
            events.send(event);
        }
    }

    // Despawn expired/hit projectiles
    for entity in to_despawn {
        world.despawn(entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::Health;
    use euca_math::Transform;

    #[test]
    fn projectile_moves_forward() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let owner = world.spawn(Health::new(100.0));

        let proj = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(
            proj,
            Projectile::new(Vec3::new(1.0, 0.0, 0.0), 10.0, 25.0, 5.0, owner),
        );

        projectile_system(&mut world, 1.0);

        let pos = world.get::<LocalTransform>(proj).unwrap().0.translation;
        assert!((pos.x - 10.0).abs() < 0.01);
    }

    #[test]
    fn projectile_despawns_on_lifetime() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let owner = world.spawn(Health::new(100.0));

        let proj = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(
            proj,
            Projectile::new(Vec3::new(1.0, 0.0, 0.0), 10.0, 25.0, 0.5, owner),
        );

        projectile_system(&mut world, 1.0); // elapsed > lifetime

        assert!(!world.is_alive(proj));
    }

    #[test]
    fn projectile_damages_on_hit() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let owner = world.spawn(Health::new(100.0));
        world.insert(
            owner,
            LocalTransform(Transform::from_translation(Vec3::new(-5.0, 0.0, 0.0))),
        );

        let target = world.spawn(Health::new(100.0));
        world.insert(
            target,
            LocalTransform(Transform::from_translation(Vec3::new(0.3, 0.0, 0.0))),
        );

        let proj = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(
            proj,
            Projectile::new(Vec3::new(1.0, 0.0, 0.0), 10.0, 50.0, 5.0, owner),
        );

        // Move projectile close to target
        projectile_system(&mut world, 0.01);

        // Check DamageEvent was emitted
        let events = world.resource::<Events>().unwrap();
        let damage_events: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(damage_events.len(), 1);
        assert_eq!(damage_events[0].target.index(), target.index());
        assert_eq!(damage_events[0].amount, 50.0);
    }
}
