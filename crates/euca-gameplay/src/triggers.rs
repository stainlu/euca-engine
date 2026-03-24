//! Trigger zones — area-based events.
//!
//! Components: `TriggerZone`.
//! Systems: `trigger_system`.

use euca_ecs::{Entity, Events, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;
use std::collections::HashSet;

use crate::health::DamageEvent;

/// What happens when an entity enters a trigger zone.
#[derive(Clone, Debug)]
pub enum TriggerAction {
    Damage { amount: f32 },
    Heal { amount: f32 },
    Teleport { destination: Vec3 },
}

/// Entity that fires an action when other entities enter its area.
#[derive(Clone, Debug)]
pub struct TriggerZone {
    pub half_extents: Vec3,
    pub action: TriggerAction,
    pub once: bool,
    pub triggered: HashSet<u32>, // entity indices that already triggered (for once mode)
}

impl TriggerZone {
    pub fn new(half_extents: Vec3, action: TriggerAction) -> Self {
        Self {
            half_extents,
            action,
            once: false,
            triggered: HashSet::new(),
        }
    }

    pub fn once(mut self) -> Self {
        self.once = true;
        self
    }
}

/// Check if entities overlap with trigger zones and execute actions.
pub fn trigger_system(world: &mut World) {
    // Collect trigger zones
    let triggers: Vec<(Entity, Vec3, Vec3, TriggerAction, bool, HashSet<u32>)> = {
        let query = Query::<(Entity, &TriggerZone, &LocalTransform)>::new(world);
        query
            .iter()
            .map(|(e, tz, lt)| {
                (
                    e,
                    lt.0.translation,
                    tz.half_extents,
                    tz.action.clone(),
                    tz.once,
                    tz.triggered.clone(),
                )
            })
            .collect()
    };

    // Collect all positioned entities (potential targets)
    let entities: Vec<(Entity, Vec3)> = {
        let query = Query::<(Entity, &LocalTransform)>::new(world);
        query.iter().map(|(e, lt)| (e, lt.0.translation)).collect()
    };

    let mut damage_events: Vec<DamageEvent> = Vec::new();
    let mut heals: Vec<(Entity, f32)> = Vec::new();
    let mut teleports: Vec<(Entity, Vec3)> = Vec::new();
    let mut trigger_updates: Vec<(Entity, u32)> = Vec::new(); // (zone, triggered entity idx)

    for (zone_entity, zone_pos, half, action, once, triggered) in &triggers {
        for (entity, entity_pos) in &entities {
            if *entity == *zone_entity {
                continue;
            }
            if *once && triggered.contains(&entity.index()) {
                continue;
            }

            // AABB overlap test
            let diff = *entity_pos - *zone_pos;
            if diff.x.abs() < half.x && diff.y.abs() < half.y && diff.z.abs() < half.z {
                match action {
                    TriggerAction::Damage { amount } => {
                        damage_events.push(DamageEvent::new(*entity, *amount, Some(*zone_entity)));
                    }
                    TriggerAction::Heal { amount } => {
                        heals.push((*entity, *amount));
                    }
                    TriggerAction::Teleport { destination } => {
                        teleports.push((*entity, *destination));
                    }
                }
                if *once {
                    trigger_updates.push((*zone_entity, entity.index()));
                }
            }
        }
    }

    // Apply effects
    if let Some(events) = world.resource_mut::<Events>() {
        for event in damage_events {
            events.send(event);
        }
    }
    for (entity, amount) in heals {
        crate::health::heal(world, entity, amount);
    }
    for (entity, dest) in teleports {
        if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
            lt.0.translation = dest;
        }
    }
    // Mark triggered entities in once-mode zones
    for (zone, idx) in trigger_updates {
        if let Some(tz) = world.get_mut::<TriggerZone>(zone) {
            tz.triggered.insert(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::Health;
    use euca_math::Transform;

    #[test]
    fn trigger_damages_overlapping_entity() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        // Trigger zone at origin
        let zone = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(
            zone,
            TriggerZone::new(
                Vec3::new(2.0, 2.0, 2.0),
                TriggerAction::Damage { amount: 10.0 },
            ),
        );

        // Entity inside the zone
        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.5, 0.0, 0.0,
        ))));
        world.insert(entity, Health::new(100.0));

        trigger_system(&mut world);

        let events = world.resource::<Events>().unwrap();
        assert_eq!(events.read::<DamageEvent>().count(), 1);
    }

    #[test]
    fn trigger_ignores_outside_entity() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let zone = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(
            zone,
            TriggerZone::new(
                Vec3::new(1.0, 1.0, 1.0),
                TriggerAction::Damage { amount: 10.0 },
            ),
        );

        // Entity far outside
        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            10.0, 0.0, 0.0,
        ))));
        world.insert(entity, Health::new(100.0));

        trigger_system(&mut world);

        let events = world.resource::<Events>().unwrap();
        assert_eq!(events.read::<DamageEvent>().count(), 0);
    }

    #[test]
    fn trigger_once_does_not_repeat() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let zone = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(
            zone,
            TriggerZone::new(
                Vec3::new(2.0, 2.0, 2.0),
                TriggerAction::Heal { amount: 25.0 },
            )
            .once(),
        );

        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.5, 0.0, 0.0,
        ))));
        world.insert(
            entity,
            Health {
                current: 50.0,
                max: 100.0,
            },
        );

        trigger_system(&mut world);
        assert_eq!(world.get::<Health>(entity).unwrap().current, 75.0);

        // Second tick — should NOT heal again
        trigger_system(&mut world);
        assert_eq!(world.get::<Health>(entity).unwrap().current, 75.0);
    }

    #[test]
    fn trigger_teleports_entity() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let dest = Vec3::new(100.0, 0.0, 100.0);
        let zone = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(
            zone,
            TriggerZone::new(
                Vec3::new(2.0, 2.0, 2.0),
                TriggerAction::Teleport { destination: dest },
            ),
        );

        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.5, 0.0, 0.0,
        ))));

        trigger_system(&mut world);

        let pos = world.get::<LocalTransform>(entity).unwrap().0.translation;
        assert_eq!(pos.x, 100.0);
        assert_eq!(pos.z, 100.0);
    }
}
