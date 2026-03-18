//! Health and damage — the foundation of combat.
//!
//! Components: `Health`, `Dead` (marker).
//! Events: `DamageEvent`, `DeathEvent`.
//! Systems: `apply_damage_system`, `death_check_system`.

use euca_ecs::{Entity, Events, Query, World};

/// Entity has hit points that can be reduced by damage or restored by healing.
#[derive(Clone, Debug)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}

impl Health {
    pub fn new(max: f32) -> Self {
        Self { current: max, max }
    }

    pub fn is_dead(&self) -> bool {
        self.current <= 0.0
    }

    pub fn fraction(&self) -> f32 {
        if self.max > 0.0 {
            (self.current / self.max).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Marker component: entity has died (health reached 0).
/// Stays until respawn or despawn.
#[derive(Clone, Copy, Debug)]
pub struct Dead;

/// Request to apply damage to an entity.
#[derive(Clone, Debug)]
pub struct DamageEvent {
    pub target: Entity,
    pub amount: f32,
    pub source: Option<Entity>,
}

/// Notification that an entity has died.
#[derive(Clone, Debug)]
pub struct DeathEvent {
    pub entity: Entity,
    pub killer: Option<Entity>,
}

/// Apply pending damage events to Health components.
pub fn apply_damage_system(world: &mut World) {
    // Collect events first to avoid borrow conflicts
    let events: Vec<DamageEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<DamageEvent>().cloned().collect())
        .unwrap_or_default();

    for event in events {
        if let Some(health) = world.get_mut::<Health>(event.target) {
            health.current = (health.current - event.amount).max(0.0);
        }
    }
}

/// Check for entities with Health <= 0 that aren't already Dead.
/// Adds the Dead marker and emits DeathEvent.
pub fn death_check_system(world: &mut World) {
    // Find entities that just died
    let newly_dead: Vec<(Entity, Option<Entity>)> = {
        let query = Query::<(Entity, &Health)>::new(world);
        query
            .iter()
            .filter(|(e, h)| h.is_dead() && world.get::<Dead>(*e).is_none())
            .map(|(e, _)| (e, None)) // killer tracking via last DamageEvent would need more state
            .collect()
    };

    for (entity, killer) in &newly_dead {
        world.insert(*entity, Dead);
        if let Some(events) = world.resource_mut::<Events>() {
            events.send(DeathEvent {
                entity: *entity,
                killer: *killer,
            });
        }
    }
}

/// Apply healing to an entity's Health (clamped to max).
pub fn heal(world: &mut World, entity: Entity, amount: f32) {
    if let Some(health) = world.get_mut::<Health>(entity) {
        health.current = (health.current + amount).min(health.max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_creation() {
        let h = Health::new(100.0);
        assert_eq!(h.current, 100.0);
        assert_eq!(h.max, 100.0);
        assert!(!h.is_dead());
        assert_eq!(h.fraction(), 1.0);
    }

    #[test]
    fn health_dead_at_zero() {
        let h = Health {
            current: 0.0,
            max: 100.0,
        };
        assert!(h.is_dead());
        assert_eq!(h.fraction(), 0.0);
    }

    #[test]
    fn apply_damage_reduces_health() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(100.0));

        // Send damage event
        world.resource_mut::<Events>().unwrap().send(DamageEvent {
            target: entity,
            amount: 30.0,
            source: None,
        });

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 70.0);
    }

    #[test]
    fn damage_cannot_go_below_zero() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(50.0));

        world.resource_mut::<Events>().unwrap().send(DamageEvent {
            target: entity,
            amount: 999.0,
            source: None,
        });

        apply_damage_system(&mut world);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 0.0);
    }

    #[test]
    fn death_check_marks_dead() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });

        assert!(world.get::<Dead>(entity).is_none());

        death_check_system(&mut world);

        assert!(world.get::<Dead>(entity).is_some());
    }

    #[test]
    fn death_check_ignores_alive() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health::new(50.0));

        death_check_system(&mut world);

        assert!(world.get::<Dead>(entity).is_none());
    }

    #[test]
    fn death_check_does_not_re_mark() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.insert(entity, Dead);

        // Should not panic or double-mark
        death_check_system(&mut world);
        assert!(world.get::<Dead>(entity).is_some());
    }

    #[test]
    fn heal_restores_health() {
        let mut world = World::new();
        let entity = world.spawn(Health {
            current: 30.0,
            max: 100.0,
        });

        heal(&mut world, entity, 50.0);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 80.0);
    }

    #[test]
    fn heal_capped_at_max() {
        let mut world = World::new();
        let entity = world.spawn(Health {
            current: 90.0,
            max: 100.0,
        });

        heal(&mut world, entity, 50.0);

        let health = world.get::<Health>(entity).unwrap();
        assert_eq!(health.current, 100.0);
    }
}
