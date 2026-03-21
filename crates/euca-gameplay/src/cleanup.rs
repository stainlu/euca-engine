//! Corpse cleanup — despawn dead entities that don't respawn.
//!
//! Components: `CorpseTimer`.
//! Systems: `corpse_cleanup_system`.

use euca_ecs::{Entity, Query, World};

use crate::health::Dead;
use crate::teams::RespawnTimer;

/// Tracks how long a dead entity has existed as a corpse.
/// When `elapsed >= duration`, the entity is despawned.
#[derive(Clone, Debug)]
pub struct CorpseTimer {
    pub elapsed: f32,
    pub duration: f32,
}

impl CorpseTimer {
    pub fn new(duration: f32) -> Self {
        Self {
            elapsed: 0.0,
            duration,
        }
    }
}

/// Tick corpse timers on dead entities and despawn them when expired.
///
/// Only processes entities that have both `Dead` and `CorpseTimer` but
/// **not** `RespawnTimer` (heroes respawn instead of being cleaned up).
pub fn corpse_cleanup_system(world: &mut World, dt: f32) {
    // Collect candidate entities: Dead + CorpseTimer, no RespawnTimer.
    let candidates: Vec<Entity> = {
        let query = Query::<(Entity, &Dead, &CorpseTimer)>::new(world);
        query
            .iter()
            .filter(|(e, _, _)| world.get::<RespawnTimer>(*e).is_none())
            .map(|(e, _, _)| e)
            .collect()
    };

    let mut to_despawn: Vec<Entity> = Vec::new();

    for entity in candidates {
        if let Some(timer) = world.get_mut::<CorpseTimer>(entity) {
            timer.elapsed += dt;
            if timer.elapsed >= timer.duration {
                to_despawn.push(entity);
            }
        }
    }

    for entity in to_despawn {
        world.despawn(entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combat::EntityRole;
    use crate::health::{Dead, Health, death_check_system};
    use crate::teams::{RespawnTimer, Team, start_respawn_on_death};
    use euca_ecs::Events;

    /// Helper: create a world with Events resource.
    fn setup_world() -> World {
        let mut world = World::new();
        world.insert_resource(Events::default());
        world
    }

    #[test]
    fn minion_despawns_after_corpse_timer() {
        let mut world = setup_world();

        // Create a dead minion with a 2-second corpse timer.
        let minion = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.insert(minion, Team(1));
        world.insert(minion, EntityRole::Minion);
        world.insert(minion, Dead);
        world.insert(minion, CorpseTimer::new(2.0));

        // Tick 1 second — should still be alive.
        corpse_cleanup_system(&mut world, 1.0);
        assert!(world.is_alive(minion), "minion should still exist at 1.0s");

        // Tick another 1 second — timer reaches 2.0, should despawn.
        corpse_cleanup_system(&mut world, 1.0);
        assert!(
            !world.is_alive(minion),
            "minion should be despawned after 2.0s"
        );
    }

    #[test]
    fn hero_does_not_get_corpse_timer() {
        let mut world = setup_world();

        // Kill a hero — trigger death + start_respawn_on_death.
        let hero = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.insert(hero, Team(1));
        world.insert(hero, EntityRole::Hero);

        death_check_system(&mut world);
        start_respawn_on_death(&mut world, 5.0);

        // Hero should have RespawnTimer, NOT CorpseTimer.
        assert!(
            world.get::<RespawnTimer>(hero).is_some(),
            "hero should get RespawnTimer"
        );
        assert!(
            world.get::<CorpseTimer>(hero).is_none(),
            "hero should NOT get CorpseTimer"
        );
    }

    #[test]
    fn corpse_timer_ticks_correctly() {
        let mut world = setup_world();

        let minion = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.insert(minion, Dead);
        world.insert(minion, CorpseTimer::new(3.0));

        // Tick 0.5s
        corpse_cleanup_system(&mut world, 0.5);
        let timer = world.get::<CorpseTimer>(minion).unwrap();
        assert!(
            (timer.elapsed - 0.5).abs() < f32::EPSILON,
            "elapsed should be 0.5"
        );

        // Tick another 1.0s
        corpse_cleanup_system(&mut world, 1.0);
        let timer = world.get::<CorpseTimer>(minion).unwrap();
        assert!(
            (timer.elapsed - 1.5).abs() < f32::EPSILON,
            "elapsed should be 1.5"
        );

        // Still alive
        assert!(world.is_alive(minion));
    }

    #[test]
    fn entity_with_respawn_timer_not_despawned() {
        let mut world = setup_world();

        // Entity that has both Dead and CorpseTimer AND RespawnTimer
        // (edge case: should NOT be despawned because RespawnTimer takes priority).
        let entity = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.insert(entity, Dead);
        world.insert(entity, CorpseTimer::new(0.0)); // would despawn immediately
        world.insert(entity, RespawnTimer { remaining: 5.0 });

        corpse_cleanup_system(&mut world, 1.0);
        assert!(
            world.is_alive(entity),
            "entity with RespawnTimer should not be despawned by corpse cleanup"
        );
    }

    #[test]
    fn minion_death_adds_corpse_timer_via_system() {
        let mut world = setup_world();

        // Create a minion that is about to die.
        let minion = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.insert(minion, Team(2));
        world.insert(minion, EntityRole::Minion);

        // Run death check — this marks Dead and emits DeathEvent.
        death_check_system(&mut world);
        assert!(world.get::<Dead>(minion).is_some());

        // Run start_respawn_on_death — minions should get CorpseTimer, not RespawnTimer.
        start_respawn_on_death(&mut world, 5.0);
        assert!(
            world.get::<CorpseTimer>(minion).is_some(),
            "minion should get CorpseTimer on death"
        );
        assert!(
            world.get::<RespawnTimer>(minion).is_none(),
            "minion should NOT get RespawnTimer"
        );
    }
}
