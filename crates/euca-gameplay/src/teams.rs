//! Teams and spawn points.
//!
//! Components: `Team`, `SpawnPoint`.
//! Systems: `respawn_system`.

use euca_ecs::{Entity, Events, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;

use crate::cleanup::CorpseTimer;
use crate::combat::{CurrentTarget, EntityRole};
use crate::health::{Dead, DeathEvent, Health, LastAttacker};

/// Which team this entity belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Team(pub u8);

/// Marks an entity as a spawn location for a specific team.
#[derive(Clone, Debug)]
pub struct SpawnPoint {
    pub team: u8,
}

/// Tracks respawn countdown for dead entities.
#[derive(Clone, Debug)]
pub struct RespawnTimer {
    pub remaining: f32,
}

/// Respawn dead entities at their team's spawn point after a delay.
///
/// Each frame: tick RespawnTimer. When it reaches 0, teleport to a spawn
/// point matching the entity's Team, restore Health, remove Dead + timer.
pub fn respawn_system(world: &mut World, dt: f32) {
    // Tick respawn timers
    let ready: Vec<Entity> = {
        let query = Query::<(Entity, &mut RespawnTimer)>::new(world);
        let mut ready = Vec::new();
        for (entity, timer) in query.iter() {
            // We need to mutate through a collected approach
            let _ = timer; // can't mutate during iteration without mut query
            ready.push(entity);
        }
        ready
    };

    // Update timers and collect those ready to respawn
    let mut to_respawn: Vec<(Entity, u8)> = Vec::new();
    for entity in ready {
        if let Some(timer) = world.get_mut::<RespawnTimer>(entity) {
            timer.remaining -= dt;
            if timer.remaining <= 0.0 {
                let team = world.get::<Team>(entity).map(|t| t.0).unwrap_or(0);
                to_respawn.push((entity, team));
            }
        }
    }

    // Find spawn points per team
    let spawn_positions: Vec<(u8, Vec3)> = {
        let query = Query::<(&SpawnPoint, &LocalTransform)>::new(world);
        query
            .iter()
            .map(|(sp, lt)| (sp.team, lt.0.translation))
            .collect()
    };

    // Respawn
    for (entity, team) in to_respawn {
        // Find a spawn point for this team
        let spawn_pos = spawn_positions
            .iter()
            .find(|(t, _)| *t == team)
            .map(|(_, pos)| *pos)
            .unwrap_or(Vec3::new(0.0, 2.0, 0.0));

        // Teleport to spawn
        if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
            lt.0.translation = spawn_pos;
        }
        // Restore health
        if let Some(health) = world.get_mut::<Health>(entity) {
            health.current = health.max;
        }
        // Remove Dead, RespawnTimer, and stale combat state so that
        // enemies don't continue attacking this entity based on pre-death
        // targeting, and damage attribution doesn't carry across lives.
        world.remove::<Dead>(entity);
        world.remove::<RespawnTimer>(entity);
        world.remove::<CurrentTarget>(entity);
        world.remove::<LastAttacker>(entity);
    }
}

/// Default corpse duration for minions (seconds).
const MINION_CORPSE_DURATION: f32 = 2.0;

/// When an entity dies, start a respawn timer or corpse timer depending on role.
///
/// - **Minions** get a `CorpseTimer` (they are despawned after a short delay).
/// - **All other entities with a Team** get a `RespawnTimer` (they respawn).
pub fn start_respawn_on_death(world: &mut World, respawn_delay: f32) {
    let deaths: Vec<Entity> = world
        .resource::<Events>()
        .map(|e| e.read::<DeathEvent>().map(|d| d.entity).collect())
        .unwrap_or_default();

    for entity in deaths {
        if world.get::<Team>(entity).is_none() {
            continue;
        }

        let is_minion = world
            .get::<EntityRole>(entity)
            .map(|r| *r == EntityRole::Minion)
            .unwrap_or(false);

        if is_minion {
            if world.get::<CorpseTimer>(entity).is_none() {
                world.insert(entity, CorpseTimer::new(MINION_CORPSE_DURATION));
            }
        } else if world.get::<RespawnTimer>(entity).is_none() {
            world.insert(
                entity,
                RespawnTimer {
                    remaining: respawn_delay,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    #[test]
    fn team_creation() {
        let t = Team(1);
        assert_eq!(t.0, 1);
    }

    #[test]
    fn respawn_restores_health_and_position() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        // Create spawn point
        let _sp = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            10.0, 0.0, 10.0,
        ))));
        world.insert(_sp, SpawnPoint { team: 1 });

        // Create dead entity with team
        let entity = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.insert(entity, Team(1));
        world.insert(entity, Dead);
        world.insert(entity, RespawnTimer { remaining: 0.0 }); // ready immediately
        world.insert(
            entity,
            LocalTransform(Transform::from_translation(Vec3::ZERO)),
        );

        respawn_system(&mut world, 0.016);

        // Health restored
        assert_eq!(world.get::<Health>(entity).unwrap().current, 100.0);
        // Dead marker removed
        assert!(world.get::<Dead>(entity).is_none());
        // Teleported to spawn point
        let pos = world.get::<LocalTransform>(entity).unwrap().0.translation;
        assert_eq!(pos.x, 10.0);
        assert_eq!(pos.z, 10.0);
    }

    #[test]
    fn respawn_timer_counts_down() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.insert(entity, Team(1));
        world.insert(entity, Dead);
        world.insert(entity, RespawnTimer { remaining: 1.0 });
        world.insert(
            entity,
            LocalTransform(Transform::from_translation(Vec3::ZERO)),
        );

        // Not ready yet (0.5s remaining)
        respawn_system(&mut world, 0.5);
        assert!(world.get::<Dead>(entity).is_some());

        // Still dead (timer at 0.5)
        respawn_system(&mut world, 0.4);
        assert!(world.get::<Dead>(entity).is_some());
    }
}
