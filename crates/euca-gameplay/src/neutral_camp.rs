//! Jungle neutral camp monsters with leash behavior.
//!
//! Neutral camps are idle until attacked. When aggro'd, they chase the attacker.
//! If the attacker dies or the neutral wanders too far from home, it leashes
//! back — returning to its home position and healing to full HP.
//!
//! Components: `NeutralCamp`.
//! Systems: `neutral_camp_system`.

use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_physics::Velocity;
use euca_scene::LocalTransform;

use crate::combat::AutoCombat;
use crate::health::{Dead, Health, LastAttacker};

/// Default chase speed when the entity has no `AutoCombat` component.
const DEFAULT_CHASE_SPEED: f32 = 3.0;

/// Distance threshold for snapping to home position during leash.
const LEASH_SNAP_DISTANCE: f32 = 0.5;

/// Marks an entity as a neutral camp monster with leash behavior.
#[derive(Clone, Debug)]
pub struct NeutralCamp {
    /// Home position — returns here when leashing.
    pub home: Vec3,
    /// Max chase distance from home before leashing back.
    pub leash_range: f32,
}

/// Per-entity snapshot taken at the start of the tick, so we can release the
/// borrow on `World` before mutating components.
struct NeutralSnapshot {
    entity: Entity,
    pos: Vec3,
    home: Vec3,
    leash_range: f32,
    speed: f32,
    attacker: Option<Entity>,
}

/// Drive neutral camp AI: idle at home, chase attacker on aggro, leash back when
/// the attacker dies or the neutral strays too far from home.
pub fn neutral_camp_system(world: &mut World, _dt: f32) {
    let neutrals: Vec<NeutralSnapshot> = {
        let query = Query::<(Entity, &NeutralCamp, &LocalTransform)>::new(world);
        query
            .iter()
            .filter(|(e, _, _)| world.get::<Dead>(*e).is_none())
            .map(|(e, camp, lt)| {
                let speed = world
                    .get::<AutoCombat>(e)
                    .map(|ac| ac.speed)
                    .unwrap_or(DEFAULT_CHASE_SPEED);
                let attacker = world.get::<LastAttacker>(e).and_then(|la| la.0);
                NeutralSnapshot {
                    entity: e,
                    pos: lt.0.translation,
                    home: camp.home,
                    leash_range: camp.leash_range,
                    speed,
                    attacker,
                }
            })
            .collect()
    };

    for neutral in &neutrals {
        match neutral.attacker {
            Some(attacker) => {
                let attacker_alive = world.get::<Dead>(attacker).is_none()
                    && world.get::<Health>(attacker).is_some();
                let dist_from_home = (neutral.pos - neutral.home).length();

                if attacker_alive && dist_from_home < neutral.leash_range {
                    // Chase: set velocity toward attacker.
                    if let Some(attacker_lt) = world.get::<LocalTransform>(attacker) {
                        let target_pos = attacker_lt.0.translation;
                        let dir = target_pos - neutral.pos;
                        let len = dir.length();
                        if len > 0.01 {
                            let normalized = dir * (1.0 / len);
                            if let Some(vel) = world.get_mut::<Velocity>(neutral.entity) {
                                vel.linear.x = normalized.x * neutral.speed;
                                vel.linear.z = normalized.z * neutral.speed;
                            }
                        }
                    }
                } else {
                    // Leash: move back toward home. Heal + clear aggro when arrived.
                    leash_toward_home(world, neutral);
                }
            }
            None => {
                // No aggro: stay idle.
                if let Some(vel) = world.get_mut::<Velocity>(neutral.entity) {
                    vel.linear.x = 0.0;
                    vel.linear.z = 0.0;
                }
            }
        }
    }
}

/// Move the neutral toward its home position. When close enough, snap to home,
/// heal to full HP, and clear the `LastAttacker` component.
fn leash_toward_home(world: &mut World, neutral: &NeutralSnapshot) {
    let to_home = neutral.home - neutral.pos;
    let dist = to_home.length();

    if dist < LEASH_SNAP_DISTANCE {
        // Arrived at home — snap position, stop, heal, clear aggro.
        if let Some(lt) = world.get_mut::<LocalTransform>(neutral.entity) {
            lt.0.translation = neutral.home;
        }
        if let Some(vel) = world.get_mut::<Velocity>(neutral.entity) {
            vel.linear.x = 0.0;
            vel.linear.z = 0.0;
        }
        if let Some(health) = world.get_mut::<Health>(neutral.entity) {
            health.current = health.max;
        }
        world.insert(neutral.entity, LastAttacker(None));
    } else {
        // Still walking home.
        let dir = to_home * (1.0 / dist);
        if let Some(vel) = world.get_mut::<Velocity>(neutral.entity) {
            vel.linear.x = dir.x * neutral.speed;
            vel.linear.z = dir.z * neutral.speed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    /// Helper: spawn a neutral camp entity with standard components.
    fn spawn_neutral(world: &mut World, pos: Vec3, home: Vec3, leash_range: f32) -> Entity {
        let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(e, NeutralCamp { home, leash_range });
        world.insert(e, Health::new(200.0));
        world.insert(e, Velocity::default());
        e
    }

    /// Helper: spawn a simple attacker entity with position and health.
    fn spawn_attacker(world: &mut World, pos: Vec3) -> Entity {
        let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(e, Health::new(100.0));
        world.insert(e, Velocity::default());
        e
    }

    #[test]
    fn neutral_chases_attacker_when_hit() {
        let mut world = World::new();

        let home = Vec3::new(10.0, 0.0, 10.0);
        let neutral = spawn_neutral(&mut world, home, home, 15.0);

        // Attacker is to the right of the neutral.
        let attacker = spawn_attacker(&mut world, Vec3::new(15.0, 0.0, 10.0));
        world.insert(neutral, LastAttacker(Some(attacker)));

        neutral_camp_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(neutral).unwrap();
        assert!(
            vel.linear.x > 0.0,
            "neutral should chase toward attacker (+X direction)"
        );
        assert!(
            vel.linear.z.abs() < 0.01,
            "no Z movement expected when attacker is directly to the right"
        );
    }

    #[test]
    fn neutral_leashes_when_attacker_beyond_range() {
        let mut world = World::new();

        let home = Vec3::new(10.0, 0.0, 10.0);
        // Neutral has wandered far from home.
        let neutral_pos = Vec3::new(30.0, 0.0, 10.0);
        let neutral = spawn_neutral(&mut world, neutral_pos, home, 15.0);

        // Attacker is even further away — but the neutral is beyond leash range from home.
        let attacker = spawn_attacker(&mut world, Vec3::new(35.0, 0.0, 10.0));
        world.insert(neutral, LastAttacker(Some(attacker)));

        neutral_camp_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(neutral).unwrap();
        assert!(
            vel.linear.x < 0.0,
            "neutral should move back toward home (-X direction)"
        );
    }

    #[test]
    fn neutral_heals_to_full_when_leashing_arrives_home() {
        let mut world = World::new();

        let home = Vec3::new(10.0, 0.0, 10.0);
        // Neutral is very close to home (within snap distance).
        let neutral_pos = Vec3::new(10.2, 0.0, 10.0);
        let neutral = spawn_neutral(&mut world, neutral_pos, home, 15.0);

        // Damage the neutral.
        world.get_mut::<Health>(neutral).unwrap().current = 50.0;

        // Attacker is dead — triggers leash.
        let attacker = spawn_attacker(&mut world, Vec3::new(50.0, 0.0, 10.0));
        world.insert(attacker, Dead);
        world.insert(neutral, LastAttacker(Some(attacker)));

        neutral_camp_system(&mut world, 0.016);

        // Should have snapped to home and healed.
        let health = world.get::<Health>(neutral).unwrap();
        assert_eq!(health.current, health.max, "neutral should heal to full HP");

        let pos = world.get::<LocalTransform>(neutral).unwrap().0.translation;
        assert_eq!(pos, home, "neutral should snap to home position");

        // LastAttacker should be cleared.
        let la = world.get::<LastAttacker>(neutral).unwrap();
        assert!(la.0.is_none(), "LastAttacker should be cleared after leash");
    }

    #[test]
    fn neutral_is_idle_when_no_aggro() {
        let mut world = World::new();

        let home = Vec3::new(10.0, 0.0, 10.0);
        let neutral = spawn_neutral(&mut world, home, home, 15.0);
        // No LastAttacker component at all.

        neutral_camp_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(neutral).unwrap();
        assert!(
            vel.linear.x.abs() < 0.001 && vel.linear.z.abs() < 0.001,
            "neutral should be idle with zero velocity"
        );
    }

    #[test]
    fn no_crash_without_last_attacker_component() {
        let mut world = World::new();

        let home = Vec3::new(5.0, 0.0, 5.0);
        let neutral = spawn_neutral(&mut world, home, home, 10.0);
        // Deliberately do NOT insert LastAttacker.

        // Should not panic.
        neutral_camp_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(neutral).unwrap();
        assert!(
            vel.linear.x.abs() < 0.001 && vel.linear.z.abs() < 0.001,
            "neutral should be idle when no LastAttacker present"
        );
    }

    #[test]
    fn neutral_leashes_when_attacker_dead() {
        let mut world = World::new();

        let home = Vec3::new(10.0, 0.0, 10.0);
        let neutral = spawn_neutral(&mut world, home, home, 15.0);

        // Attacker is dead.
        let attacker = spawn_attacker(&mut world, Vec3::new(12.0, 0.0, 10.0));
        world.insert(attacker, Dead);
        world.insert(neutral, LastAttacker(Some(attacker)));

        neutral_camp_system(&mut world, 0.016);

        // Neutral is at home already so should snap and clear.
        let la = world.get::<LastAttacker>(neutral).unwrap();
        assert!(
            la.0.is_none(),
            "should clear aggro when attacker is dead and neutral is at home"
        );
    }

    #[test]
    fn dead_neutral_is_skipped() {
        let mut world = World::new();

        let home = Vec3::new(10.0, 0.0, 10.0);
        let neutral = spawn_neutral(&mut world, home, home, 15.0);
        world.insert(neutral, Dead);

        let attacker = spawn_attacker(&mut world, Vec3::new(12.0, 0.0, 10.0));
        world.insert(neutral, LastAttacker(Some(attacker)));

        neutral_camp_system(&mut world, 0.016);

        // Velocity should remain at default (zero) — dead neutrals are skipped.
        let vel = world.get::<Velocity>(neutral).unwrap();
        assert!(
            vel.linear.x.abs() < 0.001 && vel.linear.z.abs() < 0.001,
            "dead neutral should not move"
        );
    }
}
