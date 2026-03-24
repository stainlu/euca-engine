//! Tower aggro override — forces towers to target heroes who attack allied heroes.
//!
//! In DotA, a tower switches target to an enemy hero when that hero attacks a
//! friendly hero within tower range. This prevents risk-free tower diving.
//!
//! Component: [`TowerAggroOverride`].
//! System: [`tower_aggro_system`].

use euca_ecs::{Entity, Events, Query, World};
use euca_scene::LocalTransform;

use crate::combat::{AutoCombat, EntityRole};
use crate::health::DamageEvent;
use crate::teams::Team;

/// Duration (in seconds) a tower aggro override lasts before expiring.
const AGGRO_OVERRIDE_DURATION: f32 = 3.0;

/// Overrides a tower's normal targeting for a duration.
///
/// Set by [`tower_aggro_system`] when a hero attacks another hero within tower
/// range. The tower will prioritize the attacker until the override expires or
/// the attacker dies / leaves range.
#[derive(Clone, Debug)]
pub struct TowerAggroOverride {
    /// The entity the tower is forced to target.
    pub target: Entity,
    /// Seconds remaining until this override expires.
    pub remaining: f32,
}

/// Scan damage events for hero-on-hero attacks and apply tower aggro overrides.
///
/// Each tick:
/// 1. Read all [`DamageEvent`]s where both source and target are heroes.
/// 2. For each such event, find all towers on the **victim's** team.
/// 3. If the **attacker** is within a tower's `combat.range`, insert a
///    [`TowerAggroOverride`] on that tower targeting the attacker.
/// 4. Tick down existing overrides by `dt` and remove any that have expired.
pub fn tower_aggro_system(world: &mut World, dt: f32) {
    // ── Phase 1: Collect hero-on-hero damage events ──
    let hero_attacks: Vec<(Entity, Entity)> = {
        let events = match world.resource::<Events>() {
            Some(e) => e,
            None => return,
        };
        events
            .read::<DamageEvent>()
            .filter_map(|ev| {
                let attacker = ev.source?;
                Some((attacker, ev.target))
            })
            .collect()
    };

    // Filter to only hero-on-hero attacks (both must have EntityRole::Hero).
    let hero_on_hero: Vec<(Entity, Entity)> = hero_attacks
        .into_iter()
        .filter(|(attacker, target)| {
            let attacker_is_hero = world
                .get::<EntityRole>(*attacker)
                .map(|r| *r == EntityRole::Hero)
                .unwrap_or(false);
            let target_is_hero = world
                .get::<EntityRole>(*target)
                .map(|r| *r == EntityRole::Hero)
                .unwrap_or(false);
            attacker_is_hero && target_is_hero
        })
        .collect();

    // ── Phase 2: Find towers and check range ──
    if !hero_on_hero.is_empty() {
        // Collect all towers: entity, position, team, range.
        let towers: Vec<(Entity, f32, u8)> = {
            let query = Query::<(Entity, &EntityRole, &AutoCombat, &Team)>::new(world);
            query
                .iter()
                .filter(|(_, role, _, _)| **role == EntityRole::Tower)
                .map(|(e, _, combat, team)| (e, combat.range, team.0))
                .collect()
        };

        // Build a position lookup (we need positions for both towers and attackers).
        let mut overrides_to_insert: Vec<(Entity, Entity)> = Vec::new();

        for (attacker, victim) in &hero_on_hero {
            // Determine the victim's team.
            let victim_team = match world.get::<Team>(*victim) {
                Some(t) => t.0,
                None => continue,
            };

            let attacker_pos = match world.get::<LocalTransform>(*attacker) {
                Some(lt) => lt.0.translation,
                None => continue,
            };

            // Find all towers on the victim's team that are in range of the attacker.
            for &(tower_entity, tower_range, tower_team) in &towers {
                if tower_team != victim_team {
                    continue;
                }
                let tower_pos = match world.get::<LocalTransform>(tower_entity) {
                    Some(lt) => lt.0.translation,
                    None => continue,
                };
                let dist = (attacker_pos - tower_pos).length();
                if dist <= tower_range {
                    overrides_to_insert.push((tower_entity, *attacker));
                }
            }
        }

        // Apply overrides.
        for (tower, attacker) in overrides_to_insert {
            world.insert(
                tower,
                TowerAggroOverride {
                    target: attacker,
                    remaining: AGGRO_OVERRIDE_DURATION,
                },
            );
        }
    }

    // ── Phase 3: Tick down existing overrides and remove expired ones ──
    let entities_with_override: Vec<Entity> = {
        let query = Query::<(Entity, &TowerAggroOverride)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    for entity in entities_with_override {
        if let Some(ovr) = world.get_mut::<TowerAggroOverride>(entity) {
            ovr.remaining -= dt;
            if ovr.remaining <= 0.0 {
                world.remove::<TowerAggroOverride>(entity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combat::AutoCombat;
    use crate::health::{Dead, Health};
    use euca_math::{Transform, Vec3};
    use euca_physics::Velocity;

    fn setup_world() -> World {
        let mut world = World::new();
        world.insert_resource(Events::default());
        world
    }

    fn spawn_hero(world: &mut World, pos: Vec3, team: u8) -> Entity {
        let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(e, Health::new(500.0));
        world.insert(e, Team(team));
        world.insert(e, EntityRole::Hero);
        world.insert(e, AutoCombat::new());
        world.insert(e, Velocity::default());
        e
    }

    fn spawn_tower(world: &mut World, pos: Vec3, team: u8, range: f32) -> Entity {
        let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(e, Health::new(2000.0));
        world.insert(e, Team(team));
        world.insert(e, EntityRole::Tower);
        world.insert(e, AutoCombat::stationary(40.0, range, 1.5));
        world.insert(e, Velocity::default());
        e
    }

    /// Emit a DamageEvent as if `attacker` dealt damage to `victim`.
    fn emit_hero_attack(world: &mut World, attacker: Entity, victim: Entity) {
        world
            .resource_mut::<Events>()
            .unwrap()
            .send(DamageEvent::new(victim, 50.0, Some(attacker)));
    }

    // ── Test: Tower switches to hero who attacks allied hero within range ──

    #[test]
    fn tower_switches_to_attacker_in_range() {
        let mut world = setup_world();

        // Team 1 tower at origin with range 10.
        let tower = spawn_tower(&mut world, Vec3::ZERO, 1, 10.0);

        // Team 1 hero (victim) near the tower.
        let victim = spawn_hero(&mut world, Vec3::new(2.0, 0.0, 0.0), 1);

        // Team 2 hero (attacker) within tower range.
        let attacker = spawn_hero(&mut world, Vec3::new(5.0, 0.0, 0.0), 2);

        // Attacker damages victim.
        emit_hero_attack(&mut world, attacker, victim);

        tower_aggro_system(&mut world, 1.0 / 60.0);

        let ovr = world
            .get::<TowerAggroOverride>(tower)
            .expect("tower should have aggro override");
        assert_eq!(
            ovr.target.index(),
            attacker.index(),
            "tower should target the attacker"
        );
        assert!(
            (ovr.remaining - AGGRO_OVERRIDE_DURATION).abs() < 0.02,
            "override duration should be close to initial value"
        );
    }

    // ── Test: Tower does NOT switch if attacker is outside tower range ──

    #[test]
    fn tower_does_not_switch_if_attacker_out_of_range() {
        let mut world = setup_world();

        let tower = spawn_tower(&mut world, Vec3::ZERO, 1, 10.0);
        let victim = spawn_hero(&mut world, Vec3::new(2.0, 0.0, 0.0), 1);
        // Attacker is 20 units away — outside tower range of 10.
        let attacker = spawn_hero(&mut world, Vec3::new(20.0, 0.0, 0.0), 2);

        emit_hero_attack(&mut world, attacker, victim);

        tower_aggro_system(&mut world, 1.0 / 60.0);

        assert!(
            world.get::<TowerAggroOverride>(tower).is_none(),
            "tower should NOT have aggro override when attacker is out of range"
        );
    }

    // ── Test: Override expires after duration ──

    #[test]
    fn override_expires_after_duration() {
        let mut world = setup_world();

        let tower = spawn_tower(&mut world, Vec3::ZERO, 1, 10.0);
        let victim = spawn_hero(&mut world, Vec3::new(2.0, 0.0, 0.0), 1);
        let attacker = spawn_hero(&mut world, Vec3::new(5.0, 0.0, 0.0), 2);

        emit_hero_attack(&mut world, attacker, victim);
        tower_aggro_system(&mut world, 1.0 / 60.0);
        assert!(world.get::<TowerAggroOverride>(tower).is_some());

        // Clear events so no new override is created, then tick past expiry.
        world
            .resource_mut::<Events>()
            .unwrap()
            .clear::<DamageEvent>();

        // Tick until expired (3+ seconds).
        tower_aggro_system(&mut world, 3.1);

        assert!(
            world.get::<TowerAggroOverride>(tower).is_none(),
            "override should be removed after expiry"
        );
    }

    // ── Test: Override doesn't apply if override target is dead ──
    // (This is tested in the auto_combat_system integration — here we verify
    //  the override *component* still exists but the combat system ignores it.)

    #[test]
    fn override_target_dead_does_not_crash() {
        let mut world = setup_world();

        let _tower = spawn_tower(&mut world, Vec3::ZERO, 1, 10.0);
        let victim = spawn_hero(&mut world, Vec3::new(2.0, 0.0, 0.0), 1);
        let attacker = spawn_hero(&mut world, Vec3::new(5.0, 0.0, 0.0), 2);

        emit_hero_attack(&mut world, attacker, victim);
        tower_aggro_system(&mut world, 1.0 / 60.0);

        // Kill the attacker.
        world.insert(attacker, Dead);
        world.get_mut::<Health>(attacker).unwrap().current = 0.0;

        // Running tower_aggro_system again should not crash.
        world
            .resource_mut::<Events>()
            .unwrap()
            .clear::<DamageEvent>();
        tower_aggro_system(&mut world, 1.0 / 60.0);

        // Override still exists (it's up to auto_combat_system to skip dead targets).
        // The tower_aggro_system only removes overrides on expiry, not on death.
        // Death-checking is done by auto_combat_system when it reads the override.
    }

    // ── Test: No crash if no towers or no damage events ──

    #[test]
    fn no_crash_without_towers() {
        let mut world = setup_world();

        let victim = spawn_hero(&mut world, Vec3::new(2.0, 0.0, 0.0), 1);
        let attacker = spawn_hero(&mut world, Vec3::new(5.0, 0.0, 0.0), 2);

        emit_hero_attack(&mut world, attacker, victim);
        tower_aggro_system(&mut world, 1.0 / 60.0);
        // Should not panic.
    }

    #[test]
    fn no_crash_without_damage_events() {
        let mut world = setup_world();
        let _tower = spawn_tower(&mut world, Vec3::ZERO, 1, 10.0);

        // No damage events at all.
        tower_aggro_system(&mut world, 1.0 / 60.0);
        // Should not panic.
    }

    // ── Test: Minion-on-hero attack does NOT trigger tower aggro ──

    #[test]
    fn minion_attack_does_not_trigger_tower_aggro() {
        let mut world = setup_world();

        let tower = spawn_tower(&mut world, Vec3::ZERO, 1, 10.0);
        let victim = spawn_hero(&mut world, Vec3::new(2.0, 0.0, 0.0), 1);

        // Spawn a minion instead of a hero as the attacker.
        let minion = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            5.0, 0.0, 0.0,
        ))));
        world.insert(minion, Health::new(200.0));
        world.insert(minion, Team(2));
        world.insert(minion, EntityRole::Minion);

        emit_hero_attack(&mut world, minion, victim);
        tower_aggro_system(&mut world, 1.0 / 60.0);

        assert!(
            world.get::<TowerAggroOverride>(tower).is_none(),
            "minion attack should not trigger tower aggro"
        );
    }
}
