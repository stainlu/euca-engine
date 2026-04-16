//! XP and leveling — heroes gain power through kills.
//!
//! Components: `Level`, `XpBounty`.
//! Resources: `XpShareRadius`.
//! Systems: `xp_on_kill_system`.

use std::collections::HashMap;

use euca_ecs::{Entity, Events, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;

use crate::combat::{AutoCombat, EntityRole};
use crate::health::{DeathEvent, Health};
use crate::stats::BaseStats;

/// Per-level stat growth values. Applied on each level-up.
///
/// Maps stat name (e.g. `"max_health"`, `"attack_damage"`) to the amount
/// gained per level. Used by `apply_level_up_stats` in the leveling system.
#[derive(Clone, Debug)]
pub struct StatGrowth(pub HashMap<String, f64>);

/// Max hero level (like LoL).
pub const MAX_LEVEL: u32 = 18;

/// XP required to reach next level (scales with current level).
pub fn xp_for_level(level: u32) -> u32 {
    100 + level * 80 // level 1→2: 180 XP, level 17→18: 1460 XP
}

/// Entity's current level and XP progress.
#[derive(Clone, Debug)]
pub struct Level {
    pub level: u32,
    pub xp: u32,
    pub xp_to_next: u32,
}

impl Level {
    pub fn new(starting_level: u32) -> Self {
        Self {
            level: starting_level.clamp(1, MAX_LEVEL),
            xp: 0,
            xp_to_next: xp_for_level(starting_level),
        }
    }
}

/// How much XP the killer receives when this entity dies.
#[derive(Clone, Copy, Debug)]
pub struct XpBounty(pub u32);

/// Configurable radius for XP sharing. All heroes within this distance
/// of the victim split XP evenly (killer included).
///
/// Default: 15.0 units.
#[derive(Clone, Copy, Debug)]
pub struct XpShareRadius(pub f32);

impl Default for XpShareRadius {
    fn default() -> Self {
        Self(15.0)
    }
}

/// Collect all hero entities (Level + EntityRole::Hero) within `radius`
/// of `center`, returning their entity IDs.
fn heroes_in_radius(world: &World, center: Vec3, radius: f32) -> Vec<Entity> {
    let query = Query::<(Entity, &Level, &EntityRole, &LocalTransform)>::new(world);
    query
        .iter()
        .filter(|(_, _, role, lt)| {
            **role == EntityRole::Hero && (lt.0.translation - center).length() <= radius
        })
        .map(|(e, _, _, _)| e)
        .collect()
}

/// Award XP to an entity and collect level-ups.
fn award_xp(world: &mut World, entity: Entity, xp: u32) -> Vec<(Entity, u32)> {
    let mut level_ups = Vec::new();
    if let Some(level) = world.get_mut::<Level>(entity) {
        level.xp += xp;
        while level.xp >= level.xp_to_next && level.level < MAX_LEVEL {
            level.xp -= level.xp_to_next;
            level.level += 1;
            level.xp_to_next = xp_for_level(level.level);
            level_ups.push((entity, level.level));
            log::info!("Entity {} leveled up to {}", entity.index(), level.level);
        }
    }
    level_ups
}

/// Apply stat growth on level-up. If the entity has a `StatGrowth` component,
/// use its per-hero values. Otherwise fall back to hardcoded defaults for
/// backward compatibility.
fn apply_level_up_stats(world: &mut World, entity: Entity) {
    let growth = world.get::<StatGrowth>(entity).cloned();

    if let Some(growth) = growth {
        // Apply each stat growth to BaseStats
        if let Some(base) = world.get_mut::<BaseStats>(entity) {
            for (stat, value) in &growth.0 {
                *base.0.entry(stat.clone()).or_insert(0.0) += value;
            }
        }
        // Special handling: if "max_health" grew, also heal by the growth amount
        if let Some(&hp_growth) = growth.0.get("max_health")
            && let Some(health) = world.get_mut::<Health>(entity)
        {
            health.max += hp_growth as f32;
            health.current += hp_growth as f32;
        }
    } else {
        // Fallback: hardcoded bonuses for entities without StatGrowth
        if let Some(health) = world.get_mut::<Health>(entity) {
            health.max += 50.0;
            health.current += 50.0;
        }
        if let Some(combat) = world.get_mut::<AutoCombat>(entity) {
            combat.damage += 5.0;
        }
    }
}

/// Award XP on kill, auto-level-up with stat boosts.
///
/// When an entity with `XpBounty` dies:
/// 1. Find all heroes within `XpShareRadius` of the victim.
/// 2. If heroes are nearby, split XP evenly among them (killer included
///    if within radius). If no heroes are nearby, only the killer gets XP.
/// 3. On level-up, apply per-hero `StatGrowth` (or hardcoded fallback).
pub fn xp_on_kill_system(world: &mut World) {
    let events: Vec<DeathEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<DeathEvent>().cloned().collect())
        .unwrap_or_default();

    let share_radius = world
        .resource::<XpShareRadius>()
        .copied()
        .unwrap_or_default();

    let mut all_level_ups: Vec<(Entity, u32)> = Vec::new();

    for event in events {
        let killer = match event.killer {
            Some(k) => k,
            None => continue,
        };

        let xp_reward = match world.get::<XpBounty>(event.entity).map(|b| b.0) {
            Some(xp) => xp,
            None => continue,
        };

        // Find the victim's position for XP sharing radius check.
        let victim_pos = world
            .get::<LocalTransform>(event.entity)
            .map(|lt| lt.0.translation);

        // Determine XP recipients: heroes near the victim.
        let recipients = if let Some(pos) = victim_pos {
            let nearby = heroes_in_radius(world, pos, share_radius.0);
            if nearby.is_empty() {
                // No heroes in radius — only killer gets XP.
                vec![killer]
            } else {
                nearby
            }
        } else {
            // Victim has no position — only killer gets XP.
            vec![killer]
        };

        // Split XP evenly among recipients.
        let share = xp_reward / recipients.len() as u32;
        let share = share.max(1); // At least 1 XP per recipient.

        for recipient in recipients {
            let level_ups = award_xp(world, recipient, share);
            all_level_ups.extend(level_ups);
        }
    }

    // Apply stat boosts for each level-up.
    for (entity, _new_level) in all_level_ups {
        apply_level_up_stats(world, entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    #[test]
    fn xp_for_level_scales() {
        assert_eq!(xp_for_level(1), 180);
        assert_eq!(xp_for_level(10), 900);
    }

    #[test]
    fn level_up_on_kill() {
        let mut world = euca_ecs::World::new();
        world.insert_resource(euca_ecs::Events::default());

        let killer = world.spawn(Level::new(1));
        world.insert(killer, Health::new(500.0));
        world.insert(killer, AutoCombat::new());

        let victim = world.spawn(XpBounty(200));

        world
            .resource_mut::<euca_ecs::Events>()
            .unwrap()
            .send(DeathEvent {
                entity: victim,
                killer: Some(killer),
            });

        xp_on_kill_system(&mut world);

        let level = world.get::<Level>(killer).unwrap();
        assert_eq!(level.level, 2); // 200 XP > 180 needed

        let health = world.get::<Health>(killer).unwrap();
        assert_eq!(health.max, 550.0); // +50 from level up

        let combat = world.get::<AutoCombat>(killer).unwrap();
        assert_eq!(combat.damage, 15.0); // 10 + 5 from level up
    }

    #[test]
    fn max_level_cap() {
        let mut world = euca_ecs::World::new();
        world.insert_resource(euca_ecs::Events::default());

        let killer = world.spawn(Level::new(MAX_LEVEL));

        let victim = world.spawn(XpBounty(99999));

        world
            .resource_mut::<euca_ecs::Events>()
            .unwrap()
            .send(DeathEvent {
                entity: victim,
                killer: Some(killer),
            });

        xp_on_kill_system(&mut world);

        let level = world.get::<Level>(killer).unwrap();
        assert_eq!(level.level, MAX_LEVEL); // capped
    }

    // ── Per-hero stat growth tests ──

    #[test]
    fn level_up_with_stat_growth_applies_per_hero_growth() {
        let mut world = euca_ecs::World::new();
        world.insert_resource(euca_ecs::Events::default());

        let killer = world.spawn(Level::new(1));
        world.insert(killer, Health::new(500.0));
        world.insert(killer, AutoCombat::new());
        world.insert(
            killer,
            BaseStats(
                [
                    ("max_health".to_string(), 500.0),
                    ("attack_damage".to_string(), 50.0),
                ]
                .into_iter()
                .collect(),
            ),
        );
        world.insert(
            killer,
            StatGrowth(
                [
                    ("max_health".to_string(), 80.0),
                    ("attack_damage".to_string(), 4.0),
                ]
                .into_iter()
                .collect(),
            ),
        );

        let victim = world.spawn(XpBounty(200));

        world
            .resource_mut::<euca_ecs::Events>()
            .unwrap()
            .send(DeathEvent {
                entity: victim,
                killer: Some(killer),
            });

        xp_on_kill_system(&mut world);

        // BaseStats should have grown
        let base = world.get::<BaseStats>(killer).unwrap();
        assert_eq!(base.0.get("max_health"), Some(&580.0)); // 500 + 80
        assert_eq!(base.0.get("attack_damage"), Some(&54.0)); // 50 + 4

        // Health max should also increase (special handling for max_health)
        let health = world.get::<Health>(killer).unwrap();
        assert_eq!(health.max, 580.0); // 500 + 80
        assert_eq!(health.current, 580.0); // healed by growth amount
    }

    #[test]
    fn level_up_without_stat_growth_uses_fallback() {
        let mut world = euca_ecs::World::new();
        world.insert_resource(euca_ecs::Events::default());

        // No StatGrowth component — should use hardcoded fallback
        let killer = world.spawn(Level::new(1));
        world.insert(killer, Health::new(500.0));
        world.insert(killer, AutoCombat::new());

        let victim = world.spawn(XpBounty(200));

        world
            .resource_mut::<euca_ecs::Events>()
            .unwrap()
            .send(DeathEvent {
                entity: victim,
                killer: Some(killer),
            });

        xp_on_kill_system(&mut world);

        let health = world.get::<Health>(killer).unwrap();
        assert_eq!(health.max, 550.0); // +50 fallback

        let combat = world.get::<AutoCombat>(killer).unwrap();
        assert_eq!(combat.damage, 15.0); // +5 fallback
    }

    // ── XP sharing tests ──

    #[test]
    fn xp_sharing_two_heroes_near_victim() {
        let mut world = euca_ecs::World::new();
        world.insert_resource(euca_ecs::Events::default());
        world.insert_resource(XpShareRadius(15.0));

        let victim_pos = Vec3::new(10.0, 0.0, 10.0);

        // Hero 1 (the killer) — near victim
        let hero1 = world.spawn(Level::new(1));
        world.insert(hero1, Health::new(500.0));
        world.insert(hero1, EntityRole::Hero);
        world.insert(
            hero1,
            LocalTransform(Transform::from_translation(Vec3::new(12.0, 0.0, 10.0))),
        );

        // Hero 2 — also near victim
        let hero2 = world.spawn(Level::new(1));
        world.insert(hero2, Health::new(500.0));
        world.insert(hero2, EntityRole::Hero);
        world.insert(
            hero2,
            LocalTransform(Transform::from_translation(Vec3::new(8.0, 0.0, 10.0))),
        );

        // Victim with XP bounty and position
        let victim = world.spawn(XpBounty(200));
        world.insert(
            victim,
            LocalTransform(Transform::from_translation(victim_pos)),
        );

        world
            .resource_mut::<euca_ecs::Events>()
            .unwrap()
            .send(DeathEvent {
                entity: victim,
                killer: Some(hero1),
            });

        xp_on_kill_system(&mut world);

        // Each hero should get 100 XP (200 / 2)
        let level1 = world.get::<Level>(hero1).unwrap();
        assert_eq!(level1.xp, 100);

        let level2 = world.get::<Level>(hero2).unwrap();
        assert_eq!(level2.xp, 100);
    }

    #[test]
    fn xp_sharing_far_hero_excluded() {
        let mut world = euca_ecs::World::new();
        world.insert_resource(euca_ecs::Events::default());
        world.insert_resource(XpShareRadius(15.0));

        let victim_pos = Vec3::new(10.0, 0.0, 10.0);

        // Hero 1 (the killer) — near victim
        let hero1 = world.spawn(Level::new(1));
        world.insert(hero1, Health::new(500.0));
        world.insert(hero1, EntityRole::Hero);
        world.insert(
            hero1,
            LocalTransform(Transform::from_translation(Vec3::new(12.0, 0.0, 10.0))),
        );

        // Hero 2 — far away from victim (beyond 15.0 radius)
        let hero2 = world.spawn(Level::new(1));
        world.insert(hero2, Health::new(500.0));
        world.insert(hero2, EntityRole::Hero);
        world.insert(
            hero2,
            LocalTransform(Transform::from_translation(Vec3::new(100.0, 0.0, 100.0))),
        );

        // Victim with XP bounty and position
        let victim = world.spawn(XpBounty(200));
        world.insert(
            victim,
            LocalTransform(Transform::from_translation(victim_pos)),
        );

        world
            .resource_mut::<euca_ecs::Events>()
            .unwrap()
            .send(DeathEvent {
                entity: victim,
                killer: Some(hero1),
            });

        xp_on_kill_system(&mut world);

        // Hero 1 is the only hero in radius — gets all 200 XP
        let level1 = world.get::<Level>(hero1).unwrap();
        assert!(level1.xp >= 180 || level1.level >= 2); // leveled up from 200 XP

        // Hero 2 gets nothing
        let level2 = world.get::<Level>(hero2).unwrap();
        assert_eq!(level2.xp, 0);
        assert_eq!(level2.level, 1);
    }
}
