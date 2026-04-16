//! Tower aggro override — forces towers to target heroes attacking allied heroes.
//!
//! In a MOBA, when an enemy hero attacks a friendly hero under a tower,
//! the tower should switch its target to that enemy hero ("tower aggro").
//!
//! Components: `TowerAggroOverride`.
//! Systems: `tower_aggro_system`.

use euca_ecs::{Entity, Query, World};

use euca_gameplay::combat::{AutoCombat, EntityRole, TargetOverride};
use euca_gameplay::health::{Dead, LastAttacker};
use euca_gameplay::teams::Team;

/// Backward-compatible alias. The concept now lives in `combat::TargetOverride`
/// since any genre can force a target override, not just tower aggro.
pub type TowerAggroOverride = TargetOverride;

/// Check for enemy heroes attacking allied heroes near towers and set
/// `TowerAggroOverride` on the tower so auto-combat picks up the override.
pub fn tower_aggro_system(world: &mut World) {
    // Collect towers (identified by EntityRole::Tower).
    let towers: Vec<(Entity, u8, f32)> = {
        let query = Query::<(Entity, &EntityRole, &Team, &AutoCombat)>::new(world);
        query
            .iter()
            .filter(|(e, role, _, _)| {
                **role == EntityRole::Tower && world.get::<Dead>(*e).is_none()
            })
            .map(|(e, _, team, combat)| (e, team.0, combat.range))
            .collect()
    };

    // Collect allied heroes that were recently attacked (have a LastAttacker).
    let attacked_heroes: Vec<(Entity, u8, Entity)> = {
        let query = Query::<(Entity, &EntityRole, &Team, &LastAttacker)>::new(world);
        query
            .iter()
            .filter(|(e, role, _, _)| **role == EntityRole::Hero && world.get::<Dead>(*e).is_none())
            .filter_map(|(e, _, team, la)| la.0.map(|attacker| (e, team.0, attacker)))
            .collect()
    };

    // For each tower, check if any allied hero is being attacked by an enemy hero.
    for (tower_entity, tower_team, _range) in &towers {
        let mut override_target = None;

        for (_hero_entity, hero_team, attacker) in &attacked_heroes {
            if hero_team != tower_team {
                continue; // Not an allied hero.
            }

            // Check if the attacker is an enemy hero.
            let attacker_is_enemy_hero = world
                .get::<EntityRole>(*attacker)
                .map(|r| *r == EntityRole::Hero)
                .unwrap_or(false)
                && world
                    .get::<Team>(*attacker)
                    .map(|t| t.0 != *tower_team)
                    .unwrap_or(false)
                && world.get::<Dead>(*attacker).is_none();

            if attacker_is_enemy_hero {
                override_target = Some(*attacker);
                break; // One override per tower per tick.
            }
        }

        if let Some(target) = override_target {
            world.insert(*tower_entity, TargetOverride { target });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_gameplay::health::Health;

    #[test]
    fn tower_retargets_to_hero_attacker() {
        let mut world = World::new();

        // Tower (team 1, role Tower).
        let tower = world.spawn(EntityRole::Tower);
        world.insert(tower, Team(1));
        world.insert(tower, AutoCombat::new());
        world.insert(tower, Health::new(2000.0));

        // Friendly hero (team 1) being attacked.
        let ally_hero = world.spawn(EntityRole::Hero);
        world.insert(ally_hero, Team(1));
        world.insert(ally_hero, Health::new(500.0));

        // Enemy hero (team 2) attacking the ally.
        let enemy_hero = world.spawn(EntityRole::Hero);
        world.insert(enemy_hero, Team(2));
        world.insert(enemy_hero, Health::new(500.0));

        // The ally hero was hit by the enemy hero.
        world.insert(ally_hero, LastAttacker(Some(enemy_hero)));

        tower_aggro_system(&mut world);

        let ovr = world.get::<TowerAggroOverride>(tower).unwrap();
        assert_eq!(
            ovr.target, enemy_hero,
            "tower should have aggro override targeting the enemy hero"
        );
    }

    #[test]
    fn tower_ignores_non_hero_attacker() {
        let mut world = World::new();

        let tower = world.spawn(EntityRole::Tower);
        world.insert(tower, Team(1));
        world.insert(tower, AutoCombat::new());
        world.insert(tower, Health::new(2000.0));

        let ally_hero = world.spawn(EntityRole::Hero);
        world.insert(ally_hero, Team(1));
        world.insert(ally_hero, Health::new(500.0));

        // Attacker is a minion, not a hero.
        let enemy_minion = world.spawn(EntityRole::Minion);
        world.insert(enemy_minion, Team(2));
        world.insert(enemy_minion, Health::new(200.0));

        world.insert(ally_hero, LastAttacker(Some(enemy_minion)));

        tower_aggro_system(&mut world);

        // Tower should not have a TowerAggroOverride.
        assert!(
            world.get::<TowerAggroOverride>(tower).is_none(),
            "tower should not get aggro override for non-hero attackers"
        );
    }
}
