//! XP and leveling — heroes gain power through kills.
//!
//! Components: `Level`, `XpBounty`.
//! Systems: `xp_on_kill_system`.

use euca_ecs::{Events, World};

use crate::combat::AutoCombat;
use crate::health::{DeathEvent, Health};

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

/// Award XP on kill, auto-level-up with stat boosts.
pub fn xp_on_kill_system(world: &mut World) {
    let events: Vec<DeathEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<DeathEvent>().cloned().collect())
        .unwrap_or_default();

    let mut level_ups: Vec<(euca_ecs::Entity, u32)> = Vec::new();

    for event in events {
        let killer = match event.killer {
            Some(k) => k,
            None => continue,
        };

        let xp_reward = world.get::<XpBounty>(event.entity).map(|b| b.0);
        let xp_reward = match xp_reward {
            Some(xp) => xp,
            None => continue,
        };

        // Award XP
        if let Some(level) = world.get_mut::<Level>(killer) {
            level.xp += xp_reward;

            // Check for level up
            while level.xp >= level.xp_to_next && level.level < MAX_LEVEL {
                level.xp -= level.xp_to_next;
                level.level += 1;
                level.xp_to_next = xp_for_level(level.level);
                level_ups.push((killer, level.level));
                log::info!("Entity {} leveled up to {}", killer.index(), level.level);
            }
        }
    }

    // Apply stat boosts for each level-up
    for (entity, _new_level) in level_ups {
        if let Some(health) = world.get_mut::<Health>(entity) {
            health.max += 50.0;
            health.current += 50.0; // heal on level up
        }
        if let Some(combat) = world.get_mut::<AutoCombat>(entity) {
            combat.damage += 5.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_ecs::Entity;

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
}
