//! Gold economy — earn gold by killing enemies.
//!
//! Components: `Gold`, `GoldBounty`.
//! Systems: `gold_on_kill_system`.

use euca_ecs::{Events, World};

use crate::health::DeathEvent;

/// How much gold this entity carries. Heroes accumulate gold from kills.
#[derive(Clone, Copy, Debug)]
pub struct Gold(pub i32);

impl Gold {
    pub fn new(amount: i32) -> Self {
        Self(amount)
    }
}

/// How much gold the killer receives when this entity dies.
#[derive(Clone, Copy, Debug)]
pub struct GoldBounty(pub i32);

/// Award gold to killers when entities with GoldBounty die.
pub fn gold_on_kill_system(world: &mut World) {
    let events: Vec<DeathEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<DeathEvent>().cloned().collect())
        .unwrap_or_default();

    for event in events {
        let killer = match event.killer {
            Some(k) => k,
            None => continue,
        };

        // Get bounty from victim
        let bounty = world.get::<GoldBounty>(event.entity).map(|b| b.0);
        let bounty = match bounty {
            Some(b) => b,
            None => continue,
        };

        // Award gold to killer
        if let Some(gold) = world.get_mut::<Gold>(killer) {
            gold.0 += bounty;
            log::info!(
                "Entity {} earned {} gold (total: {})",
                killer.index(),
                bounty,
                gold.0
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::{DeathEvent, Health};
    use euca_ecs::Entity;

    #[test]
    fn gold_on_kill_awards_bounty() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let killer = world.spawn(Gold(0));
        let victim = world.spawn(GoldBounty(100));

        // Emit death event
        world.resource_mut::<Events>().unwrap().send(DeathEvent {
            entity: victim,
            killer: Some(killer),
        });

        gold_on_kill_system(&mut world);

        assert_eq!(world.get::<Gold>(killer).unwrap().0, 100);
    }

    #[test]
    fn no_gold_without_bounty() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let killer = world.spawn(Gold(50));
        let victim = world.spawn(Health::new(100.0)); // no GoldBounty

        world.resource_mut::<Events>().unwrap().send(DeathEvent {
            entity: victim,
            killer: Some(killer),
        });

        gold_on_kill_system(&mut world);

        assert_eq!(world.get::<Gold>(killer).unwrap().0, 50); // unchanged
    }

    #[test]
    fn no_gold_without_killer() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let victim = world.spawn(GoldBounty(100));

        world.resource_mut::<Events>().unwrap().send(DeathEvent {
            entity: victim,
            killer: None, // no killer
        });

        gold_on_kill_system(&mut world);
        // Should not panic
    }
}
