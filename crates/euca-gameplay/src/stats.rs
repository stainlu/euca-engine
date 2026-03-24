//! Stat block and damage resistance — data-driven entity attributes.
//!
//! Components: `BaseStats`, `ResolvedStats`, `DamageResistance`.
//! Systems: `stat_resolution_system`.
//!
//! Stats are arbitrary key-value pairs (no hardcoded stat names), making this
//! system genre-agnostic. Games define their own stat keys as data.

use std::collections::HashMap;

use euca_ecs::World;

/// Entity's intrinsic stats — arbitrary key-value pairs.
///
/// Examples: `"max_health": 500.0`, `"attack_damage": 80.0`, `"physical_armor": 50.0`.
/// No stat names are hardcoded; games define them freely.
#[derive(Clone, Debug, Default)]
pub struct BaseStats(pub HashMap<String, f64>);

/// Computed stats after applying modifiers (equipment, buffs, etc.).
///
/// For now, this is a direct copy of `BaseStats`. Future units will layer
/// equipment bonuses and status-effect modifiers on top.
#[derive(Clone, Debug, Default)]
pub struct ResolvedStats(pub HashMap<String, f64>);

/// Maps damage categories to resistance values.
///
/// Resistance formula: `effective = raw * (100.0 / (100.0 + resistance))`.
/// The special category `"true"` bypasses resistance entirely.
///
/// Examples: `"physical": 50.0`, `"magical": 30.0`.
#[derive(Clone, Debug, Default)]
pub struct DamageResistance(pub HashMap<String, f64>);

/// For each entity with `BaseStats`, compute and write `ResolvedStats`.
///
/// Currently a simple copy. Future units will add equipment and effect
/// modifier layers here.
pub fn stat_resolution_system(world: &mut World) {
    use euca_ecs::{Entity, Query};

    let entities_with_stats: Vec<(Entity, HashMap<String, f64>)> = {
        let query = Query::<(Entity, &BaseStats)>::new(world);
        query.iter().map(|(e, bs)| (e, bs.0.clone())).collect()
    };

    for (entity, stats) in entities_with_stats {
        world.insert(entity, ResolvedStats(stats));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_ecs::World;

    #[test]
    fn stat_resolution_copies_base_stats() {
        let mut world = World::new();

        let mut stats = HashMap::new();
        stats.insert("max_health".to_string(), 500.0);
        stats.insert("attack_damage".to_string(), 80.0);

        let entity = world.spawn(BaseStats(stats.clone()));

        stat_resolution_system(&mut world);

        let resolved = world
            .get::<ResolvedStats>(entity)
            .expect("should have ResolvedStats after resolution");
        assert_eq!(resolved.0.get("max_health"), Some(&500.0));
        assert_eq!(resolved.0.get("attack_damage"), Some(&80.0));
        assert_eq!(resolved.0.len(), 2);
    }

    #[test]
    fn stat_resolution_overwrites_on_rerun() {
        let mut world = World::new();

        let mut stats = HashMap::new();
        stats.insert("armor".to_string(), 30.0);
        let entity = world.spawn(BaseStats(stats));

        stat_resolution_system(&mut world);

        // Change base stats
        world
            .get_mut::<BaseStats>(entity)
            .unwrap()
            .0
            .insert("armor".to_string(), 60.0);

        stat_resolution_system(&mut world);

        let resolved = world.get::<ResolvedStats>(entity).unwrap();
        assert_eq!(resolved.0.get("armor"), Some(&60.0));
    }

    #[test]
    fn entities_without_base_stats_unaffected() {
        let mut world = World::new();

        // Entity with no BaseStats
        let entity = world.spawn(42_u32);

        stat_resolution_system(&mut world);

        assert!(world.get::<ResolvedStats>(entity).is_none());
    }

    #[test]
    fn damage_resistance_default_is_empty() {
        let dr = DamageResistance::default();
        assert!(dr.0.is_empty());
    }
}
