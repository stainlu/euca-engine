//! Hero definitions, per-hero stat growth, and hero registry.
//!
//! Components: `HeroName`, `StatGrowth`.
//! Resources: `HeroRegistry`.
//! Functions: `spawn_hero`.

use std::collections::HashMap;

use euca_ecs::{Entity, World};

use crate::abilities::{Ability, AbilityEffect, AbilitySet, AbilitySlot, Mana};
use crate::attributes::{
    AttributeGrowth, BaseAttributes, HeroAttributes, HeroTimings, PrimaryAttribute,
};
use crate::combat::{AutoCombat, EntityRole};
use crate::economy::{Gold, HeroEconomy};
use crate::health::Health;
use crate::inventory::Inventory;
use crate::leveling::Level;
use crate::stats::BaseStats;

// ── Components ──

/// Marker component: which hero this entity is playing.
#[derive(Clone, Debug)]
pub struct HeroName(pub String);

/// Per-level stat growth values. Applied on each level-up.
///
/// Maps stat name (e.g. `"max_health"`, `"attack_damage"`) to the amount
/// gained per level.
#[derive(Clone, Debug)]
pub struct StatGrowth(pub HashMap<String, f64>);

// ── Data types ──

/// Definition of an ability for a hero.
#[derive(Clone, Debug)]
pub struct AbilityDef {
    pub slot: AbilitySlot,
    pub name: String,
    pub cooldown: f32,
    pub mana_cost: f32,
    pub effect: AbilityEffect,
}

/// Complete definition of a hero: stats, abilities, and growth.
#[derive(Clone, Debug)]
pub struct HeroDef {
    /// Unique hero name (e.g. "Axe", "Crystal Maiden").
    pub name: String,
    /// Base stats at level 1.
    pub base_stats: HashMap<String, f64>,
    /// Stat growth per level-up.
    pub growth: HashMap<String, f64>,
    /// Starting health.
    pub health: f32,
    /// Starting mana.
    pub mana: f32,
    /// Starting gold.
    pub gold: i32,
    /// Combat damage.
    pub damage: f32,
    /// Combat range.
    pub range: f32,
    /// Ability definitions.
    pub abilities: Vec<AbilityDef>,
    /// Primary attribute (STR/AGI/INT/Universal). `None` for legacy heroes.
    pub primary_attribute: Option<PrimaryAttribute>,
    /// Base attribute values at level 1 (STR/AGI/INT).
    pub base_attributes: Option<BaseAttributes>,
    /// Per-level attribute growth rates.
    pub attribute_growth: Option<AttributeGrowth>,
    /// Movement, attack animation, vision, and projectile parameters.
    pub hero_timings: Option<HeroTimings>,
}

// ── Resources ──

/// Registry of all hero definitions, keyed by name.
#[derive(Clone, Debug, Default)]
pub struct HeroRegistry {
    pub heroes: HashMap<String, HeroDef>,
}

impl HeroRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, def: HeroDef) {
        self.heroes.insert(def.name.clone(), def);
    }

    pub fn get(&self, name: &str) -> Option<&HeroDef> {
        self.heroes.get(name)
    }
}

// ── Functions ──

/// Spawn a hero entity from a `HeroDef`, applying all components.
pub fn spawn_hero(world: &mut World, def: &HeroDef) -> Entity {
    let entity = world.spawn(HeroName(def.name.clone()));

    world.insert(entity, Health::new(def.health));
    world.insert(entity, Mana::new(def.mana, 5.0));
    world.insert(entity, Gold::new(def.gold));
    world.insert(entity, HeroEconomy::new());
    world.insert(entity, Level::new(1));
    world.insert(entity, Inventory::new(6));
    world.insert(entity, EntityRole::Hero);
    world.insert(entity, BaseStats(def.base_stats.clone()));
    world.insert(entity, StatGrowth(def.growth.clone()));

    let mut combat = AutoCombat::new();
    combat.damage = def.damage;
    combat.range = def.range;
    world.insert(entity, combat);

    // Set up abilities.
    let mut ability_set = AbilitySet::new();
    for ability_def in &def.abilities {
        ability_set.add(
            ability_def.slot,
            Ability {
                name: ability_def.name.clone(),
                cooldown: ability_def.cooldown,
                cooldown_remaining: 0.0,
                mana_cost: ability_def.mana_cost,
                effect: ability_def.effect.clone(),
                ..Default::default()
            },
        );
    }
    world.insert(entity, ability_set);

    // If the definition has Dota 2 attribute data, attach HeroAttributes.
    if let (Some(primary), Some(base), Some(growth)) = (
        def.primary_attribute,
        def.base_attributes,
        def.attribute_growth,
    ) {
        world.insert(
            entity,
            HeroAttributes {
                primary,
                base,
                growth,
                timings: def.hero_timings.unwrap_or_default(),
            },
        );
    }

    entity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_hero_creates_entity_with_components() {
        let mut world = World::new();

        let def = HeroDef {
            name: "TestHero".into(),
            base_stats: [("max_health".into(), 600.0)].into_iter().collect(),
            growth: [("max_health".into(), 80.0)].into_iter().collect(),
            health: 600.0,
            mana: 300.0,
            gold: 625,
            damage: 55.0,
            range: 1.5,
            abilities: vec![AbilityDef {
                slot: AbilitySlot::Q,
                name: "Fireball".into(),
                cooldown: 8.0,
                mana_cost: 90.0,
                effect: AbilityEffect::AreaDamage {
                    radius: 5.0,
                    damage: 100.0,
                },
            }],
            primary_attribute: None,
            base_attributes: None,
            attribute_growth: None,
            hero_timings: None,
        };

        let entity = spawn_hero(&mut world, &def);

        assert!(world.get::<HeroName>(entity).is_some());
        assert_eq!(world.get::<HeroName>(entity).unwrap().0, "TestHero");
        assert_eq!(world.get::<Health>(entity).unwrap().max, 600.0);
        assert_eq!(world.get::<Gold>(entity).unwrap().0, 625);
        assert_eq!(world.get::<Level>(entity).unwrap().level, 1);
        assert_eq!(*world.get::<EntityRole>(entity).unwrap(), EntityRole::Hero);

        // BaseStats
        let base = world.get::<BaseStats>(entity).unwrap();
        assert_eq!(base.0.get("max_health"), Some(&600.0));

        // StatGrowth
        let growth = world.get::<StatGrowth>(entity).unwrap();
        assert_eq!(growth.0.get("max_health"), Some(&80.0));

        // Abilities
        let abilities = world.get::<AbilitySet>(entity).unwrap();
        let q = abilities
            .get(AbilitySlot::Q)
            .expect("should have Q ability");
        assert_eq!(q.name, "Fireball");
        assert_eq!(q.cooldown, 8.0);
        assert_eq!(q.mana_cost, 90.0);

        // Mana
        let mana = world.get::<Mana>(entity).unwrap();
        assert_eq!(mana.max, 300.0);

        // HeroEconomy
        let econ = world.get::<HeroEconomy>(entity).unwrap();
        assert_eq!(econ.wallet.total(), crate::economy::STARTING_GOLD);

        // Inventory
        let inv = world.get::<Inventory>(entity).unwrap();
        assert_eq!(inv.max_slots, 6);
    }

    #[test]
    fn hero_registry_operations() {
        let mut registry = HeroRegistry::new();
        registry.register(HeroDef {
            name: "Axe".into(),
            base_stats: HashMap::new(),
            growth: HashMap::new(),
            health: 700.0,
            mana: 200.0,
            gold: 625,
            damage: 52.0,
            range: 1.5,
            abilities: vec![],
            primary_attribute: None,
            base_attributes: None,
            attribute_growth: None,
            hero_timings: None,
        });

        assert!(registry.get("Axe").is_some());
        assert_eq!(registry.get("Axe").unwrap().name, "Axe");
        assert!(registry.get("Unknown").is_none());
    }
}
