//! Dota 2 hero attribute system.
//!
//! Heroes have three attributes (Strength, Agility, Intelligence) that grow
//! per level and convert to combat stats. One attribute is "primary" and
//! also grants bonus attack damage.

use euca_ecs::{Entity, Query, World};
use serde::{Deserialize, Serialize};

use euca_gameplay::combat::AutoCombat;
use euca_gameplay::health::Health;
use euca_gameplay::leveling::Level;

// ── Enums ──

/// Which attribute is a hero's primary (determines bonus attack damage source).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimaryAttribute {
    Strength,
    Agility,
    Intelligence,
    /// All attributes contribute damage at a reduced rate (0.7x).
    Universal,
}

// ── Attribute data ──

/// Base attribute values at level 1.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BaseAttributes {
    pub strength: f32,
    pub agility: f32,
    pub intelligence: f32,
}

/// Per-level attribute growth (applied once per level beyond 1).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AttributeGrowth {
    pub strength: f32,
    pub agility: f32,
    pub intelligence: f32,
}

/// Current computed attributes at a given level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ComputedAttributes {
    pub strength: f32,
    pub agility: f32,
    pub intelligence: f32,
}

// ── Derived combat stats ──

/// Combat stats derived from attribute values.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct DerivedStats {
    /// STR x 22
    pub bonus_hp: f32,
    /// STR x 0.1
    pub bonus_hp_regen: f32,
    /// AGI x 0.167
    pub bonus_armor: f32,
    /// AGI x 1.0
    pub bonus_attack_speed: f32,
    /// INT x 12
    pub bonus_mana: f32,
    /// INT x 0.05
    pub bonus_mana_regen: f32,
    /// Primary attribute bonus to attack damage.
    pub bonus_damage: f32,
    /// INT x 0.07% (expressed as a fraction, e.g. 0.07% = 0.0007).
    pub bonus_spell_amp: f32,
}

// ── Conversion constants (Dota 2 7.34+) ──

pub const HP_PER_STRENGTH: f32 = 22.0;
pub const HP_REGEN_PER_STRENGTH: f32 = 0.1;
pub const ARMOR_PER_AGILITY: f32 = 0.167;
pub const ATTACK_SPEED_PER_AGILITY: f32 = 1.0;
pub const MANA_PER_INTELLIGENCE: f32 = 12.0;
pub const MANA_REGEN_PER_INTELLIGENCE: f32 = 0.05;
/// 0.07% per INT point, stored as a fraction.
pub const SPELL_AMP_PER_INTELLIGENCE: f32 = 0.0007;
/// Universal heroes get 0.7x damage from every attribute.
pub const UNIVERSAL_DAMAGE_FACTOR: f32 = 0.7;

// ── Movement & animation timing ──

/// Movement, attack animation, vision, and projectile parameters for a hero.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HeroTimings {
    /// Radians per second (default 0.6).
    pub turn_rate: f32,
    /// Seconds from attack start to damage point (e.g. 0.3).
    pub attack_point: f32,
    /// Recovery time after the damage point (e.g. 0.7).
    pub attack_backswing: f32,
    /// Base attack time in seconds (default 1.7).
    pub base_attack_time: f32,
    /// Movement speed in units per second (default 300).
    pub movement_speed: f32,
    /// Default cast time for abilities.
    pub cast_point: f32,
    /// Day vision range (default 1800).
    pub vision_day: f32,
    /// Night vision range (default 800).
    pub vision_night: f32,
    /// Projectile speed (0 for melee, 900-1800 for ranged).
    pub projectile_speed: f32,
    /// Attack range (150 for melee, 400-700 for ranged).
    pub attack_range: f32,
}

impl Default for HeroTimings {
    fn default() -> Self {
        Self {
            turn_rate: 0.6,
            attack_point: 0.3,
            attack_backswing: 0.7,
            base_attack_time: 1.7,
            movement_speed: 300.0,
            cast_point: 0.3,
            vision_day: 1800.0,
            vision_night: 800.0,
            projectile_speed: 0.0,
            attack_range: 150.0,
        }
    }
}

// ── Complete definition ──

/// Complete hero attribute definition combining primary attribute, base values,
/// growth rates, and timing parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroAttributes {
    pub primary: PrimaryAttribute,
    pub base: BaseAttributes,
    pub growth: AttributeGrowth,
    pub timings: HeroTimings,
}

// ── Functions ──

/// Compute attribute totals at a given level.
///
/// Formula: `base + growth * (level - 1)`. At level 1, returns base values.
pub fn compute_attributes(
    base: &BaseAttributes,
    growth: &AttributeGrowth,
    level: u32,
) -> ComputedAttributes {
    let levels_gained = (level.saturating_sub(1)) as f32;
    ComputedAttributes {
        strength: base.strength + growth.strength * levels_gained,
        agility: base.agility + growth.agility * levels_gained,
        intelligence: base.intelligence + growth.intelligence * levels_gained,
    }
}

/// Derive combat stats from computed attributes and primary attribute type.
pub fn derive_stats(attrs: &ComputedAttributes, primary: PrimaryAttribute) -> DerivedStats {
    let bonus_damage = match primary {
        PrimaryAttribute::Strength => attrs.strength,
        PrimaryAttribute::Agility => attrs.agility,
        PrimaryAttribute::Intelligence => attrs.intelligence,
        PrimaryAttribute::Universal => {
            (attrs.strength + attrs.agility + attrs.intelligence) * UNIVERSAL_DAMAGE_FACTOR
        }
    };

    DerivedStats {
        bonus_hp: attrs.strength * HP_PER_STRENGTH,
        bonus_hp_regen: attrs.strength * HP_REGEN_PER_STRENGTH,
        bonus_armor: attrs.agility * ARMOR_PER_AGILITY,
        bonus_attack_speed: attrs.agility * ATTACK_SPEED_PER_AGILITY,
        bonus_mana: attrs.intelligence * MANA_PER_INTELLIGENCE,
        bonus_mana_regen: attrs.intelligence * MANA_REGEN_PER_INTELLIGENCE,
        bonus_damage,
        bonus_spell_amp: attrs.intelligence * SPELL_AMP_PER_INTELLIGENCE,
    }
}

/// Total HP = base HP + attribute bonus.
pub fn total_hp(base_hp: f32, derived: &DerivedStats) -> f32 {
    base_hp + derived.bonus_hp
}

/// Total mana = base mana + attribute bonus.
pub fn total_mana(base_mana: f32, derived: &DerivedStats) -> f32 {
    base_mana + derived.bonus_mana
}

/// Total armor = base armor + attribute bonus.
pub fn total_armor(base_armor: f32, derived: &DerivedStats) -> f32 {
    base_armor + derived.bonus_armor
}

/// Total attack damage = base damage + primary attribute bonus.
pub fn total_damage(base_damage: f32, derived: &DerivedStats) -> f32 {
    base_damage + derived.bonus_damage
}

/// Total attack speed (IAS) = base IAS + agility bonus.
///
/// Clamped to [20, 700] per Dota 2 rules.
pub fn total_attack_speed(base_ias: f32, derived: &DerivedStats) -> f32 {
    (base_ias + derived.bonus_attack_speed).clamp(20.0, 700.0)
}

/// Time between attacks in seconds.
///
/// Formula: `BAT / (total_attack_speed / 100)`.
pub fn attack_interval(bat: f32, total_attack_speed: f32) -> f32 {
    bat / (total_attack_speed / 100.0)
}

/// Time in seconds to turn from `current_facing` to `target_angle`.
///
/// Both angles are in radians. The function computes the shortest angular
/// distance and divides by the turn rate (radians per second).
pub fn turn_time(current_facing: f32, target_angle: f32, turn_rate: f32) -> f32 {
    let mut delta = (target_angle - current_facing) % std::f32::consts::TAU;
    if delta < 0.0 {
        delta += std::f32::consts::TAU;
    }
    if delta > std::f32::consts::PI {
        delta = std::f32::consts::TAU - delta;
    }
    delta / turn_rate
}

// ── Systems ──

/// Per-tick system that recomputes attribute-derived stats for every entity
/// that has `HeroAttributes` and `Level`.
///
/// For each qualifying entity:
/// 1. Compute current attributes from `base + growth * (level - 1)`.
/// 2. Derive bonus stats (HP, armor, damage, attack speed, mana).
/// 3. Update `Health.max` and `AutoCombat.damage` with the new totals.
///
/// The system stores a `DerivedStats` component on each entity as a cache
/// of what was last applied. On each tick it undoes the previous bonus and
/// applies the current one, so level-ups propagate automatically without
/// drift.
///
/// Entities without `HeroAttributes` are silently skipped, preserving
/// backward compatibility with legacy heroes.
pub fn attribute_update_system(world: &mut World) {
    // Read phase: collect entities that have HeroAttributes + Level.
    let entities: Vec<(Entity, HeroAttributes, u32)> = {
        let query = Query::<(Entity, &HeroAttributes, &Level)>::new(world);
        query
            .iter()
            .map(|(e, attrs, lvl)| (e, attrs.clone(), lvl.level))
            .collect()
    };

    // Write phase: update Health and AutoCombat from derived stats.
    for (entity, attrs, level) in entities {
        let computed = compute_attributes(&attrs.base, &attrs.growth, level);
        let derived = derive_stats(&computed, attrs.primary);

        // Read the previously applied bonus (defaults to zero on first tick).
        let prev = world
            .get::<DerivedStats>(entity)
            .copied()
            .unwrap_or_default();

        if let Some(health) = world.get_mut::<Health>(entity) {
            // Undo previous attribute bonus, apply current.
            let base_hp = health.max - prev.bonus_hp;
            health.max = base_hp + derived.bonus_hp;

            // Clamp current HP to new max (e.g. if max decreased).
            if health.current > health.max {
                health.current = health.max;
            }
        }

        if let Some(combat) = world.get_mut::<AutoCombat>(entity) {
            let base_damage = combat.damage - prev.bonus_damage;
            combat.damage = base_damage + derived.bonus_damage;
        }

        // Cache the derived stats so next tick can undo them.
        world.insert(entity, derived);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Helper: create a simple set of base attributes.
    fn sample_base() -> BaseAttributes {
        BaseAttributes {
            strength: 20.0,
            agility: 15.0,
            intelligence: 18.0,
        }
    }

    /// Helper: create a simple growth profile.
    fn sample_growth() -> AttributeGrowth {
        AttributeGrowth {
            strength: 3.0,
            agility: 2.0,
            intelligence: 1.5,
        }
    }

    // ── Attribute computation ──

    #[test]
    fn test_attributes_at_level_1() {
        let attrs = compute_attributes(&sample_base(), &sample_growth(), 1);
        assert_eq!(attrs.strength, 20.0);
        assert_eq!(attrs.agility, 15.0);
        assert_eq!(attrs.intelligence, 18.0);
    }

    #[test]
    fn test_attributes_at_level_25() {
        // base + growth * 24
        let attrs = compute_attributes(&sample_base(), &sample_growth(), 25);
        assert!((attrs.strength - (20.0 + 3.0 * 24.0)).abs() < 1e-4);
        assert!((attrs.agility - (15.0 + 2.0 * 24.0)).abs() < 1e-4);
        assert!((attrs.intelligence - (18.0 + 1.5 * 24.0)).abs() < 1e-4);
    }

    // ── Strength conversions ──

    #[test]
    fn test_str_gives_hp() {
        let attrs = ComputedAttributes {
            strength: 20.0,
            agility: 0.0,
            intelligence: 0.0,
        };
        let stats = derive_stats(&attrs, PrimaryAttribute::Strength);
        assert!((stats.bonus_hp - 440.0).abs() < 1e-4); // 20 * 22 = 440
    }

    #[test]
    fn test_str_gives_hp_regen() {
        let attrs = ComputedAttributes {
            strength: 20.0,
            agility: 0.0,
            intelligence: 0.0,
        };
        let stats = derive_stats(&attrs, PrimaryAttribute::Strength);
        assert!((stats.bonus_hp_regen - 2.0).abs() < 1e-4); // 20 * 0.1 = 2.0
    }

    // ── Agility conversions ──

    #[test]
    fn test_agi_gives_armor() {
        let attrs = ComputedAttributes {
            strength: 0.0,
            agility: 30.0,
            intelligence: 0.0,
        };
        let stats = derive_stats(&attrs, PrimaryAttribute::Agility);
        assert!((stats.bonus_armor - 5.01).abs() < 1e-4); // 30 * 0.167 = 5.01
    }

    #[test]
    fn test_agi_gives_attack_speed() {
        let attrs = ComputedAttributes {
            strength: 0.0,
            agility: 30.0,
            intelligence: 0.0,
        };
        let stats = derive_stats(&attrs, PrimaryAttribute::Agility);
        assert!((stats.bonus_attack_speed - 30.0).abs() < 1e-4); // 30 * 1.0 = 30
    }

    // ── Intelligence conversions ──

    #[test]
    fn test_int_gives_mana() {
        let attrs = ComputedAttributes {
            strength: 0.0,
            agility: 0.0,
            intelligence: 25.0,
        };
        let stats = derive_stats(&attrs, PrimaryAttribute::Intelligence);
        assert!((stats.bonus_mana - 300.0).abs() < 1e-4); // 25 * 12 = 300
    }

    #[test]
    fn test_int_gives_mana_regen() {
        let attrs = ComputedAttributes {
            strength: 0.0,
            agility: 0.0,
            intelligence: 25.0,
        };
        let stats = derive_stats(&attrs, PrimaryAttribute::Intelligence);
        assert!((stats.bonus_mana_regen - 1.25).abs() < 1e-4); // 25 * 0.05 = 1.25
    }

    // ── Primary attribute damage ──

    #[test]
    fn test_primary_str_damage() {
        let attrs = ComputedAttributes {
            strength: 25.0,
            agility: 10.0,
            intelligence: 15.0,
        };
        let stats = derive_stats(&attrs, PrimaryAttribute::Strength);
        assert!((stats.bonus_damage - 25.0).abs() < 1e-4);
    }

    #[test]
    fn test_primary_agi_damage() {
        let attrs = ComputedAttributes {
            strength: 10.0,
            agility: 30.0,
            intelligence: 15.0,
        };
        let stats = derive_stats(&attrs, PrimaryAttribute::Agility);
        assert!((stats.bonus_damage - 30.0).abs() < 1e-4);
    }

    #[test]
    fn test_universal_damage() {
        let attrs = ComputedAttributes {
            strength: 20.0,
            agility: 20.0,
            intelligence: 20.0,
        };
        let stats = derive_stats(&attrs, PrimaryAttribute::Universal);
        // (20 + 20 + 20) * 0.7 = 42
        assert!((stats.bonus_damage - 42.0).abs() < 1e-4);
    }

    // ── Spell amplification ──

    #[test]
    fn test_spell_amp_from_int() {
        let attrs = ComputedAttributes {
            strength: 0.0,
            agility: 0.0,
            intelligence: 100.0,
        };
        let stats = derive_stats(&attrs, PrimaryAttribute::Intelligence);
        // 100 * 0.0007 = 0.07 (7% spell amp)
        assert!((stats.bonus_spell_amp - 0.07).abs() < 1e-4);
    }

    // ── Attack interval ──

    #[test]
    fn test_attack_interval() {
        // 1.7 BAT, 100 total IAS -> 1.7 / (100/100) = 1.7
        assert!((attack_interval(1.7, 100.0) - 1.7).abs() < 1e-4);
        // 1.7 BAT, 200 total IAS -> 1.7 / 2.0 = 0.85
        assert!((attack_interval(1.7, 200.0) - 0.85).abs() < 1e-4);
    }

    // ── Turn time ──

    #[test]
    fn test_turn_time() {
        // 180 degrees = PI radians at 0.6 rad/s -> PI / 0.6
        let time = turn_time(0.0, PI, 0.6);
        assert!((time - (PI / 0.6)).abs() < 1e-4);
    }

    // ── Default timings ──

    #[test]
    fn test_default_timings() {
        let t = HeroTimings::default();
        assert_eq!(t.turn_rate, 0.6);
        assert_eq!(t.attack_point, 0.3);
        assert_eq!(t.attack_backswing, 0.7);
        assert_eq!(t.base_attack_time, 1.7);
        assert_eq!(t.movement_speed, 300.0);
        assert_eq!(t.cast_point, 0.3);
        assert_eq!(t.vision_day, 1800.0);
        assert_eq!(t.vision_night, 800.0);
        assert_eq!(t.projectile_speed, 0.0);
        assert_eq!(t.attack_range, 150.0);
    }

    // ── Total stat helpers ──

    #[test]
    fn test_total_hp() {
        let stats = DerivedStats {
            bonus_hp: 440.0,
            ..Default::default()
        };
        assert!((total_hp(200.0, &stats) - 640.0).abs() < 1e-4);
    }

    #[test]
    fn test_total_mana() {
        let stats = DerivedStats {
            bonus_mana: 300.0,
            ..Default::default()
        };
        assert!((total_mana(75.0, &stats) - 375.0).abs() < 1e-4);
    }

    #[test]
    fn test_total_armor() {
        let stats = DerivedStats {
            bonus_armor: 5.01,
            ..Default::default()
        };
        assert!((total_armor(1.0, &stats) - 6.01).abs() < 1e-4);
    }

    #[test]
    fn test_total_damage() {
        let stats = DerivedStats {
            bonus_damage: 25.0,
            ..Default::default()
        };
        assert!((total_damage(50.0, &stats) - 75.0).abs() < 1e-4);
    }

    #[test]
    fn test_total_attack_speed_clamped() {
        let stats = DerivedStats {
            bonus_attack_speed: 800.0,
            ..Default::default()
        };
        // base 100 + 800 = 900, clamped to 700
        assert_eq!(total_attack_speed(100.0, &stats), 700.0);
        // base 0 + 10 = 10, clamped to 20
        let low_stats = DerivedStats {
            bonus_attack_speed: 10.0,
            ..Default::default()
        };
        assert_eq!(total_attack_speed(0.0, &low_stats), 20.0);
    }

    #[test]
    fn test_turn_time_shortest_path() {
        // Turning from 350 degrees to 10 degrees should take the short 20-degree path,
        // not the long 340-degree path.
        let facing = 350.0_f32.to_radians();
        let target = 10.0_f32.to_radians();
        let time = turn_time(facing, target, 0.6);
        let expected_delta = 20.0_f32.to_radians();
        assert!((time - expected_delta / 0.6).abs() < 1e-3);
    }
}
