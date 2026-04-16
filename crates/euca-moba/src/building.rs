//! Dota 2 tower and building system.
//!
//! Covers building types (towers, barracks, ancient, fountain, effigy, outpost),
//! backdoor protection, fortification (glyph), tower aggro priority, building
//! bounties, and barracks destruction effects on creep strength.
//!
//! This module is pure data and logic — no ECS dependency. Wire it into your
//! ECS by reading/writing these structs as components or resources.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Building types
// ---------------------------------------------------------------------------

/// Every distinct building kind in a Dota 2-style map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BuildingType {
    /// Outer lane tower: 1800 HP, 10 armor.
    Tier1Tower,
    /// Inner lane tower: 2500 HP, 15 armor.
    Tier2Tower,
    /// Base tower (high-ground): 2500 HP, 22 armor.
    Tier3Tower,
    /// Ancient guard tower: 2600 HP, 25 armor.
    Tier4Tower,
    /// Melee barracks: 2200 HP, 13 armor.
    MeleeBarracks,
    /// Ranged barracks: 1300 HP, 5 armor.
    RangedBarracks,
    /// The Ancient — destroy to win: 4500 HP, 15 armor.
    Ancient,
    /// Fountain — invulnerable, heals 6% HP/s + 4% mana/s to allies.
    Fountain,
    /// Destroyable cosmetic effigy: 100 HP.
    Effigy,
    /// Capturable objective that grants periodic XP.
    Outpost,
}

/// Which lane a building belongs to (towers and barracks are lane-specific).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lane {
    Top,
    Mid,
    Bot,
}

// ---------------------------------------------------------------------------
// Building stats
// ---------------------------------------------------------------------------

/// Runtime state for a single building instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingStats {
    pub building_type: BuildingType,
    pub max_hp: f32,
    pub current_hp: f32,
    pub armor: f32,
    pub team: u32,
    pub lane: Option<Lane>,
    /// Towers deal damage; barracks and the ancient do not.
    pub attack_damage: Option<f32>,
    pub attack_range: Option<f32>,
    pub attack_speed: Option<f32>,
    pub is_alive: bool,
}

/// Factory: create a `BuildingStats` with canonical Dota 2 values.
///
/// `team` and `lane` are caller-supplied; everything else comes from the type.
pub fn building_stats(building_type: BuildingType, team: u32, lane: Option<Lane>) -> BuildingStats {
    let (max_hp, armor, attack_damage, attack_range, attack_speed) = match building_type {
        BuildingType::Tier1Tower => (1800.0, 10.0, Some(110.0), Some(7.0), Some(1.0)),
        BuildingType::Tier2Tower => (2500.0, 15.0, Some(150.0), Some(7.0), Some(1.0)),
        BuildingType::Tier3Tower => (2500.0, 22.0, Some(175.0), Some(7.0), Some(1.0)),
        BuildingType::Tier4Tower => (2600.0, 25.0, Some(195.0), Some(7.0), Some(1.0)),
        BuildingType::MeleeBarracks => (2200.0, 13.0, None, None, None),
        BuildingType::RangedBarracks => (1300.0, 5.0, None, None, None),
        BuildingType::Ancient => (4500.0, 15.0, None, None, None),
        BuildingType::Fountain => (f32::INFINITY, 0.0, Some(300.0), Some(12.0), Some(0.15)),
        BuildingType::Effigy => (100.0, 0.0, None, None, None),
        BuildingType::Outpost => (0.0, 0.0, None, None, None), // not destroyable
    };

    BuildingStats {
        building_type,
        max_hp,
        current_hp: max_hp,
        armor,
        team,
        lane,
        attack_damage,
        attack_range,
        attack_speed,
        is_alive: true,
    }
}

// ---------------------------------------------------------------------------
// Backdoor protection
// ---------------------------------------------------------------------------

/// Buildings behind the front line regenerate HP and take vastly reduced damage
/// when no enemy creeps are nearby.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackdoorProtection {
    /// Whether protection is currently active (no enemy creeps within radius).
    pub active: bool,
    /// Radius in game units to scan for enemy creeps.
    pub check_radius: f32,
    /// Fraction of damage *reduced* while active (0.75 means take only 25%).
    pub damage_reduction: f32,
    /// HP regenerated per second while protection is active.
    pub hp_regen_per_sec: f32,
}

impl Default for BackdoorProtection {
    fn default() -> Self {
        Self {
            active: true,
            check_radius: 9.0,
            damage_reduction: 0.75,
            hp_regen_per_sec: 90.0,
        }
    }
}

/// Toggle backdoor protection based on whether enemy creeps are within range.
///
/// The caller is responsible for the spatial query and passes the result in.
pub fn update_backdoor_protection(
    protection: &mut BackdoorProtection,
    enemy_creeps_in_range: bool,
) {
    protection.active = !enemy_creeps_in_range;
}

/// Returns the damage multiplier to apply when hitting a building.
///
/// * Protection active  -> `1.0 - damage_reduction` (e.g. 0.25).
/// * Protection inactive -> `1.0` (full damage).
pub fn backdoor_damage_modifier(protection: &BackdoorProtection) -> f32 {
    if protection.active {
        1.0 - protection.damage_reduction
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Fortification (Glyph of Fortification)
// ---------------------------------------------------------------------------

/// Team-wide building invulnerability on a long cooldown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fortification {
    /// Total cooldown in seconds (300 = 5 minutes).
    pub cooldown: f32,
    /// Seconds remaining until the ability is available again.
    pub remaining_cooldown: f32,
    /// How long buildings are invulnerable when activated (seconds).
    pub duration: f32,
    /// Seconds remaining on the current activation (0 = not active).
    pub active_remaining: f32,
}

impl Default for Fortification {
    fn default() -> Self {
        Self {
            cooldown: 300.0,
            remaining_cooldown: 0.0,
            duration: 5.0,
            active_remaining: 0.0,
        }
    }
}

/// Try to activate fortification. Fails if still on cooldown.
pub fn activate_fortification(fort: &mut Fortification) -> Result<(), &'static str> {
    if fort.remaining_cooldown > 0.0 {
        return Err("Fortification is on cooldown");
    }
    fort.active_remaining = fort.duration;
    fort.remaining_cooldown = fort.cooldown;
    Ok(())
}

/// Advance timers each frame.
pub fn tick_fortification(fort: &mut Fortification, dt: f32) {
    if fort.active_remaining > 0.0 {
        fort.active_remaining = (fort.active_remaining - dt).max(0.0);
    }
    if fort.remaining_cooldown > 0.0 {
        fort.remaining_cooldown = (fort.remaining_cooldown - dt).max(0.0);
    }
}

/// Returns `true` while the fortification buff is active on buildings.
pub fn is_building_invulnerable(fort: &Fortification) -> bool {
    fort.active_remaining > 0.0
}

// ---------------------------------------------------------------------------
// Tower aggro (pure-data version)
// ---------------------------------------------------------------------------

/// Tracks which entity a tower is currently attacking and why.
///
/// This is a lightweight, ECS-free representation. The ECS-level
/// `tower_aggro_system` in `tower_aggro.rs` handles the actual World mutation;
/// this struct is useful for standalone logic and tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TowerAggro {
    /// Entity ID of the current attack target (if any).
    pub current_target: Option<u64>,
    /// Entity ID of a hero that attacked an ally under this tower.
    /// Takes priority over `current_target`.
    pub priority_target: Option<u64>,
    /// Range within which the tower acquires targets.
    pub aggro_range: f32,
    /// Cooldown preventing rapid target switching (seconds remaining).
    pub aggro_switch_cooldown: f32,
}

impl Default for TowerAggro {
    fn default() -> Self {
        Self {
            current_target: None,
            priority_target: None,
            aggro_range: 7.0,
            aggro_switch_cooldown: 0.0,
        }
    }
}

/// Decide which entity the tower should attack.
///
/// * If a hero is attacking an ally (`hero_attacking_ally`), it gets priority.
/// * Otherwise fall back to `closest_enemy`.
pub fn update_tower_aggro(
    aggro: &mut TowerAggro,
    hero_attacking_ally: Option<u64>,
    closest_enemy: Option<u64>,
) {
    aggro.priority_target = hero_attacking_ally;

    if let Some(hero) = hero_attacking_ally {
        aggro.current_target = Some(hero);
    } else if let Some(enemy) = closest_enemy {
        aggro.current_target = Some(enemy);
    } else {
        aggro.current_target = None;
    }
}

// ---------------------------------------------------------------------------
// Building bounties
// ---------------------------------------------------------------------------

/// Gold reward split among the killing team when a building is destroyed.
pub fn tower_bounty(building_type: BuildingType) -> u32 {
    match building_type {
        BuildingType::Tier1Tower => 500,
        BuildingType::Tier2Tower => 550,
        BuildingType::Tier3Tower => 600,
        BuildingType::Tier4Tower => 650,
        BuildingType::MeleeBarracks => 225,
        BuildingType::RangedBarracks => 150,
        BuildingType::Ancient => 0,  // game over — no bounty
        BuildingType::Fountain => 0, // invulnerable
        BuildingType::Effigy => 50,
        BuildingType::Outpost => 0, // not destroyed, captured
    }
}

// ---------------------------------------------------------------------------
// Barracks destruction effects
// ---------------------------------------------------------------------------

/// Describes how lane creeps are upgraded after barracks fall.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreepEffect {
    /// The lane whose creeps are affected.
    pub lane: Lane,
    /// Whether melee creeps in this lane become super creeps.
    pub super_melee: bool,
    /// Whether ranged creeps in this lane become super creeps.
    pub super_ranged: bool,
    /// Whether creeps become mega creeps (all barracks in all lanes destroyed).
    pub mega: bool,
}

/// Determine the creep upgrade for `lane` after a barracks falls.
///
/// Pass the type of barracks that was just destroyed. In Dota 2, destroying
/// the melee barracks upgrades the lane's melee creeps to super creeps, and
/// vice-versa for ranged.
///
/// Mega creeps require *all six* barracks to fall; that decision lives at a
/// higher level. This function only reports the immediate single-barracks
/// effect.
pub fn barracks_destroyed_effect(building_type: BuildingType, lane: Lane) -> CreepEffect {
    match building_type {
        BuildingType::MeleeBarracks => CreepEffect {
            lane,
            super_melee: true,
            super_ranged: false,
            mega: false,
        },
        BuildingType::RangedBarracks => CreepEffect {
            lane,
            super_melee: false,
            super_ranged: true,
            mega: false,
        },
        // Non-barracks types produce no creep effect.
        _ => CreepEffect {
            lane,
            super_melee: false,
            super_ranged: false,
            mega: false,
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Building stats factory ──

    #[test]
    fn test_building_stats_tier1() {
        let stats = building_stats(BuildingType::Tier1Tower, 1, Some(Lane::Mid));
        assert_eq!(stats.max_hp, 1800.0);
        assert_eq!(stats.current_hp, 1800.0);
        assert_eq!(stats.armor, 10.0);
        assert!(stats.attack_damage.is_some());
        assert_eq!(stats.attack_range, Some(7.0));
        assert!(stats.is_alive);
    }

    #[test]
    fn test_building_stats_ancient() {
        let stats = building_stats(BuildingType::Ancient, 2, None);
        assert_eq!(stats.max_hp, 4500.0);
        assert_eq!(stats.armor, 15.0);
        assert!(stats.attack_damage.is_none(), "Ancient does not attack");
    }

    #[test]
    fn test_building_stats_barracks_no_attack() {
        let melee = building_stats(BuildingType::MeleeBarracks, 1, Some(Lane::Top));
        let ranged = building_stats(BuildingType::RangedBarracks, 1, Some(Lane::Top));
        assert!(melee.attack_damage.is_none());
        assert!(ranged.attack_damage.is_none());
        assert_eq!(melee.max_hp, 2200.0);
        assert_eq!(melee.armor, 13.0);
        assert_eq!(ranged.max_hp, 1300.0);
        assert_eq!(ranged.armor, 5.0);
    }

    // ── Backdoor protection ──

    #[test]
    fn test_backdoor_active() {
        let protection = BackdoorProtection::default();
        // No creeps nearby -> active -> 75% damage reduction.
        let modifier = backdoor_damage_modifier(&protection);
        assert!((modifier - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_backdoor_inactive() {
        let mut protection = BackdoorProtection::default();
        update_backdoor_protection(&mut protection, true); // creeps nearby
        assert!(!protection.active);
        let modifier = backdoor_damage_modifier(&protection);
        assert!((modifier - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_backdoor_regen() {
        let protection = BackdoorProtection::default();
        assert!(protection.active);
        // When active, regen should be 90 HP/s.
        assert_eq!(protection.hp_regen_per_sec, 90.0);
    }

    // ── Fortification ──

    #[test]
    fn test_fortification_activate() {
        let mut fort = Fortification::default();
        let result = activate_fortification(&mut fort);
        assert!(result.is_ok());
        assert_eq!(fort.active_remaining, 5.0);
        assert!(is_building_invulnerable(&fort));
    }

    #[test]
    fn test_fortification_cooldown() {
        let mut fort = Fortification::default();
        activate_fortification(&mut fort).unwrap();
        assert_eq!(fort.remaining_cooldown, 300.0);
    }

    #[test]
    fn test_fortification_on_cooldown() {
        let mut fort = Fortification::default();
        activate_fortification(&mut fort).unwrap();
        let result = activate_fortification(&mut fort);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Fortification is on cooldown");
    }

    #[test]
    fn test_fortification_tick_expires() {
        let mut fort = Fortification::default();
        activate_fortification(&mut fort).unwrap();
        // Tick past the full duration.
        tick_fortification(&mut fort, 6.0);
        assert!(!is_building_invulnerable(&fort));
        assert_eq!(fort.active_remaining, 0.0);
    }

    // ── Tower aggro ──

    #[test]
    fn test_tower_aggro_hero_priority() {
        let mut aggro = TowerAggro::default();
        // A hero (id 42) is attacking an ally; closest enemy is a creep (id 7).
        update_tower_aggro(&mut aggro, Some(42), Some(7));
        assert_eq!(aggro.current_target, Some(42));
        assert_eq!(aggro.priority_target, Some(42));
    }

    #[test]
    fn test_tower_aggro_fallback() {
        let mut aggro = TowerAggro::default();
        // No hero attacking ally; closest enemy is creep (id 7).
        update_tower_aggro(&mut aggro, None, Some(7));
        assert_eq!(aggro.current_target, Some(7));
        assert!(aggro.priority_target.is_none());
    }

    // ── Building bounties ──

    #[test]
    fn test_tower_bounty() {
        assert_eq!(tower_bounty(BuildingType::Tier1Tower), 500);
        assert_eq!(tower_bounty(BuildingType::Tier2Tower), 550);
        assert_eq!(tower_bounty(BuildingType::Tier3Tower), 600);
        assert_eq!(tower_bounty(BuildingType::Tier4Tower), 650);
        assert_eq!(tower_bounty(BuildingType::MeleeBarracks), 225);
        assert_eq!(tower_bounty(BuildingType::RangedBarracks), 150);
        assert_eq!(tower_bounty(BuildingType::Ancient), 0);
    }

    // ── Barracks destruction ──

    #[test]
    fn test_barracks_destroyed_super_creeps() {
        let effect = barracks_destroyed_effect(BuildingType::MeleeBarracks, Lane::Bot);
        assert_eq!(effect.lane, Lane::Bot);
        assert!(effect.super_melee);
        assert!(!effect.super_ranged);
        assert!(!effect.mega);

        let effect = barracks_destroyed_effect(BuildingType::RangedBarracks, Lane::Top);
        assert!(effect.super_ranged);
        assert!(!effect.super_melee);
    }

    #[test]
    fn test_non_barracks_no_creep_effect() {
        let effect = barracks_destroyed_effect(BuildingType::Tier1Tower, Lane::Mid);
        assert!(!effect.super_melee);
        assert!(!effect.super_ranged);
        assert!(!effect.mega);
    }
}
