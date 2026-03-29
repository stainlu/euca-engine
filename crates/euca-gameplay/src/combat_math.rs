//! Dota 2-accurate combat math formulas.
//!
//! All formulas match Dota 2 7.34+ mechanics. Each function is pure
//! (no side effects) and extensively tested.

use serde::{Deserialize, Serialize};

/// Damage type determines which resistance applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DamageType {
    /// Reduced by armor.
    Physical,
    /// Reduced by magic resistance.
    Magical,
    /// Ignores both armor and magic resistance.
    Pure,
    /// Not reduced by anything, doesn't trigger on-damage effects.
    HpRemoval,
}

/// Critical strike result.
#[derive(Debug, Clone, Copy)]
pub struct CritResult {
    pub damage: f32,
    pub was_crit: bool,
}

/// Calculate physical damage reduction from armor.
///
/// Dota 2 formula: `reduction = armor * 0.06 / (1 + |armor| * 0.06)`
///
/// Positive armor reduces damage; negative armor amplifies it (returns negative reduction).
pub fn armor_reduction(armor: f32) -> f32 {
    armor * 0.06 / (1.0 + armor.abs() * 0.06)
}

/// Apply physical damage after armor reduction.
pub fn physical_damage_after_armor(raw_damage: f32, armor: f32) -> f32 {
    let reduction = armor_reduction(armor);
    raw_damage * (1.0 - reduction)
}

/// Calculate magic resistance from multiple sources (multiplicative stacking).
///
/// Base magic resistance is typically 0.25 (25%).
/// Additional sources stack multiplicatively:
/// `total = 1 - (1-base) * (1-bonus1) * (1-bonus2) ...`
pub fn magic_resistance_stacked(resistances: &[f32]) -> f32 {
    let mut pass_through = 1.0_f32;
    for &r in resistances {
        pass_through *= 1.0 - r;
    }
    1.0 - pass_through
}

/// Apply magical damage after magic resistance.
pub fn magical_damage_after_resistance(raw_damage: f32, resistances: &[f32]) -> f32 {
    let total_resistance = magic_resistance_stacked(resistances);
    raw_damage * (1.0 - total_resistance)
}

/// Calculate final damage considering damage type, armor, and magic resistances.
pub fn apply_damage(
    raw_damage: f32,
    damage_type: DamageType,
    armor: f32,
    magic_resistances: &[f32],
) -> f32 {
    match damage_type {
        DamageType::Physical => physical_damage_after_armor(raw_damage, armor),
        DamageType::Magical => magical_damage_after_resistance(raw_damage, magic_resistances),
        DamageType::Pure | DamageType::HpRemoval => raw_damage,
    }
}

/// Roll for critical strike.
///
/// Multiple crit sources use the highest multiplier if any procs.
/// Each source is `(chance, multiplier)` where chance is `0.0..=1.0`
/// and multiplier is e.g. `2.0` for 200% damage.
///
/// `rng_roll` is a uniform `[0.0, 1.0)` value provided by the caller.
pub fn roll_crit(base_damage: f32, sources: &[(f32, f32)], rng_roll: f32) -> CritResult {
    let mut best_multiplier = 1.0_f32;
    let mut any_crit = false;
    for &(chance, multiplier) in sources {
        if rng_roll < chance {
            if multiplier > best_multiplier {
                best_multiplier = multiplier;
            }
            any_crit = true;
        }
    }
    CritResult {
        damage: base_damage * best_multiplier,
        was_crit: any_crit,
    }
}

/// Roll for evasion. Returns `true` if the attack is evaded (missed).
///
/// Multiple evasion sources stack multiplicatively.
/// `has_true_strike` bypasses all evasion.
pub fn roll_evasion(evasion_sources: &[f32], has_true_strike: bool, rng_roll: f32) -> bool {
    if has_true_strike {
        return false;
    }
    let mut hit_chance = 1.0_f32;
    for &evasion in evasion_sources {
        hit_chance *= 1.0 - evasion;
    }
    let evasion_chance = 1.0 - hit_chance;
    rng_roll < evasion_chance
}

/// Calculate lifesteal healing from damage dealt.
pub fn lifesteal(damage_dealt: f32, lifesteal_percent: f32) -> f32 {
    damage_dealt * lifesteal_percent
}

/// Calculate spell lifesteal (reduced to 1/5 for AoE spells).
pub fn spell_lifesteal(damage_dealt: f32, lifesteal_percent: f32, is_aoe: bool) -> f32 {
    let effective = if is_aoe {
        lifesteal_percent / 5.0
    } else {
        lifesteal_percent
    };
    damage_dealt * effective
}

/// Calculate seconds between attacks.
///
/// - `base_attack_time` (BAT): default 1.7 for most heroes.
/// - `increased_attack_speed` (IAS): from agility + items + buffs.
///
/// Formula: `BAT / (1 + IAS / 100)`
///
/// IAS is clamped to `[-80, 600]` (Dota 2 caps).
pub fn attack_interval(base_attack_time: f32, increased_attack_speed: f32) -> f32 {
    let ias = increased_attack_speed.clamp(-80.0, 600.0);
    base_attack_time / (1.0 + ias / 100.0)
}

/// Calculate cleave damage (AoE physical from melee attack).
///
/// Cleave ignores armor of secondary targets — it is based on the attacker's
/// raw damage, not the damage dealt to the primary target.
pub fn cleave_damage(attack_damage: f32, cleave_percent: f32) -> f32 {
    attack_damage * cleave_percent
}

/// Calculate spell amplification bonus damage.
///
/// `spell_amp_percent` is a fraction, e.g. `0.15` for 15%.
pub fn amplified_spell_damage(base_damage: f32, spell_amp_percent: f32) -> f32 {
    base_damage * (1.0 + spell_amp_percent)
}

/// Effective HP against physical damage.
///
/// Answers: "How much raw physical damage is needed to kill this unit?"
pub fn effective_hp_physical(hp: f32, armor: f32) -> f32 {
    hp / (1.0 - armor_reduction(armor))
}

/// Effective HP against magical damage.
///
/// Answers: "How much raw magical damage is needed to kill this unit?"
pub fn effective_hp_magical(hp: f32, magic_resistances: &[f32]) -> f32 {
    let total_resistance = magic_resistance_stacked(magic_resistances);
    hp / (1.0 - total_resistance)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.01;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // ── Armor reduction ──

    #[test]
    fn test_armor_reduction_zero() {
        assert!(approx_eq(armor_reduction(0.0), 0.0));
    }

    #[test]
    fn test_armor_reduction_positive() {
        // 10 armor: 10 * 0.06 / (1 + 10 * 0.06) = 0.6 / 1.6 = 0.375
        assert!(approx_eq(armor_reduction(10.0), 0.375));
    }

    #[test]
    fn test_armor_reduction_negative() {
        // -10 armor: -10 * 0.06 / (1 + 10 * 0.06) = -0.6 / 1.6 = -0.375
        assert!(approx_eq(armor_reduction(-10.0), -0.375));
    }

    #[test]
    fn test_armor_reduction_high() {
        // 100 armor: 100 * 0.06 / (1 + 100 * 0.06) = 6.0 / 7.0 ≈ 0.857
        assert!(approx_eq(armor_reduction(100.0), 6.0 / 7.0));
    }

    // ── Physical damage ──

    #[test]
    fn test_physical_damage() {
        // 100 raw, 10 armor → 100 * (1 - 0.375) = 62.5
        assert!(approx_eq(physical_damage_after_armor(100.0, 10.0), 62.5));
    }

    #[test]
    fn test_physical_damage_negative_armor() {
        // -10 armor → 100 * (1 - (-0.375)) = 100 * 1.375 = 137.5
        assert!(approx_eq(physical_damage_after_armor(100.0, -10.0), 137.5));
    }

    // ── Magic resistance ──

    #[test]
    fn test_magic_resistance_single() {
        // 25% base → total = 0.25
        assert!(approx_eq(magic_resistance_stacked(&[0.25]), 0.25));
    }

    #[test]
    fn test_magic_resistance_stacking() {
        // 25% + 30% → 1 - (0.75 * 0.70) = 1 - 0.525 = 0.475
        assert!(approx_eq(magic_resistance_stacked(&[0.25, 0.30]), 0.475));
    }

    #[test]
    fn test_magic_resistance_empty() {
        // No resistances → 0% total
        assert!(approx_eq(magic_resistance_stacked(&[]), 0.0));
    }

    #[test]
    fn test_magical_damage_after_resistance() {
        // 100 raw, 25% base → 100 * 0.75 = 75
        assert!(approx_eq(
            magical_damage_after_resistance(100.0, &[0.25]),
            75.0
        ));
    }

    // ── apply_damage dispatch ──

    #[test]
    fn test_pure_damage_ignores_all() {
        let dmg = apply_damage(100.0, DamageType::Pure, 50.0, &[0.25, 0.30]);
        assert!(approx_eq(dmg, 100.0));
    }

    #[test]
    fn test_hp_removal() {
        let dmg = apply_damage(100.0, DamageType::HpRemoval, 50.0, &[0.25, 0.30]);
        assert!(approx_eq(dmg, 100.0));
    }

    #[test]
    fn test_apply_damage_physical() {
        let dmg = apply_damage(100.0, DamageType::Physical, 10.0, &[]);
        assert!(approx_eq(dmg, 62.5));
    }

    #[test]
    fn test_apply_damage_magical() {
        let dmg = apply_damage(100.0, DamageType::Magical, 0.0, &[0.25]);
        assert!(approx_eq(dmg, 75.0));
    }

    // ── Critical strike ──

    #[test]
    fn test_crit_no_proc() {
        // Roll 0.8, chance 0.3 → no crit
        let result = roll_crit(100.0, &[(0.3, 2.0)], 0.8);
        assert!(!result.was_crit);
        assert!(approx_eq(result.damage, 100.0));
    }

    #[test]
    fn test_crit_proc() {
        // Roll 0.1, chance 0.3 → crit at 2x
        let result = roll_crit(100.0, &[(0.3, 2.0)], 0.1);
        assert!(result.was_crit);
        assert!(approx_eq(result.damage, 200.0));
    }

    #[test]
    fn test_crit_multiple_highest() {
        // Two sources: (0.5, 1.5) and (0.5, 2.5). Roll 0.3 → both proc, use 2.5x
        let result = roll_crit(100.0, &[(0.5, 1.5), (0.5, 2.5)], 0.3);
        assert!(result.was_crit);
        assert!(approx_eq(result.damage, 250.0));
    }

    #[test]
    fn test_crit_partial_proc() {
        // (0.2, 1.5) and (0.5, 2.0). Roll 0.3 → only second procs
        let result = roll_crit(100.0, &[(0.2, 1.5), (0.5, 2.0)], 0.3);
        assert!(result.was_crit);
        assert!(approx_eq(result.damage, 200.0));
    }

    // ── Evasion ──

    #[test]
    fn test_evasion_miss() {
        // 35% evasion, roll 0.2 → 0.2 < 0.35 → evaded
        assert!(roll_evasion(&[0.35], false, 0.2));
    }

    #[test]
    fn test_evasion_hit() {
        // 35% evasion, roll 0.5 → 0.5 >= 0.35 → not evaded
        assert!(!roll_evasion(&[0.35], false, 0.5));
    }

    #[test]
    fn test_evasion_true_strike() {
        // True strike bypasses all evasion regardless of roll
        assert!(!roll_evasion(&[0.90], true, 0.01));
    }

    #[test]
    fn test_evasion_stacking() {
        // Two sources: 35% + 25%. Hit chance = 0.65 * 0.75 = 0.4875. Evasion = 0.5125
        // Roll 0.50 → 0.50 < 0.5125 → evaded
        assert!(roll_evasion(&[0.35, 0.25], false, 0.50));
        // Roll 0.52 → 0.52 >= 0.5125 → not evaded
        assert!(!roll_evasion(&[0.35, 0.25], false, 0.52));
    }

    // ── Lifesteal ──

    #[test]
    fn test_lifesteal() {
        assert!(approx_eq(lifesteal(100.0, 0.20), 20.0));
    }

    #[test]
    fn test_spell_lifesteal_single() {
        // Single-target: full rate
        assert!(approx_eq(spell_lifesteal(100.0, 0.10, false), 10.0));
    }

    #[test]
    fn test_spell_lifesteal_aoe() {
        // AoE: 1/5 rate → 100 * 0.10 / 5 = 2.0
        assert!(approx_eq(spell_lifesteal(100.0, 0.10, true), 2.0));
    }

    // ── Attack interval ──

    #[test]
    fn test_attack_interval() {
        // BAT 1.7, IAS 100 → 1.7 / (1 + 1.0) = 0.85
        assert!(approx_eq(attack_interval(1.7, 100.0), 0.85));
    }

    #[test]
    fn test_attack_interval_cap() {
        // IAS 700 clamped to 600 → 1.7 / (1 + 6.0) = 1.7 / 7.0 ≈ 0.2429
        assert!(approx_eq(attack_interval(1.7, 700.0), 1.7 / 7.0));
    }

    #[test]
    fn test_attack_interval_zero_ias() {
        // BAT 1.7, IAS 0 → 1.7 / 1.0 = 1.7
        assert!(approx_eq(attack_interval(1.7, 0.0), 1.7));
    }

    // ── Cleave ──

    #[test]
    fn test_cleave() {
        assert!(approx_eq(cleave_damage(100.0, 0.40), 40.0));
    }

    // ── Spell amplification ──

    #[test]
    fn test_spell_amp() {
        assert!(approx_eq(amplified_spell_damage(100.0, 0.15), 115.0));
    }

    // ── Effective HP ──

    #[test]
    fn test_effective_hp_physical() {
        // 1000 HP, 10 armor → reduction 0.375 → EHP = 1000 / 0.625 = 1600
        assert!(approx_eq(effective_hp_physical(1000.0, 10.0), 1600.0));
    }

    #[test]
    fn test_effective_hp_physical_zero_armor() {
        // 0 armor → EHP equals raw HP
        assert!(approx_eq(effective_hp_physical(1000.0, 0.0), 1000.0));
    }

    #[test]
    fn test_effective_hp_magical() {
        // 1000 HP, 25% magic resist → EHP = 1000 / 0.75 ≈ 1333.33
        assert!(approx_eq(effective_hp_magical(1000.0, &[0.25]), 1333.33));
    }

    #[test]
    fn test_effective_hp_magical_stacked() {
        // 1000 HP, 25% + 30% = 47.5% total → EHP = 1000 / 0.525 ≈ 1904.76
        assert!(approx_eq(
            effective_hp_magical(1000.0, &[0.25, 0.30]),
            1904.76
        ));
    }
}
