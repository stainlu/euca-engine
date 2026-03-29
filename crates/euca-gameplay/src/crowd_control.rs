//! Dota 2 crowd control mechanics.
//!
//! Concrete CC types that disable specific unit capabilities,
//! plus dispel and spell immunity systems.
//!
//! This module is **pure data + logic** — it does not depend on ECS or World.
//! Game systems compose [`CcState`] as a component on entities and drive it
//! each tick.

use serde::{Deserialize, Serialize};

// ── CC type enum ──

/// Specific crowd control type with its behavioral restrictions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CcType {
    /// Disables movement, attacks, abilities, and items.
    Stun,
    /// Disables abilities only. Can still move and attack.
    Silence,
    /// Disables movement. Can still attack and use abilities.
    Root,
    /// Disables everything. Transforms unit model. Removes passives.
    Hex,
    /// Disables attacks only. Can still move and use abilities.
    Disarm,
    /// Disables passive abilities.
    Break,
    /// Disables item usage.
    Mute,
    /// Forced sleep. Any damage wakes the unit. Disables everything.
    Sleep,
    /// Forced to attack a specific target. Can't do anything else.
    Taunt { target_entity: u64 },
    /// Slowed movement speed by percentage (0.0 = no slow, 1.0 = full stop).
    Slow(f32),
    /// Hidden from enemy vision. Broken by attacks/abilities.
    Invisibility { fade_time: f32 },
}

// ── Disable flags ──

/// What this CC instance prevents the unit from doing.
///
/// Multiple flags can be combined via [`DisableFlags::merge`] to produce the
/// union of all active restrictions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct DisableFlags {
    pub prevents_movement: bool,
    pub prevents_attack: bool,
    pub prevents_abilities: bool,
    pub prevents_items: bool,
    pub prevents_passives: bool,
    pub forced_target: Option<u64>,
}

impl DisableFlags {
    /// Merge two flag sets: any `true` in either input stays `true`.
    /// `forced_target` takes the first `Some` value encountered.
    pub fn merge(self, other: DisableFlags) -> DisableFlags {
        DisableFlags {
            prevents_movement: self.prevents_movement || other.prevents_movement,
            prevents_attack: self.prevents_attack || other.prevents_attack,
            prevents_abilities: self.prevents_abilities || other.prevents_abilities,
            prevents_items: self.prevents_items || other.prevents_items,
            prevents_passives: self.prevents_passives || other.prevents_passives,
            forced_target: self.forced_target.or(other.forced_target),
        }
    }
}

impl CcType {
    /// Get the disable flags for this CC type.
    ///
    /// Each CC type maps to a fixed set of restrictions, matching Dota 2 semantics.
    pub fn disable_flags(&self) -> DisableFlags {
        match self {
            CcType::Stun => DisableFlags {
                prevents_movement: true,
                prevents_attack: true,
                prevents_abilities: true,
                prevents_items: true,
                ..DisableFlags::default()
            },
            CcType::Silence => DisableFlags {
                prevents_abilities: true,
                ..DisableFlags::default()
            },
            CcType::Root => DisableFlags {
                prevents_movement: true,
                ..DisableFlags::default()
            },
            CcType::Hex => DisableFlags {
                prevents_movement: true,
                prevents_attack: true,
                prevents_abilities: true,
                prevents_items: true,
                prevents_passives: true,
                forced_target: None,
            },
            CcType::Disarm => DisableFlags {
                prevents_attack: true,
                ..DisableFlags::default()
            },
            CcType::Break => DisableFlags {
                prevents_passives: true,
                ..DisableFlags::default()
            },
            CcType::Mute => DisableFlags {
                prevents_items: true,
                ..DisableFlags::default()
            },
            CcType::Sleep => DisableFlags {
                prevents_movement: true,
                prevents_attack: true,
                prevents_abilities: true,
                prevents_items: true,
                ..DisableFlags::default()
            },
            CcType::Taunt { target_entity } => DisableFlags {
                prevents_movement: true,
                prevents_abilities: true,
                prevents_items: true,
                forced_target: Some(*target_entity),
                ..DisableFlags::default()
            },
            CcType::Slow(_) => DisableFlags::default(),
            CcType::Invisibility { .. } => DisableFlags::default(),
        }
    }
}

// ── Dispel ──

/// How strong a dispel is needed to remove this effect.
///
/// Ordered from weakest to strongest — `PartialOrd`/`Ord` derives let you
/// compare dispel strengths directly (e.g. `effect.dispel_type <= strength`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DispelType {
    /// Removed by any dispel (most debuffs).
    BasicDispel,
    /// Requires strong dispel (BKB, Abaddon ult, etc.).
    StrongDispel,
    /// Cannot be dispelled (Doom, some ultimates).
    Undispellable,
}

// ── Crowd control instance ──

/// An active CC effect on a unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrowdControl {
    pub cc_type: CcType,
    /// Original duration before status resistance was applied.
    pub duration: f32,
    /// Time remaining before this effect expires.
    pub remaining: f32,
    /// Entity that applied this CC (for kill attribution, etc.).
    pub source_entity: Option<u64>,
    /// Dispel strength required to remove this effect.
    pub dispel_type: DispelType,
}

// ── Spell immunity ──

/// Spell immunity state (BKB-like).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellImmunity {
    /// Time remaining on spell immunity.
    pub remaining: f32,
    /// Whether this blocks magical damage.
    pub blocks_magical_damage: bool,
    /// Whether this blocks targeted spells.
    pub blocks_targeted_spells: bool,
}

// ── Status resistance ──

/// Status resistance reduces CC durations.
///
/// Applied multiplicatively when CC is first applied:
/// `effective_duration = duration * (1.0 - percent)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StatusResistance {
    /// 0.0 = no resistance, 0.5 = 50% shorter CC durations, 1.0 = immune.
    pub percent: f32,
}

impl Default for StatusResistance {
    fn default() -> Self {
        Self { percent: 0.0 }
    }
}

// ── Aggregate state ──

/// Collection of active CC effects on a unit.
///
/// This is the main component to attach to entities. Call [`CcState::apply_cc`]
/// to add new effects, [`CcState::remove_expired`] each tick, and query with
/// [`CcState::can_move`], [`CcState::can_attack`], etc.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CcState {
    pub effects: Vec<CrowdControl>,
    pub spell_immunity: Option<SpellImmunity>,
    pub status_resistance: StatusResistance,
}

impl CcState {
    /// Apply a CC effect, reducing its duration by status resistance.
    ///
    /// The effect's `remaining` field is set to `duration * (1.0 - resistance)`.
    /// The original `duration` field is preserved for UI display.
    pub fn apply_cc(&mut self, mut cc: CrowdControl) {
        let factor = 1.0 - self.status_resistance.percent.clamp(0.0, 1.0);
        cc.remaining = cc.duration * factor;
        self.effects.push(cc);
    }

    /// Tick down all durations by `dt` seconds and remove expired effects.
    ///
    /// Also ticks down spell immunity. Returns the number of effects removed.
    pub fn remove_expired(&mut self, dt: f32) -> usize {
        // Tick spell immunity.
        if let Some(si) = &mut self.spell_immunity {
            si.remaining -= dt;
            if si.remaining <= 0.0 {
                self.spell_immunity = None;
            }
        }

        // Tick CC effects.
        for effect in &mut self.effects {
            effect.remaining -= dt;
        }

        let before = self.effects.len();
        self.effects.retain(|e| e.remaining > 0.0);
        before - self.effects.len()
    }

    /// Dispel: remove all effects whose `dispel_type` is at or below `strength`.
    ///
    /// `DispelType::Undispellable` effects are never removed unless you pass
    /// `Undispellable` as the strength (which should not happen in normal gameplay).
    pub fn dispel(&mut self, strength: DispelType) {
        self.effects.retain(|e| e.dispel_type > strength);
    }

    /// Merge all active CC disable flags into a single combined set.
    pub fn combined_flags(&self) -> DisableFlags {
        self.effects
            .iter()
            .map(|e| e.cc_type.disable_flags())
            .fold(DisableFlags::default(), DisableFlags::merge)
    }

    // ── Type-specific queries ──

    /// Returns `true` if any active stun effect is present.
    pub fn is_stunned(&self) -> bool {
        self.effects
            .iter()
            .any(|e| matches!(e.cc_type, CcType::Stun))
    }

    /// Returns `true` if any active silence effect is present.
    pub fn is_silenced(&self) -> bool {
        self.effects
            .iter()
            .any(|e| matches!(e.cc_type, CcType::Silence))
    }

    /// Returns `true` if any active root effect is present.
    pub fn is_rooted(&self) -> bool {
        self.effects
            .iter()
            .any(|e| matches!(e.cc_type, CcType::Root))
    }

    /// Returns `true` if any active hex effect is present.
    pub fn is_hexed(&self) -> bool {
        self.effects
            .iter()
            .any(|e| matches!(e.cc_type, CcType::Hex))
    }

    /// Returns `true` if spell immunity is active.
    pub fn is_spell_immune(&self) -> bool {
        self.spell_immunity
            .as_ref()
            .is_some_and(|si| si.remaining > 0.0)
    }

    // ── Capability queries (derived from combined flags) ──

    /// Can the unit move? `false` if any active CC prevents movement.
    pub fn can_move(&self) -> bool {
        !self.combined_flags().prevents_movement
    }

    /// Can the unit attack? `false` if any active CC prevents attacks.
    pub fn can_attack(&self) -> bool {
        !self.combined_flags().prevents_attack
    }

    /// Can the unit cast abilities? `false` if any active CC prevents abilities.
    pub fn can_cast(&self) -> bool {
        !self.combined_flags().prevents_abilities
    }

    /// Can the unit use items? `false` if any active CC prevents item usage.
    pub fn can_use_items(&self) -> bool {
        !self.combined_flags().prevents_items
    }

    /// Duration of the longest remaining CC effect (for status bar display).
    /// Returns 0.0 if no effects are active.
    pub fn longest_cc_remaining(&self) -> f32 {
        self.effects
            .iter()
            .map(|e| e.remaining)
            .fold(0.0_f32, f32::max)
    }

    /// Interrupt all sleep effects (called when the unit takes damage).
    ///
    /// In Dota 2, any damage wakes a sleeping unit. This removes all
    /// `CcType::Sleep` effects immediately.
    pub fn interrupt_sleep(&mut self) {
        self.effects.retain(|e| !matches!(e.cc_type, CcType::Sleep));
    }

    /// Get the strongest active slow percentage (0.0 if no slows active).
    pub fn strongest_slow(&self) -> f32 {
        self.effects
            .iter()
            .filter_map(|e| match e.cc_type {
                CcType::Slow(pct) => Some(pct),
                _ => None,
            })
            .fold(0.0_f32, f32::max)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a basic CC effect with sensible defaults.
    fn make_cc(cc_type: CcType, duration: f32) -> CrowdControl {
        CrowdControl {
            cc_type,
            duration,
            remaining: duration,
            source_entity: None,
            dispel_type: DispelType::BasicDispel,
        }
    }

    #[test]
    fn test_stun_disables_all() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Stun, 2.0));

        let flags = state.combined_flags();
        assert!(flags.prevents_movement);
        assert!(flags.prevents_attack);
        assert!(flags.prevents_abilities);
        assert!(flags.prevents_items);
        assert!(!flags.prevents_passives);
    }

    #[test]
    fn test_silence_only_abilities() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Silence, 3.0));

        let flags = state.combined_flags();
        assert!(!flags.prevents_movement);
        assert!(!flags.prevents_attack);
        assert!(flags.prevents_abilities);
        assert!(!flags.prevents_items);
    }

    #[test]
    fn test_root_only_movement() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Root, 2.0));

        let flags = state.combined_flags();
        assert!(flags.prevents_movement);
        assert!(!flags.prevents_attack);
        assert!(!flags.prevents_abilities);
    }

    #[test]
    fn test_hex_disables_all_plus_passives() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Hex, 3.5));

        let flags = state.combined_flags();
        assert!(flags.prevents_movement);
        assert!(flags.prevents_attack);
        assert!(flags.prevents_abilities);
        assert!(flags.prevents_items);
        assert!(flags.prevents_passives);
    }

    #[test]
    fn test_disarm_only_attack() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Disarm, 4.0));

        let flags = state.combined_flags();
        assert!(!flags.prevents_movement);
        assert!(flags.prevents_attack);
        assert!(!flags.prevents_abilities);
    }

    #[test]
    fn test_break_only_passives() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Break, 5.0));

        let flags = state.combined_flags();
        assert!(!flags.prevents_movement);
        assert!(!flags.prevents_attack);
        assert!(!flags.prevents_abilities);
        assert!(!flags.prevents_items);
        assert!(flags.prevents_passives);
    }

    #[test]
    fn test_mute_only_items() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Mute, 3.0));

        let flags = state.combined_flags();
        assert!(!flags.prevents_movement);
        assert!(!flags.prevents_attack);
        assert!(!flags.prevents_abilities);
        assert!(flags.prevents_items);
    }

    #[test]
    fn test_slow_reduces_speed() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Slow(0.4), 3.0));

        // Slow does not set any disable flags.
        let flags = state.combined_flags();
        assert!(!flags.prevents_movement);

        // But we can query the slow percentage.
        assert!((state.strongest_slow() - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_basic_dispel() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Stun, 2.0)); // BasicDispel

        let mut strong = make_cc(CcType::Root, 3.0);
        strong.dispel_type = DispelType::StrongDispel;
        state.apply_cc(strong);

        let mut undispellable = make_cc(CcType::Silence, 5.0);
        undispellable.dispel_type = DispelType::Undispellable;
        state.apply_cc(undispellable);

        state.dispel(DispelType::BasicDispel);

        // Only BasicDispel should be removed.
        assert_eq!(state.effects.len(), 2);
        assert!(state.is_rooted());
        assert!(state.is_silenced());
        assert!(!state.is_stunned());
    }

    #[test]
    fn test_strong_dispel() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Stun, 2.0)); // BasicDispel

        let mut strong = make_cc(CcType::Root, 3.0);
        strong.dispel_type = DispelType::StrongDispel;
        state.apply_cc(strong);

        let mut undispellable = make_cc(CcType::Silence, 5.0);
        undispellable.dispel_type = DispelType::Undispellable;
        state.apply_cc(undispellable);

        state.dispel(DispelType::StrongDispel);

        // BasicDispel + StrongDispel removed, Undispellable stays.
        assert_eq!(state.effects.len(), 1);
        assert!(state.is_silenced());
        assert!(!state.is_stunned());
        assert!(!state.is_rooted());
    }

    #[test]
    fn test_status_resistance_reduces_duration() {
        let mut state = CcState::default();
        state.status_resistance = StatusResistance { percent: 0.5 };

        let cc = make_cc(CcType::Stun, 4.0);
        state.apply_cc(cc);

        // 50% resistance => 4.0 * 0.5 = 2.0 effective duration.
        assert!((state.effects[0].remaining - 2.0).abs() < f32::EPSILON);
        // Original duration preserved.
        assert!((state.effects[0].duration - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spell_immunity_blocks_magical() {
        let mut state = CcState::default();
        state.spell_immunity = Some(SpellImmunity {
            remaining: 5.0,
            blocks_magical_damage: true,
            blocks_targeted_spells: true,
        });

        assert!(state.is_spell_immune());

        // Tick past the duration.
        state.remove_expired(6.0);
        assert!(!state.is_spell_immune());
    }

    #[test]
    fn test_multiple_cc_combined_flags() {
        let mut state = CcState::default();
        // Stun: prevents move + attack + abilities + items.
        state.apply_cc(make_cc(CcType::Stun, 2.0));
        // Silence: prevents abilities (already covered by stun).
        state.apply_cc(make_cc(CcType::Silence, 3.0));
        // Break: prevents passives (not in stun).
        state.apply_cc(make_cc(CcType::Break, 4.0));

        let flags = state.combined_flags();
        assert!(flags.prevents_movement);
        assert!(flags.prevents_attack);
        assert!(flags.prevents_abilities);
        assert!(flags.prevents_items);
        assert!(flags.prevents_passives);
    }

    #[test]
    fn test_cc_expiry() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Stun, 1.0));
        state.apply_cc(make_cc(CcType::Silence, 3.0));

        // After 1.5s, stun should expire but silence remains.
        let removed = state.remove_expired(1.5);
        assert_eq!(removed, 1);
        assert!(!state.is_stunned());
        assert!(state.is_silenced());
        assert_eq!(state.effects.len(), 1);
    }

    #[test]
    fn test_sleep_interrupted_by_damage() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Sleep, 10.0));
        state.apply_cc(make_cc(CcType::Silence, 5.0));

        // Simulate taking damage — interrupt sleep only.
        state.interrupt_sleep();

        assert_eq!(state.effects.len(), 1);
        assert!(state.is_silenced());
        assert!(!state.combined_flags().prevents_movement);
    }

    #[test]
    fn test_taunt_forced_target() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Taunt { target_entity: 42 }, 3.0));

        let flags = state.combined_flags();
        assert_eq!(flags.forced_target, Some(42));
        assert!(flags.prevents_movement);
        assert!(flags.prevents_abilities);
        assert!(flags.prevents_items);
        // Taunt does NOT prevent attack (you're forced to attack the target).
        assert!(!flags.prevents_attack);
    }

    #[test]
    fn test_can_move_with_silence() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Silence, 5.0));

        assert!(state.can_move());
        assert!(state.can_attack());
        assert!(!state.can_cast());
        assert!(state.can_use_items());
    }

    #[test]
    fn test_longest_cc_remaining() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Stun, 1.0));
        state.apply_cc(make_cc(CcType::Silence, 5.0));
        state.apply_cc(make_cc(CcType::Root, 3.0));

        assert!((state.longest_cc_remaining() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_empty_state() {
        let state = CcState::default();

        assert!(state.can_move());
        assert!(state.can_attack());
        assert!(state.can_cast());
        assert!(state.can_use_items());
        assert!(!state.is_stunned());
        assert!(!state.is_spell_immune());
        assert!((state.longest_cc_remaining() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_status_resistance_clamped() {
        // Resistance > 1.0 should be clamped to 1.0 (immune).
        let mut state = CcState::default();
        state.status_resistance = StatusResistance { percent: 1.5 };

        state.apply_cc(make_cc(CcType::Stun, 4.0));
        assert!((state.effects[0].remaining - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dispel_type_ordering() {
        // Verify the Ord derivation gives the correct ordering.
        assert!(DispelType::BasicDispel < DispelType::StrongDispel);
        assert!(DispelType::StrongDispel < DispelType::Undispellable);
    }

    #[test]
    fn test_invisibility_no_disables() {
        let mut state = CcState::default();
        state.apply_cc(make_cc(CcType::Invisibility { fade_time: 0.3 }, 20.0));

        let flags = state.combined_flags();
        assert!(!flags.prevents_movement);
        assert!(!flags.prevents_attack);
        assert!(!flags.prevents_abilities);
        assert!(!flags.prevents_items);
        assert!(!flags.prevents_passives);
    }
}
