//! Stat block and damage resistance — data-driven entity attributes.
//!
//! Components: `BaseStats`, `ResolvedStats`, `DamageResistance`.
//! Systems: `stat_resolution_system`.
//!
//! Stats are arbitrary key-value pairs (no hardcoded stat names), making this
//! system genre-agnostic. Games define their own stat keys as data.

use std::collections::{HashMap, HashSet};

use euca_ecs::World;

use crate::inventory::StatModifiers;
use crate::status_effects::{ModifierOp, StatusEffects};

/// Entity's intrinsic stats — arbitrary key-value pairs.
///
/// Examples: `"max_health": 500.0`, `"attack_damage": 80.0`, `"physical_armor": 50.0`.
/// No stat names are hardcoded; games define them freely.
#[derive(Clone, Debug, Default)]
pub struct BaseStats(pub HashMap<String, f64>);

/// Computed stats after merging base values, equipment bonuses, and status
/// effect modifiers.
///
/// Resolution order per stat key:
/// 1. Start with the base value from `BaseStats` (default 0.0).
/// 2. Add equipment bonus from `StatModifiers` (additive).
/// 3. Apply active `StatusEffects` modifiers (Set / Add / Multiply priority).
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
/// Merges three data sources per stat key:
/// - `BaseStats`: intrinsic values (default 0.0 for missing keys).
/// - `StatModifiers`: additive equipment bonuses.
/// - `StatusEffects`: Set / Add / Multiply modifiers applied with the same
///   priority as [`crate::status_effects::effective_stat`].
pub fn stat_resolution_system(world: &mut World) {
    use euca_ecs::{Entity, Query};

    // Phase 1: snapshot all data we need. We clone to release borrows so we
    // can call `world.insert` in phase 2.
    struct Snapshot {
        entity: Entity,
        base: HashMap<String, f64>,
        equipment: Option<HashMap<String, f64>>,
        effects: Option<StatusEffects>,
    }

    let snapshots: Vec<Snapshot> = {
        let query = Query::<(Entity, &BaseStats)>::new(world);
        query
            .iter()
            .map(|(entity, bs)| {
                let equipment = world.get::<StatModifiers>(entity).map(|m| m.values.clone());
                let effects = world.get::<StatusEffects>(entity).cloned();
                Snapshot {
                    entity,
                    base: bs.0.clone(),
                    equipment,
                    effects,
                }
            })
            .collect()
    };

    // Phase 2: resolve each entity's stats and write `ResolvedStats`.
    for snap in snapshots {
        // Collect the union of all stat keys from base + equipment.
        let mut all_keys: HashSet<&str> = snap.base.keys().map(String::as_str).collect();
        if let Some(ref equip) = snap.equipment {
            all_keys.extend(equip.keys().map(String::as_str));
        }

        let mut resolved = HashMap::with_capacity(all_keys.len());

        for key in all_keys {
            // Step 1 + 2: base value + additive equipment bonus.
            let base = snap.base.get(key).copied().unwrap_or(0.0);
            let equip_bonus = snap
                .equipment
                .as_ref()
                .and_then(|m| m.get(key).copied())
                .unwrap_or(0.0);
            let equipped = base + equip_bonus;

            // Step 3: apply status effect modifiers (Set / Add / Multiply).
            let final_value = if let Some(ref effects) = snap.effects {
                apply_status_modifiers(equipped, effects, key)
            } else {
                equipped
            };

            resolved.insert(key.to_string(), final_value);
        }

        world.insert(snap.entity, ResolvedStats(resolved));
    }
}

/// Apply status effect modifiers for a single stat key.
///
/// Uses the same priority as [`crate::status_effects::effective_stat`]:
/// - Last `Set` value wins (overrides the base).
/// - All `Add` values are summed.
/// - All `Multiply` values are multiplied together.
/// - Result: `(set_or_base + adds) * multiplies`.
fn apply_status_modifiers(base: f64, effects: &StatusEffects, stat: &str) -> f64 {
    let mut last_set: Option<f64> = None;
    let mut add_sum: f64 = 0.0;
    let mut mul_product: f64 = 1.0;

    for effect in &effects.effects {
        for modifier in &effect.modifiers {
            if modifier.stat == stat {
                match modifier.op {
                    ModifierOp::Set => last_set = Some(modifier.value),
                    ModifierOp::Add => add_sum += modifier.value,
                    ModifierOp::Multiply => mul_product *= modifier.value,
                }
            }
        }
    }

    let value = last_set.unwrap_or(base);
    (value + add_sum) * mul_product
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::status_effects::{StackPolicy, StatModifier, StatusEffect};
    use euca_ecs::World;

    // ── Helper ──

    fn make_status_effect(tag: &str, modifiers: Vec<StatModifier>) -> StatusEffect {
        StatusEffect {
            tag: tag.to_string(),
            modifiers,
            duration: 10.0,
            remaining: 10.0,
            source: None,
            stack_policy: StackPolicy::Replace,
            tick_effect: None,
        }
    }

    // ── Existing tests (unchanged) ──

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

    // ── New tests: equipment modifiers ──

    #[test]
    fn equipment_adds_to_base_stats() {
        let mut world = World::new();

        let entity = world.spawn(BaseStats(
            [("attack".into(), 50.0), ("speed".into(), 10.0)]
                .into_iter()
                .collect(),
        ));
        world.insert(
            entity,
            StatModifiers {
                values: [("attack".into(), 20.0), ("speed".into(), 5.0)]
                    .into_iter()
                    .collect(),
            },
        );

        stat_resolution_system(&mut world);

        let resolved = world.get::<ResolvedStats>(entity).unwrap();
        assert_eq!(resolved.0.get("attack"), Some(&70.0)); // 50 + 20
        assert_eq!(resolved.0.get("speed"), Some(&15.0)); // 10 + 5
    }

    // ── New tests: status effect modifiers ──

    #[test]
    fn status_effect_multiplies_equipped_value() {
        let mut world = World::new();

        let entity = world.spawn(BaseStats([("speed".into(), 10.0)].into_iter().collect()));
        // Equipment adds 5 -> equipped = 15
        world.insert(
            entity,
            StatModifiers {
                values: [("speed".into(), 5.0)].into_iter().collect(),
            },
        );
        // Status effect multiplies by 2.0 -> final = 15 * 2 = 30
        world.insert(
            entity,
            StatusEffects {
                effects: vec![make_status_effect(
                    "haste",
                    vec![StatModifier {
                        stat: "speed".to_string(),
                        op: ModifierOp::Multiply,
                        value: 2.0,
                    }],
                )],
            },
        );

        stat_resolution_system(&mut world);

        let resolved = world.get::<ResolvedStats>(entity).unwrap();
        assert_eq!(resolved.0.get("speed"), Some(&30.0));
    }

    #[test]
    fn set_modifier_overrides_everything() {
        let mut world = World::new();

        let entity = world.spawn(BaseStats([("armor".into(), 100.0)].into_iter().collect()));
        // Equipment adds 50 -> equipped = 150
        world.insert(
            entity,
            StatModifiers {
                values: [("armor".into(), 50.0)].into_iter().collect(),
            },
        );
        // Set overrides to 0.0 -> final = (0.0) * 1.0 = 0.0
        world.insert(
            entity,
            StatusEffects {
                effects: vec![make_status_effect(
                    "shatter",
                    vec![StatModifier {
                        stat: "armor".to_string(),
                        op: ModifierOp::Set,
                        value: 0.0,
                    }],
                )],
            },
        );

        stat_resolution_system(&mut world);

        let resolved = world.get::<ResolvedStats>(entity).unwrap();
        assert_eq!(resolved.0.get("armor"), Some(&0.0));
    }

    #[test]
    fn empty_modifiers_yields_base_only() {
        let mut world = World::new();

        let entity = world.spawn(BaseStats(
            [("max_health".into(), 200.0)].into_iter().collect(),
        ));
        // Attach empty equipment and status effects.
        world.insert(entity, StatModifiers::default());
        world.insert(entity, StatusEffects::new());

        stat_resolution_system(&mut world);

        let resolved = world.get::<ResolvedStats>(entity).unwrap();
        assert_eq!(resolved.0.get("max_health"), Some(&200.0));
        assert_eq!(resolved.0.len(), 1);
    }

    #[test]
    fn equipment_only_keys_appear_in_resolved() {
        let mut world = World::new();

        // BaseStats has "health", equipment adds "armor" (not in base).
        let entity = world.spawn(BaseStats([("health".into(), 100.0)].into_iter().collect()));
        world.insert(
            entity,
            StatModifiers {
                values: [("armor".into(), 30.0)].into_iter().collect(),
            },
        );

        stat_resolution_system(&mut world);

        let resolved = world.get::<ResolvedStats>(entity).unwrap();
        assert_eq!(resolved.0.get("health"), Some(&100.0));
        // "armor" exists only in equipment — base defaults to 0.0, so final = 0 + 30 = 30.
        assert_eq!(resolved.0.get("armor"), Some(&30.0));
        assert_eq!(resolved.0.len(), 2);
    }
}
