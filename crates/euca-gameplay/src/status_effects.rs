//! Genre-agnostic status effect (modifier) system.
//!
//! Status effects are **pure data** — a tag, a list of stat modifiers, a duration,
//! and optional per-tick effects. There are no hardcoded Stun/Slow/Shield enums.
//! Instead, game data defines effects as tag + modifier combinations:
//!
//! - "stun" = tag "stun" + `StatModifier { stat: "can_move", op: Set, value: 0.0 }`
//! - "slow" = tag "slow" + `StatModifier { stat: "speed", op: Multiply, value: 0.5 }`
//! - "poison" = tag "debuff" + `TickEffect::DamagePerSecond(5.0)`
//!
//! Components: `StatusEffects` (a Vec of active effects on an entity).
//! Events: `StatusEffectExpired`.
//! Systems: `status_effect_tick_system` (ticks durations, applies tick effects, removes expired).

use euca_ecs::{Entity, Events, Query, World};

use crate::health::{DamageEvent, heal};

// ── Data types ──

/// How a stat is modified by this effect.
#[derive(Clone, Debug, PartialEq)]
pub enum ModifierOp {
    /// Overwrite the stat to an absolute value.
    Set,
    /// Add to the stat's base value.
    Add,
    /// Multiply the stat's current value.
    Multiply,
}

/// A single stat modification. Pure data — no behavior attached.
///
/// The `stat` field is a free-form string matching whatever stat names
/// the game defines (e.g. "speed", "can_move", "attack_damage").
#[derive(Clone, Debug)]
pub struct StatModifier {
    /// Name of the stat to modify (e.g. "speed", "can_move").
    pub stat: String,
    /// How to apply the modification.
    pub op: ModifierOp,
    /// The numeric value for the operation.
    pub value: f64,
}

/// Optional per-tick effect applied while the status effect is active.
#[derive(Clone, Debug)]
pub enum TickEffect {
    /// Deal damage every second.
    DamagePerSecond(f32),
    /// Heal every second.
    HealPerSecond(f32),
    /// Custom identifier for future Lua/rules dispatch.
    Custom(String),
}

/// Policy for what happens when the same effect tag is applied again.
#[derive(Clone, Debug, PartialEq)]
pub enum StackPolicy {
    /// New application replaces the existing effect (resets duration).
    Replace,
    /// Effects stack up to `max` instances.
    Stack { max: u32 },
}

/// A single status effect instance attached to an entity.
#[derive(Clone, Debug)]
pub struct StatusEffect {
    /// Identifier tag (e.g. "stun", "poison", "buff_speed"). Also used for
    /// cleanse filtering.
    pub tag: String,
    /// Stat modifications while this effect is active.
    pub modifiers: Vec<StatModifier>,
    /// Total duration in seconds. `f32::INFINITY` for permanent effects.
    pub duration: f32,
    /// Time remaining before expiry.
    pub remaining: f32,
    /// Entity that applied this effect (for attribution).
    pub source: Option<Entity>,
    /// What happens when the same tag is applied again.
    pub stack_policy: StackPolicy,
    /// Optional per-tick effect.
    pub tick_effect: Option<TickEffect>,
}

/// Component: all active status effects on an entity.
#[derive(Clone, Debug, Default)]
pub struct StatusEffects {
    pub effects: Vec<StatusEffect>,
}

impl StatusEffects {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Event emitted when a status effect expires naturally (duration runs out).
#[derive(Clone, Debug)]
pub struct StatusEffectExpired {
    /// Entity the effect was on.
    pub entity: Entity,
    /// Tag of the expired effect.
    pub tag: String,
}

// ── Application ──

/// Apply a status effect to an entity, respecting stack policy.
///
/// If the entity does not yet have a `StatusEffects` component, one is created.
pub fn apply_status_effect(world: &mut World, entity: Entity, effect: StatusEffect) {
    // Ensure the component exists.
    if world.get::<StatusEffects>(entity).is_none() {
        world.insert(entity, StatusEffects::new());
    }

    let effects = world.get_mut::<StatusEffects>(entity).unwrap();

    match &effect.stack_policy {
        StackPolicy::Replace => {
            // Remove all existing effects with the same tag, then add the new one.
            effects.effects.retain(|e| e.tag != effect.tag);
            effects.effects.push(effect);
        }
        StackPolicy::Stack { max } => {
            let count = effects
                .effects
                .iter()
                .filter(|e| e.tag == effect.tag)
                .count() as u32;
            if count < *max {
                effects.effects.push(effect);
            }
            // At max stacks: silently ignore (no replacement).
        }
    }
}

// ── Cleanse ──

/// Remove all status effects whose tag contains `filter` as a substring.
///
/// For example, `cleanse(world, entity, "debuff")` removes effects tagged
/// "debuff", "debuff_poison", etc.
pub fn cleanse(world: &mut World, entity: Entity, filter: &str) -> u32 {
    let Some(effects) = world.get_mut::<StatusEffects>(entity) else {
        return 0;
    };
    let before = effects.effects.len();
    effects.effects.retain(|e| !e.tag.contains(filter));
    (before - effects.effects.len()) as u32
}

// ── Tick system ──

/// Tick all status effects: decrement durations, apply tick effects, remove
/// expired effects, and emit `StatusEffectExpired` events.
pub fn status_effect_tick_system(world: &mut World, dt: f32) {
    // Phase 1: collect tick actions and expired effects.
    let mut damage_events: Vec<DamageEvent> = Vec::new();
    let mut heals: Vec<(Entity, f32)> = Vec::new();
    let mut expired: Vec<StatusEffectExpired> = Vec::new();

    {
        let query = Query::<(Entity, &mut StatusEffects)>::new(world);
        for (entity, effects) in query.iter() {
            for effect in effects.effects.iter_mut() {
                // Decrement duration.
                effect.remaining -= dt;

                // Apply tick effects (proportional to dt).
                match &effect.tick_effect {
                    Some(TickEffect::DamagePerSecond(dps)) => {
                        damage_events.push(DamageEvent {
                            target: entity,
                            amount: dps * dt,
                            source: effect.source,
                        });
                    }
                    Some(TickEffect::HealPerSecond(hps)) => {
                        heals.push((entity, hps * dt));
                    }
                    Some(TickEffect::Custom(_)) => {
                        // Custom tick effects are dispatched by external systems
                        // (e.g. Lua scripting). This system does nothing for them.
                    }
                    None => {}
                }

                // Collect expired effects for event emission.
                if effect.remaining <= 0.0 {
                    expired.push(StatusEffectExpired {
                        entity,
                        tag: effect.tag.clone(),
                    });
                }
            }

            // Remove expired effects.
            effects.effects.retain(|e| e.remaining > 0.0);
        }
    }

    // Phase 2: apply collected actions (avoids borrow conflicts with world).
    if let Some(events) = world.resource_mut::<Events>() {
        for dmg in damage_events {
            events.send(dmg);
        }
        for exp in expired {
            events.send(exp);
        }
    }

    for (entity, amount) in heals {
        heal(world, entity, amount);
    }
}

// ── Query helpers ──

/// Compute the effective value of a stat for an entity, starting from `base`
/// and applying all active modifiers in order: Set, then Add, then Multiply.
///
/// Returns `base` unchanged if the entity has no status effects or no modifiers
/// for the requested stat.
pub fn effective_stat(world: &World, entity: Entity, stat: &str, base: f64) -> f64 {
    let Some(effects) = world.get::<StatusEffects>(entity) else {
        return base;
    };

    // Single-pass accumulation: track last Set, sum of Adds, product of Multiplies.
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

    // Apply in priority order: last Set wins, then sum Adds, then product of Multiplies.
    let value = last_set.unwrap_or(base);
    (value + add_sum) * mul_product
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_world() -> World {
        let mut world = World::new();
        world.insert_resource(Events::default());
        world
    }

    fn make_effect(tag: &str, duration: f32) -> StatusEffect {
        StatusEffect {
            tag: tag.to_string(),
            modifiers: Vec::new(),
            duration,
            remaining: duration,
            source: None,
            stack_policy: StackPolicy::Replace,
            tick_effect: None,
        }
    }

    #[test]
    fn apply_adds_component_and_effect() {
        let mut world = test_world();
        let entity = world.spawn(crate::health::Health::new(100.0));

        apply_status_effect(&mut world, entity, make_effect("stun", 2.0));

        let effects = world.get::<StatusEffects>(entity).unwrap();
        assert_eq!(effects.effects.len(), 1);
        assert_eq!(effects.effects[0].tag, "stun");
        assert_eq!(effects.effects[0].remaining, 2.0);
    }

    #[test]
    fn replace_policy_replaces_existing() {
        let mut world = test_world();
        let entity = world.spawn(crate::health::Health::new(100.0));

        apply_status_effect(&mut world, entity, make_effect("stun", 2.0));
        // Apply again with longer duration — should replace.
        apply_status_effect(&mut world, entity, make_effect("stun", 5.0));

        let effects = world.get::<StatusEffects>(entity).unwrap();
        assert_eq!(effects.effects.len(), 1);
        assert_eq!(effects.effects[0].remaining, 5.0);
    }

    #[test]
    fn stack_policy_stacks_up_to_max() {
        let mut world = test_world();
        let entity = world.spawn(crate::health::Health::new(100.0));

        let mut effect = make_effect("poison", 3.0);
        effect.stack_policy = StackPolicy::Stack { max: 3 };

        apply_status_effect(&mut world, entity, effect.clone());
        apply_status_effect(&mut world, entity, effect.clone());
        apply_status_effect(&mut world, entity, effect.clone());
        // Fourth should be ignored.
        apply_status_effect(&mut world, entity, effect.clone());

        let effects = world.get::<StatusEffects>(entity).unwrap();
        assert_eq!(effects.effects.len(), 3);
    }

    #[test]
    fn expiry_removes_effect_and_emits_event() {
        let mut world = test_world();
        let entity = world.spawn(crate::health::Health::new(100.0));

        apply_status_effect(&mut world, entity, make_effect("buff", 1.0));

        // Tick past the duration.
        status_effect_tick_system(&mut world, 1.5);

        let effects = world.get::<StatusEffects>(entity).unwrap();
        assert!(effects.effects.is_empty());

        // Check expired event was emitted.
        let events = world.resource::<Events>().unwrap();
        let expired: Vec<_> = events.read::<StatusEffectExpired>().collect();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].tag, "buff");
        assert_eq!(expired[0].entity, entity);
    }

    #[test]
    fn cleanse_removes_matching_effects() {
        let mut world = test_world();
        let entity = world.spawn(crate::health::Health::new(100.0));

        apply_status_effect(&mut world, entity, make_effect("debuff_poison", 5.0));
        apply_status_effect(&mut world, entity, {
            let mut e = make_effect("debuff_slow", 5.0);
            e.stack_policy = StackPolicy::Stack { max: 5 };
            e
        });
        apply_status_effect(&mut world, entity, make_effect("buff_speed", 5.0));

        let removed = cleanse(&mut world, entity, "debuff");
        assert_eq!(removed, 2);

        let effects = world.get::<StatusEffects>(entity).unwrap();
        assert_eq!(effects.effects.len(), 1);
        assert_eq!(effects.effects[0].tag, "buff_speed");
    }

    #[test]
    fn tick_effect_dps_sends_damage_event() {
        let mut world = test_world();
        let entity = world.spawn(crate::health::Health::new(100.0));

        let mut effect = make_effect("burn", 5.0);
        effect.tick_effect = Some(TickEffect::DamagePerSecond(10.0));
        apply_status_effect(&mut world, entity, effect);

        status_effect_tick_system(&mut world, 1.0);

        let events = world.resource::<Events>().unwrap();
        let damage: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(damage.len(), 1);
        assert_eq!(damage[0].target, entity);
        assert_eq!(damage[0].amount, 10.0);
    }

    #[test]
    fn tick_effect_hps_heals() {
        let mut world = test_world();
        let entity = world.spawn(crate::health::Health {
            current: 50.0,
            max: 100.0,
        });

        let mut effect = make_effect("regen", 5.0);
        effect.tick_effect = Some(TickEffect::HealPerSecond(20.0));
        apply_status_effect(&mut world, entity, effect);

        status_effect_tick_system(&mut world, 1.0);

        let health = world.get::<crate::health::Health>(entity).unwrap();
        assert_eq!(health.current, 70.0);
    }

    #[test]
    fn source_tracking() {
        let mut world = test_world();
        let caster = world.spawn(crate::health::Health::new(100.0));
        let target = world.spawn(crate::health::Health::new(100.0));

        let mut effect = make_effect("curse", 3.0);
        effect.source = Some(caster);
        effect.tick_effect = Some(TickEffect::DamagePerSecond(5.0));
        apply_status_effect(&mut world, target, effect);

        status_effect_tick_system(&mut world, 1.0);

        let events = world.resource::<Events>().unwrap();
        let damage: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(damage[0].source, Some(caster));
    }

    #[test]
    fn effective_stat_computation() {
        let mut world = test_world();
        let entity = world.spawn(crate::health::Health::new(100.0));

        // Apply a speed buff: multiply by 1.5
        let mut speed_buff = make_effect("speed_buff", 10.0);
        speed_buff.modifiers.push(StatModifier {
            stat: "speed".to_string(),
            op: ModifierOp::Multiply,
            value: 1.5,
        });
        apply_status_effect(&mut world, entity, speed_buff);

        let speed = effective_stat(&world, entity, "speed", 10.0);
        assert_eq!(speed, 15.0);
    }

    #[test]
    fn effective_stat_set_overrides_base() {
        let mut world = test_world();
        let entity = world.spawn(crate::health::Health::new(100.0));

        let mut stun = make_effect("stun", 2.0);
        stun.modifiers.push(StatModifier {
            stat: "can_move".to_string(),
            op: ModifierOp::Set,
            value: 0.0,
        });
        apply_status_effect(&mut world, entity, stun);

        let can_move = effective_stat(&world, entity, "can_move", 1.0);
        assert_eq!(can_move, 0.0);
    }

    #[test]
    fn effective_stat_combined_ops() {
        let mut world = test_world();
        let entity = world.spawn(crate::health::Health::new(100.0));

        // Set base to 100, add 20, multiply by 0.5 => (100 + 20) * 0.5 = 60
        let mut effect = make_effect("complex", 10.0);
        effect.modifiers.push(StatModifier {
            stat: "attack".to_string(),
            op: ModifierOp::Set,
            value: 100.0,
        });
        effect.modifiers.push(StatModifier {
            stat: "attack".to_string(),
            op: ModifierOp::Add,
            value: 20.0,
        });
        effect.modifiers.push(StatModifier {
            stat: "attack".to_string(),
            op: ModifierOp::Multiply,
            value: 0.5,
        });
        apply_status_effect(&mut world, entity, effect);

        let attack = effective_stat(&world, entity, "attack", 50.0);
        assert_eq!(attack, 60.0);
    }

    #[test]
    fn no_effects_returns_base() {
        let world = test_world();
        let entity = Entity::from_raw(999, 0);
        assert_eq!(effective_stat(&world, entity, "speed", 10.0), 10.0);
    }

    #[test]
    fn cleanse_on_entity_without_effects_returns_zero() {
        let mut world = test_world();
        let entity = world.spawn(crate::health::Health::new(100.0));
        assert_eq!(cleanse(&mut world, entity, "debuff"), 0);
    }
}
