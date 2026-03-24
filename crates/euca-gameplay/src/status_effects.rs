//! Status effects (modifiers) — data-driven stat modification over time.
//!
//! Components: `StatusEffects`.
//! Events: `StatusEffectExpired`.
//! Systems: `status_effect_tick_system`.
//!
//! Status effects are pure data: a tag, a set of stat modifiers, a duration,
//! and optional tick effects.  There are no hardcoded Stun/Slow/Shield enums —
//! those emerge from tag + modifier combinations defined by game data.
//!
//! ```text
//! // Stun = can't move
//! StatusEffect { tag: "stun", modifiers: [can_move:Set:0], duration: 2.0, .. }
//!
//! // Slow = half speed
//! StatusEffect { tag: "slow_debuff", modifiers: [speed:Multiply:0.5], duration: 5.0, .. }
//! ```

use euca_ecs::{Entity, Events, World};

use crate::health::{self, DamageEvent};

// ── Data types ──

/// How a modifier changes a stat value.
#[derive(Clone, Debug, PartialEq)]
pub enum ModifierOp {
    /// Replace the stat value entirely.
    Set,
    /// Add to (or subtract from) the stat value.
    Add,
    /// Multiply the stat value.
    Multiply,
}

/// A single stat modification: which stat, what operation, what value.
///
/// Generic over stat names — the gameplay layer interprets them.
/// Examples: `("can_move", Set, 0.0)` = stun, `("speed", Multiply, 0.5)` = slow.
#[derive(Clone, Debug)]
pub struct StatModifier {
    /// Stat name (e.g. "speed", "can_move", "armor").
    pub stat: String,
    /// How to apply the value.
    pub op: ModifierOp,
    /// The modifier value.
    pub value: f64,
}

/// What happens every tick while the effect is active.
#[derive(Clone, Debug)]
pub enum TickEffect {
    /// Deal damage per second to the affected entity.
    DamagePerSecond(f32),
    /// Heal per second on the affected entity.
    HealPerSecond(f32),
    /// Game-layer extension point — an opaque tag processed by user systems.
    Custom(String),
}

/// How multiple applications of the same effect tag interact.
#[derive(Clone, Debug, PartialEq)]
pub enum StackPolicy {
    /// New application replaces the existing effect (resets duration).
    Replace,
    /// Effects stack up to a maximum count.
    Stack { max: u32 },
}

/// A single status effect instance on an entity.
#[derive(Clone, Debug)]
pub struct StatusEffect {
    /// Identifier for this effect type (e.g. "stun", "poison_debuff", "speed_buff").
    pub tag: String,
    /// Stat modifications applied while this effect is active.
    pub modifiers: Vec<StatModifier>,
    /// Total duration in seconds.
    pub duration: f32,
    /// Time remaining before expiry.
    pub remaining: f32,
    /// Entity that applied this effect (for attribution).
    pub source: Option<Entity>,
    /// How duplicate applications are handled.
    pub stack_policy: StackPolicy,
    /// Optional per-tick effect (DPS, HPS, or custom).
    pub tick_effect: Option<TickEffect>,
}

/// Component: all active status effects on an entity.
#[derive(Clone, Debug, Default)]
pub struct StatusEffects {
    pub effects: Vec<StatusEffect>,
}

// ── Events ──

/// Emitted when a status effect expires (duration reaches zero).
#[derive(Clone, Debug)]
pub struct StatusEffectExpired {
    /// The entity the effect was on.
    pub entity: Entity,
    /// The tag of the expired effect.
    pub tag: String,
}

// ── Application ──

/// Apply a status effect to an entity, respecting stack policy.
pub fn apply_effect(world: &mut World, entity: Entity, effect: StatusEffect) {
    if world.get::<StatusEffects>(entity).is_none() {
        world.insert(entity, StatusEffects::default());
    }

    let effects = world
        .get_mut::<StatusEffects>(entity)
        .expect("just inserted");

    match &effect.stack_policy {
        StackPolicy::Replace => {
            // Remove any existing effect with the same tag, then add the new one.
            effects.effects.retain(|e| e.tag != effect.tag);
            effects.effects.push(effect);
        }
        StackPolicy::Stack { max } => {
            let current_count = effects
                .effects
                .iter()
                .filter(|e| e.tag == effect.tag)
                .count() as u32;
            if current_count < *max {
                effects.effects.push(effect);
            }
            // At max stacks: silently ignore the new application.
        }
    }
}

// ── Cleanse ──

/// Remove all effects whose tag contains the given filter substring.
///
/// Example: `cleanse(world, entity, "debuff")` removes "poison_debuff", "slow_debuff", etc.
pub fn cleanse(world: &mut World, entity: Entity, filter: &str) -> u32 {
    let Some(effects) = world.get_mut::<StatusEffects>(entity) else {
        return 0;
    };
    let before = effects.effects.len();
    effects.effects.retain(|e| !e.tag.contains(filter));
    (before - effects.effects.len()) as u32
}

// ── Tick system ──

/// Tick all status effects: decrement durations, apply tick effects, remove expired, emit events.
pub fn status_effect_tick_system(world: &mut World, dt: f32) {
    // Collect entities with StatusEffects to avoid borrow conflicts.
    let entities: Vec<Entity> = {
        let query = euca_ecs::Query::<(euca_ecs::Entity, &StatusEffects)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    let mut expired_events: Vec<StatusEffectExpired> = Vec::new();
    let mut damage_events: Vec<DamageEvent> = Vec::new();
    let mut heals: Vec<(Entity, f32)> = Vec::new();

    for entity in entities {
        let Some(status) = world.get_mut::<StatusEffects>(entity) else {
            continue;
        };

        // Process each effect: apply tick, decrement duration, collect expired.
        for effect in &mut status.effects {
            // Apply tick effects before decrementing duration.
            if let Some(ref tick) = effect.tick_effect {
                match tick {
                    TickEffect::DamagePerSecond(dps) => {
                        damage_events.push(DamageEvent {
                            target: entity,
                            amount: dps * dt,
                            source: effect.source,
                        });
                    }
                    TickEffect::HealPerSecond(hps) => {
                        heals.push((entity, hps * dt));
                    }
                    TickEffect::Custom(_) => {
                        // Custom tick effects are handled by game-layer systems.
                    }
                }
            }

            effect.remaining -= dt;
        }

        // Collect expired tags before removing.
        let expired_tags: Vec<String> = status
            .effects
            .iter()
            .filter(|e| e.remaining <= 0.0)
            .map(|e| e.tag.clone())
            .collect();

        // Remove expired effects.
        status.effects.retain(|e| e.remaining > 0.0);

        for tag in expired_tags {
            expired_events.push(StatusEffectExpired { entity, tag });
        }
    }

    // Apply heals.
    for (entity, amount) in heals {
        health::heal(world, entity, amount);
    }

    // Emit events.
    if let Some(events) = world.resource_mut::<Events>() {
        for event in damage_events {
            events.send(event);
        }
        for event in expired_events {
            events.send(event);
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::Health;
    use euca_ecs::Events;

    fn setup() -> World {
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
    fn apply_single_effect() {
        let mut world = setup();
        let entity = world.spawn(Health::new(100.0));

        let effect = make_effect("stun", 2.0);
        apply_effect(&mut world, entity, effect);

        let status = world.get::<StatusEffects>(entity).unwrap();
        assert_eq!(status.effects.len(), 1);
        assert_eq!(status.effects[0].tag, "stun");
        assert_eq!(status.effects[0].remaining, 2.0);
    }

    #[test]
    fn replace_policy_replaces_existing() {
        let mut world = setup();
        let entity = world.spawn(Health::new(100.0));

        apply_effect(&mut world, entity, make_effect("stun", 2.0));
        apply_effect(&mut world, entity, make_effect("stun", 5.0));

        let status = world.get::<StatusEffects>(entity).unwrap();
        assert_eq!(status.effects.len(), 1);
        assert_eq!(status.effects[0].remaining, 5.0);
    }

    #[test]
    fn stack_policy_stacks_up_to_max() {
        let mut world = setup();
        let entity = world.spawn(Health::new(100.0));

        let mut effect = make_effect("poison", 3.0);
        effect.stack_policy = StackPolicy::Stack { max: 3 };

        apply_effect(&mut world, entity, effect.clone());
        apply_effect(&mut world, entity, effect.clone());
        apply_effect(&mut world, entity, effect.clone());
        // This one should be silently dropped.
        apply_effect(&mut world, entity, effect);

        let status = world.get::<StatusEffects>(entity).unwrap();
        assert_eq!(status.effects.len(), 3);
    }

    #[test]
    fn expiry_removes_effect_and_emits_event() {
        let mut world = setup();
        let entity = world.spawn(Health::new(100.0));

        apply_effect(&mut world, entity, make_effect("stun", 1.0));

        // Tick past the duration.
        status_effect_tick_system(&mut world, 1.5);

        let status = world.get::<StatusEffects>(entity).unwrap();
        assert!(status.effects.is_empty());

        // Check that an expired event was emitted.
        let events = world.resource::<Events>().unwrap();
        let expired: Vec<_> = events.read::<StatusEffectExpired>().collect();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].tag, "stun");
    }

    #[test]
    fn tick_dps_sends_damage_event() {
        let mut world = setup();
        let entity = world.spawn(Health::new(100.0));

        let source = world.spawn(Health::new(100.0));
        let mut effect = make_effect("poison_debuff", 5.0);
        effect.tick_effect = Some(TickEffect::DamagePerSecond(10.0));
        effect.source = Some(source);

        apply_effect(&mut world, entity, effect);
        status_effect_tick_system(&mut world, 1.0);

        let events = world.resource::<Events>().unwrap();
        let damage: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(damage.len(), 1);
        assert_eq!(damage[0].target.index(), entity.index());
        assert!((damage[0].amount - 10.0).abs() < 0.01);
        assert_eq!(damage[0].source.unwrap().index(), source.index());
    }

    #[test]
    fn tick_hps_heals_entity() {
        let mut world = setup();
        let entity = world.spawn(Health {
            current: 50.0,
            max: 100.0,
        });

        let mut effect = make_effect("regen_buff", 5.0);
        effect.tick_effect = Some(TickEffect::HealPerSecond(20.0));

        apply_effect(&mut world, entity, effect);
        status_effect_tick_system(&mut world, 1.0);

        let health = world.get::<Health>(entity).unwrap();
        assert!((health.current - 70.0).abs() < 0.01);
    }

    #[test]
    fn cleanse_removes_matching_effects() {
        let mut world = setup();
        let entity = world.spawn(Health::new(100.0));

        apply_effect(&mut world, entity, make_effect("poison_debuff", 5.0));
        apply_effect(&mut world, entity, make_effect("slow_debuff", 3.0));
        apply_effect(&mut world, entity, make_effect("speed_buff", 4.0));

        let removed = cleanse(&mut world, entity, "debuff");
        assert_eq!(removed, 2);

        let status = world.get::<StatusEffects>(entity).unwrap();
        assert_eq!(status.effects.len(), 1);
        assert_eq!(status.effects[0].tag, "speed_buff");
    }

    #[test]
    fn cleanse_on_entity_without_effects_returns_zero() {
        let mut world = setup();
        let entity = world.spawn(Health::new(100.0));

        let removed = cleanse(&mut world, entity, "debuff");
        assert_eq!(removed, 0);
    }

    #[test]
    fn source_tracking() {
        let mut world = setup();
        let target = world.spawn(Health::new(100.0));
        let source = world.spawn(Health::new(100.0));

        let mut effect = make_effect("stun", 2.0);
        effect.source = Some(source);

        apply_effect(&mut world, target, effect);

        let status = world.get::<StatusEffects>(target).unwrap();
        assert_eq!(status.effects[0].source.unwrap().index(), source.index());
    }

    #[test]
    fn modifiers_are_stored_correctly() {
        let mut world = setup();
        let entity = world.spawn(Health::new(100.0));

        let mut effect = make_effect("stun", 2.0);
        effect.modifiers = vec![
            StatModifier {
                stat: "can_move".to_string(),
                op: ModifierOp::Set,
                value: 0.0,
            },
            StatModifier {
                stat: "speed".to_string(),
                op: ModifierOp::Multiply,
                value: 0.0,
            },
        ];

        apply_effect(&mut world, entity, effect);

        let status = world.get::<StatusEffects>(entity).unwrap();
        assert_eq!(status.effects[0].modifiers.len(), 2);
        assert_eq!(status.effects[0].modifiers[0].stat, "can_move");
        assert_eq!(status.effects[0].modifiers[0].op, ModifierOp::Set);
        assert_eq!(status.effects[0].modifiers[0].value, 0.0);
    }

    #[test]
    fn different_tags_coexist() {
        let mut world = setup();
        let entity = world.spawn(Health::new(100.0));

        apply_effect(&mut world, entity, make_effect("stun", 2.0));
        apply_effect(&mut world, entity, make_effect("slow", 3.0));

        let status = world.get::<StatusEffects>(entity).unwrap();
        assert_eq!(status.effects.len(), 2);
    }

    #[test]
    fn partial_tick_does_not_expire() {
        let mut world = setup();
        let entity = world.spawn(Health::new(100.0));

        apply_effect(&mut world, entity, make_effect("stun", 2.0));
        status_effect_tick_system(&mut world, 0.5);

        let status = world.get::<StatusEffects>(entity).unwrap();
        assert_eq!(status.effects.len(), 1);
        assert!((status.effects[0].remaining - 1.5).abs() < 0.01);
    }
}
