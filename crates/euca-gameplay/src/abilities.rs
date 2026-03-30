//! Ability system — cooldown-based skills with composable effects.
//!
//! Components: `AbilitySet`, `Mana`, `AppliedEffect`.
//! Events: `UseAbilityEvent`.
//! Systems: `ability_tick_system`, `use_ability_system`.

use euca_ecs::{Entity, Events, Query, World};
use euca_math::{Transform, Vec3};
use euca_scene::LocalTransform;
use serde::{Deserialize, Serialize};

use crate::combat::Projectile;
use crate::crowd_control::{CcState, CcType, CrowdControl, DispelType};
use crate::health::DamageEvent;
use crate::teams::Team;

/// What an ability does when activated.
///
/// Effects are composable: `AreaEffect` applies an inner effect to all entities
/// in radius, and `Chain` sequences multiple effects. This allows abilities like
/// "dash forward, then deal area damage" to be expressed declaratively.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AbilityEffect {
    // ── Legacy effects (backward compatible) ──
    /// Deal damage to all enemies within radius of caster.
    AreaDamage { radius: f32, damage: f32 },
    /// Heal the caster.
    Heal { amount: f32 },
    /// Temporary speed boost.
    SpeedBoost { multiplier: f32, duration: f32 },

    // ── Composable primitives ──
    /// Deal damage with an arbitrary damage category.
    Damage { amount: f32, category: String },
    /// Spawn a projectile entity that travels forward from the caster.
    SpawnProjectile {
        speed: f32,
        range: f32,
        width: f32,
        damage: f32,
        category: String,
    },
    /// Displace the caster in their facing direction.
    Dash { distance: f32 },
    /// Apply a status effect as a marker component.
    /// Each modifier tuple is `(stat_name, op_name, value)` where op_name
    /// is one of `"set"`, `"add"`, or `"multiply"`.
    ApplyEffect {
        tag: String,
        modifiers: Vec<(String, String, f64)>,
        duration: f32,
    },
    /// Apply a crowd control effect to the target entity.
    ///
    /// When used inside an `AreaEffect`, the CC is applied to each affected entity.
    /// When used standalone, it targets the caster (combine with `AreaEffect` or
    /// `Chain` with a projectile for targeted CC).
    ApplyCc {
        cc_type: CcType,
        duration: f32,
        dispel: DispelType,
    },
    /// Apply an inner effect to all entities within radius (recursive).
    AreaEffect {
        radius: f32,
        effect: Box<AbilityEffect>,
    },
    /// Apply multiple effects in sequence.
    Chain(Vec<AbilityEffect>),
}

// ── Ability behavior & targeting ─────────────────────────────────────────────

/// How an ability is activated.
///
/// Models the full spectrum of MOBA ability activation patterns:
/// active skills, passive auras, toggles, channels, and auto-cast modifiers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AbilityBehavior {
    /// Click to cast (default). Has cooldown.
    Active,
    /// Always on, no activation needed. Effects apply permanently.
    Passive,
    /// Toggle on/off. May drain mana per second while active.
    Toggle { mana_per_second: f32 },
    /// Must hold to cast. Interrupted by stun/silence/moving.
    Channeled { channel_duration: f32 },
    /// Can be toggled to auto-cast (like Drow Frost Arrows).
    AutoCast,
}

impl Default for AbilityBehavior {
    fn default() -> Self {
        Self::Active
    }
}

/// What the ability targets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TargetType {
    /// No target needed (self-cast, aura, toggle).
    NoTarget,
    /// Must target a unit (ally, enemy, or self).
    UnitTarget,
    /// Must target a point on the ground.
    PointTarget,
    /// Two-point directional targeting (like Mirana arrow direction).
    VectorTarget,
    /// Targets an area around the caster (no click needed).
    Aura { radius: f32 },
}

impl Default for TargetType {
    fn default() -> Self {
        Self::NoTarget
    }
}

/// Damage classification for abilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DamageType {
    /// Reduced by armor.
    Physical,
    /// Reduced by magic resistance.
    Magical,
    /// Ignores all resistances.
    Pure,
}

/// Cast timing for abilities that aren't instant.
///
/// Models the Dota 2 cast animation system: `cast_point` is the wind-up
/// before the effect fires, and `backswing` is the recovery animation
/// that can be cancelled by issuing a new command.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CastTime {
    /// Time from cast start to when the effect fires (seconds).
    pub cast_point: f32,
    /// Recovery time after effect fires before next action (seconds).
    pub backswing: f32,
}

/// Tracks ability leveling (1-4 for normals, 1-3 for ults).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityLevel {
    /// Current rank of this ability (0 = not yet learned).
    pub current_level: u32,
    /// Maximum rank this ability can reach.
    pub max_level: u32,
    /// Hero levels at which this ability can be leveled up.
    /// e.g., `[1, 3, 5, 7]` for normal abilities, `[6, 12, 18]` for ults.
    pub level_requirements: Vec<u32>,
}

/// Per-level scaling values for an ability.
///
/// Each `Vec` is indexed by `(ability_level - 1)`. A 4-level ability has
/// 4 entries; a 3-level ult has 3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityScaling {
    /// Damage per level: e.g. `[75.0, 150.0, 225.0, 300.0]`.
    pub damage: Vec<f32>,
    /// Mana cost per level: e.g. `[90.0, 100.0, 110.0, 120.0]`.
    pub mana_cost: Vec<f32>,
    /// Cooldown per level: e.g. `[12.0, 10.0, 8.0, 6.0]`.
    pub cooldown: Vec<f32>,
    /// Duration per level (if applicable): e.g. `[2.0, 3.0, 4.0, 5.0]`.
    pub duration: Vec<f32>,
    /// Cast range per level: e.g. `[600.0, 700.0, 800.0, 900.0]`.
    pub cast_range: Vec<f32>,
}

/// State for channeled abilities currently being channeled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelState {
    /// Seconds remaining in the channel.
    pub remaining: f32,
    /// Total channel duration (for progress calculations).
    pub total: f32,
    /// Whether this channel can be interrupted by stun/silence.
    pub can_be_interrupted: bool,
}

// ── Ability component ───────────────────────────────────────────────────────

/// A single ability with cooldown, mana cost, behavior, targeting, and leveling.
#[derive(Clone, Debug)]
pub struct Ability {
    /// Display name (e.g. "Fireball").
    pub name: String,
    /// Total cooldown duration in seconds.
    pub cooldown: f32,
    /// Seconds remaining until the ability is ready again.
    pub cooldown_remaining: f32,
    /// Mana consumed on activation.
    pub mana_cost: f32,
    /// What happens when the ability fires.
    pub effect: AbilityEffect,
    /// How this ability is activated (active, passive, toggle, channel, auto-cast).
    pub behavior: AbilityBehavior,
    /// What the ability targets (no-target, unit, point, vector, aura).
    pub target_type: TargetType,
    /// Cast animation timing. `None` means instant cast.
    pub cast_time: Option<CastTime>,
    /// Ability leveling state. `None` means the ability has no levels.
    pub level: Option<AbilityLevel>,
    /// Per-level stat scaling. `None` means fixed values.
    pub scaling: Option<AbilityScaling>,
    /// Damage classification for resistance calculations.
    pub damage_type: Option<DamageType>,
    /// Whether the ability is currently toggled on (for `Toggle` and `AutoCast`).
    pub is_toggled_on: bool,
    /// Active channel state. `Some` while the ability is being channeled.
    pub channel_state: Option<ChannelState>,
}

impl Default for Ability {
    fn default() -> Self {
        Self {
            name: String::new(),
            cooldown: 0.0,
            cooldown_remaining: 0.0,
            mana_cost: 0.0,
            effect: AbilityEffect::Heal { amount: 0.0 },
            behavior: AbilityBehavior::Active,
            target_type: TargetType::NoTarget,
            cast_time: None,
            level: None,
            scaling: None,
            damage_type: None,
            is_toggled_on: false,
            channel_state: None,
        }
    }
}

impl Ability {
    /// Returns `true` when the cooldown has expired and the ability can be used.
    pub fn is_ready(&self) -> bool {
        self.cooldown_remaining <= 0.0
    }
}

// ── Ability leveling & scaling ──────────────────────────────────────────────

/// Check if an ability can be leveled up at the hero's current level.
///
/// Returns `true` when:
/// - The ability has leveling data.
/// - It hasn't reached `max_level`.
/// - The hero's level meets the requirement for the next rank.
pub fn can_level_ability(ability: &Ability, hero_level: u32) -> bool {
    let level_data = match &ability.level {
        Some(l) => l,
        None => return false,
    };
    if level_data.current_level >= level_data.max_level {
        return false;
    }
    let next_rank = level_data.current_level as usize;
    match level_data.level_requirements.get(next_rank) {
        Some(&req) => hero_level >= req,
        None => false,
    }
}

/// Level up an ability, updating its stats from scaling values.
///
/// Increments `current_level` and applies the corresponding values from
/// `scaling` (cooldown, mana_cost) if present.
pub fn level_up_ability(ability: &mut Ability) -> Result<(), &'static str> {
    let level_data = match &mut ability.level {
        Some(l) => l,
        None => return Err("ability has no leveling data"),
    };
    if level_data.current_level >= level_data.max_level {
        return Err("ability is already at max level");
    }
    level_data.current_level += 1;
    let new_level = level_data.current_level;

    // Apply scaling values for the new level.
    if let Some(scaling) = &ability.scaling {
        ability.cooldown = scaled_value(&scaling.cooldown, new_level);
        ability.mana_cost = scaled_value(&scaling.mana_cost, new_level);
    }

    Ok(())
}

/// Get the current effective value from a scaling table based on ability level.
///
/// Returns the value at index `level - 1`. If the table is empty or the level
/// exceeds the table length, returns the last entry (or 0.0 for empty tables).
pub fn scaled_value(scaling: &[f32], level: u32) -> f32 {
    if scaling.is_empty() {
        return 0.0;
    }
    let idx = (level as usize).saturating_sub(1).min(scaling.len() - 1);
    scaling[idx]
}

/// Toggle an ability on/off.
///
/// Only works for `Toggle` and `AutoCast` behaviors. Returns the new
/// toggled state on success.
pub fn toggle_ability(ability: &mut Ability) -> Result<bool, &'static str> {
    match ability.behavior {
        AbilityBehavior::Toggle { .. } | AbilityBehavior::AutoCast => {
            ability.is_toggled_on = !ability.is_toggled_on;
            Ok(ability.is_toggled_on)
        }
        _ => Err("ability is not a toggle or auto-cast type"),
    }
}

/// Start channeling an ability. Sets the channel state.
///
/// Only works for `Channeled` behaviors. Fails if already channeling.
pub fn start_channel(ability: &mut Ability) -> Result<(), &'static str> {
    match ability.behavior {
        AbilityBehavior::Channeled { channel_duration } => {
            if ability.channel_state.is_some() {
                return Err("already channeling");
            }
            ability.channel_state = Some(ChannelState {
                remaining: channel_duration,
                total: channel_duration,
                can_be_interrupted: true,
            });
            Ok(())
        }
        _ => Err("ability is not a channeled type"),
    }
}

/// Interrupt a channeling ability (e.g., from stun).
///
/// Clears the channel state. No-op if not currently channeling.
pub fn interrupt_channel(ability: &mut Ability) {
    ability.channel_state = None;
}

/// Tick channel progress. Returns `true` when the channel completes.
///
/// When the channel finishes (remaining reaches 0), the channel state
/// is cleared automatically.
pub fn tick_channel(ability: &mut Ability, dt: f32) -> bool {
    let completed = match &mut ability.channel_state {
        Some(state) => {
            state.remaining -= dt;
            state.remaining <= 0.0
        }
        None => false,
    };
    if completed {
        ability.channel_state = None;
    }
    completed
}

/// Ability slot identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AbilitySlot {
    Q,
    W,
    E,
    R,
}

/// Holds up to 4 abilities (Q/W/E/R).
#[derive(Clone, Debug, Default)]
pub struct AbilitySet {
    /// Slot-ability pairs. Order is insertion order, not slot order.
    pub abilities: Vec<(AbilitySlot, Ability)>,
}

impl AbilitySet {
    /// Create an empty ability set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind an ability to a slot.
    pub fn add(&mut self, slot: AbilitySlot, ability: Ability) {
        self.abilities.push((slot, ability));
    }

    /// Look up an ability by slot.
    pub fn get(&self, slot: AbilitySlot) -> Option<&Ability> {
        self.abilities
            .iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, a)| a)
    }

    /// Mutably look up an ability by slot (e.g. to reset cooldown).
    pub fn get_mut(&mut self, slot: AbilitySlot) -> Option<&mut Ability> {
        self.abilities
            .iter_mut()
            .find(|(s, _)| *s == slot)
            .map(|(_, a)| a)
    }
}

/// Mana resource for ability casting.
#[derive(Clone, Debug)]
pub struct Mana {
    /// Current mana available.
    pub current: f32,
    /// Maximum mana (caps regeneration).
    pub max: f32,
    /// Mana regenerated per second.
    pub regen: f32,
}

impl Mana {
    /// Create a full mana pool with the given cap and regen rate.
    pub fn new(max: f32, regen: f32) -> Self {
        Self {
            current: max,
            max,
            regen,
        }
    }
}

/// Request to use an ability.
#[derive(Clone, Debug)]
pub struct UseAbilityEvent {
    /// Entity that wants to cast.
    pub entity: Entity,
    /// Which ability slot to activate.
    pub slot: AbilitySlot,
}

/// Active speed boost effect (temporary).
#[derive(Clone, Debug)]
pub struct SpeedBuff {
    /// Speed multiplier applied while active.
    pub multiplier: f32,
    /// Seconds remaining before the buff expires.
    pub remaining: f32,
    /// Speed value to restore when the buff expires.
    pub original_speed: f32,
}

/// A status effect applied to an entity via `AbilityEffect::ApplyEffect`.
///
/// Stored as a standalone component so it works even without a full
/// StatusEffects system. Each modifier is `(stat_name, operation, value)`.
#[derive(Clone, Debug)]
pub struct AppliedEffect {
    /// Identifies the effect (e.g. "burning", "slowed").
    pub tag: String,
    /// Stat modifiers: `(stat_name, operation, value)`.
    /// Operation is one of `"set"`, `"add"`, or `"multiply"`.
    pub modifiers: Vec<(String, String, f64)>,
    /// Seconds remaining before the effect expires.
    pub remaining: f32,
}

/// Tick cooldowns and regenerate mana.
pub fn ability_tick_system(world: &mut World, dt: f32) {
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &AbilitySet)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    for entity in entities {
        // Tick ability cooldowns
        if let Some(set) = world.get_mut::<AbilitySet>(entity) {
            for (_, ability) in &mut set.abilities {
                if ability.cooldown_remaining > 0.0 {
                    ability.cooldown_remaining -= dt;
                }
            }
        }

        // Regen mana
        if let Some(mana) = world.get_mut::<Mana>(entity) {
            mana.current = (mana.current + mana.regen * dt).min(mana.max);
        }

        // Tick speed buffs
        if let Some(buff) = world.get_mut::<SpeedBuff>(entity) {
            buff.remaining -= dt;
            if buff.remaining <= 0.0 {
                // Restore original speed
                let original = buff.original_speed;
                if let Some(combat) = world.get_mut::<crate::combat::AutoCombat>(entity) {
                    combat.speed = original;
                }
                // Remove buff component
                // (Can't remove during iteration — mark for removal)
            }
        }
    }
}

// ── Deferred effect actions ──────────────────────────────────────────────────

/// A deferred mutation produced by resolving an effect. Collected first,
/// then applied to the world after all effects are resolved to avoid
/// borrow conflicts.
enum EffectAction {
    SendDamage(DamageEvent),
    Heal(Entity, f32),
    SpeedBoost(Entity, f32, f32),
    Dash(Entity, Vec3),
    SpawnProjectile {
        owner: Entity,
        position: Vec3,
        direction: Vec3,
        speed: f32,
        lifetime: f32,
        width: f32,
        damage: f32,
    },
    InsertAppliedEffect(Entity, AppliedEffect),
    ApplyCrowdControl {
        target: Entity,
        cc: CrowdControl,
    },
}

/// Context needed to resolve an effect. Passed by value to avoid
/// lifetime entanglements with the world borrow.
struct EffectContext {
    caster: Entity,
    caster_pos: Vec3,
    caster_rotation: euca_math::Quat,
    caster_team: Option<u8>,
}

impl EffectContext {
    /// Compute the caster's facing direction from rotation.
    /// Falls back to +X if the rotation produces a degenerate vector.
    fn facing_direction(&self) -> Vec3 {
        let dir = self.caster_rotation * Vec3::Z;
        if dir.length() > 0.001 {
            dir.normalize()
        } else {
            Vec3::X
        }
    }
}

/// Recursively resolve an `AbilityEffect` into a list of deferred actions.
///
/// This function reads from the world (immutably via queries) but never
/// mutates it. All mutations are deferred as `EffectAction` values.
fn resolve_effect(
    effect: &AbilityEffect,
    ctx: &EffectContext,
    world: &World,
    actions: &mut Vec<EffectAction>,
) {
    match effect {
        AbilityEffect::AreaDamage { radius, damage } => {
            let targets: Vec<Entity> = {
                let query = Query::<(Entity, &LocalTransform, &Team)>::new(world);
                query
                    .iter()
                    .filter(|(e, lt, t)| {
                        *e != ctx.caster
                            && ctx.caster_team.is_some_and(|ct| ct != t.0)
                            && (lt.0.translation - ctx.caster_pos).length() <= *radius
                    })
                    .map(|(e, _, _)| e)
                    .collect()
            };
            for target in targets {
                actions.push(EffectAction::SendDamage(DamageEvent::new(
                    target,
                    *damage,
                    Some(ctx.caster),
                )));
            }
        }

        AbilityEffect::Heal { amount } => {
            actions.push(EffectAction::Heal(ctx.caster, *amount));
        }

        AbilityEffect::SpeedBoost {
            multiplier,
            duration,
        } => {
            actions.push(EffectAction::SpeedBoost(ctx.caster, *multiplier, *duration));
        }

        AbilityEffect::Damage { amount, .. } => {
            // Targets ctx.caster: standalone this is self-damage, but inside
            // AreaEffect the caster is rebound to each affected entity.
            actions.push(EffectAction::SendDamage(DamageEvent::new(
                ctx.caster,
                *amount,
                Some(ctx.caster),
            )));
        }

        AbilityEffect::SpawnProjectile {
            speed,
            range,
            width,
            damage,
            ..
        } => {
            let direction = ctx.facing_direction();
            let lifetime = if *speed > 0.0 { *range / *speed } else { 0.0 };
            actions.push(EffectAction::SpawnProjectile {
                owner: ctx.caster,
                position: ctx.caster_pos,
                direction,
                speed: *speed,
                lifetime,
                width: *width,
                damage: *damage,
            });
        }

        AbilityEffect::Dash { distance } => {
            let displacement = ctx.facing_direction() * *distance;
            actions.push(EffectAction::Dash(ctx.caster, displacement));
        }

        AbilityEffect::ApplyEffect {
            tag,
            modifiers,
            duration,
        } => {
            actions.push(EffectAction::InsertAppliedEffect(
                ctx.caster,
                AppliedEffect {
                    tag: tag.clone(),
                    modifiers: modifiers.clone(),
                    remaining: *duration,
                },
            ));
        }

        AbilityEffect::ApplyCc {
            cc_type,
            duration,
            dispel,
        } => {
            actions.push(EffectAction::ApplyCrowdControl {
                target: ctx.caster,
                cc: CrowdControl {
                    cc_type: cc_type.clone(),
                    duration: *duration,
                    remaining: *duration,
                    source_entity: Some(ctx.caster.index() as u64),
                    dispel_type: *dispel,
                },
            });
        }

        AbilityEffect::AreaEffect { radius, effect } => {
            // Find all entities (except caster) within radius.
            let targets: Vec<(Entity, Vec3, euca_math::Quat)> = {
                let query = Query::<(Entity, &LocalTransform)>::new(world);
                query
                    .iter()
                    .filter(|(e, lt)| {
                        *e != ctx.caster && (lt.0.translation - ctx.caster_pos).length() <= *radius
                    })
                    .map(|(e, lt)| (e, lt.0.translation, lt.0.rotation))
                    .collect()
            };
            // Resolve the inner effect for each target, with the target
            // as the "caster" so that Damage/ApplyEffect etc. apply to it.
            for (target, target_pos, target_rot) in targets {
                let target_ctx = EffectContext {
                    caster: target,
                    caster_pos: target_pos,
                    caster_rotation: target_rot,
                    caster_team: ctx.caster_team,
                };
                resolve_effect(effect, &target_ctx, world, actions);
            }
        }

        AbilityEffect::Chain(effects) => {
            for sub in effects {
                resolve_effect(sub, ctx, world, actions);
            }
        }
    }
}

/// Apply all deferred effect actions to the world.
fn apply_actions(world: &mut World, actions: Vec<EffectAction>) {
    let mut damage_events: Vec<DamageEvent> = Vec::new();

    for action in actions {
        match action {
            EffectAction::SendDamage(dmg) => {
                damage_events.push(dmg);
            }
            EffectAction::Heal(entity, amount) => {
                crate::health::heal(world, entity, amount);
            }
            EffectAction::SpeedBoost(entity, multiplier, duration) => {
                let original_speed = world
                    .get::<crate::combat::AutoCombat>(entity)
                    .map(|c| c.speed)
                    .unwrap_or(3.0);
                if let Some(combat) = world.get_mut::<crate::combat::AutoCombat>(entity) {
                    combat.speed *= multiplier;
                }
                world.insert(
                    entity,
                    SpeedBuff {
                        multiplier,
                        remaining: duration,
                        original_speed,
                    },
                );
            }
            EffectAction::Dash(entity, displacement) => {
                if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
                    lt.0.translation = lt.0.translation + displacement;
                }
            }
            EffectAction::SpawnProjectile {
                owner,
                position,
                direction,
                speed,
                lifetime,
                width,
                damage,
            } => {
                let proj_entity =
                    world.spawn(LocalTransform(Transform::from_translation(position)));
                let mut projectile = Projectile::new(direction, speed, damage, lifetime, owner);
                projectile.radius = width;
                world.insert(proj_entity, projectile);
            }
            EffectAction::InsertAppliedEffect(entity, applied) => {
                world.insert(entity, applied);
            }
            EffectAction::ApplyCrowdControl { target, cc } => {
                if let Some(cc_state) = world.get_mut::<CcState>(target) {
                    cc_state.apply_cc(cc);
                }
                // Entities without CcState are unaffected — no-op.
            }
        }
    }

    // Send all damage events in one batch.
    if let Some(events) = world.resource_mut::<Events>() {
        for dmg in damage_events {
            events.send(dmg);
        }
    }
}

/// Process UseAbilityEvents: validate cooldown/mana, execute effect.
pub fn use_ability_system(world: &mut World) {
    let events: Vec<UseAbilityEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<UseAbilityEvent>().cloned().collect())
        .unwrap_or_default();

    let mut actions: Vec<EffectAction> = Vec::new();

    for event in events {
        // Get ability info
        let ability_info = world.get::<AbilitySet>(event.entity).and_then(|set| {
            set.get(event.slot)
                .map(|a| (a.is_ready(), a.mana_cost, a.effect.clone()))
        });

        let (ready, mana_cost, effect) = match ability_info {
            Some((true, cost, eff)) => (true, cost, eff),
            _ => continue,
        };

        if !ready {
            continue;
        }

        // Check mana
        let has_mana = world
            .get::<Mana>(event.entity)
            .is_none_or(|m| m.current >= mana_cost);
        if !has_mana {
            continue;
        }

        // Deduct mana
        if let Some(mana) = world.get_mut::<Mana>(event.entity) {
            mana.current -= mana_cost;
        }

        // Set cooldown
        if let Some(set) = world.get_mut::<AbilitySet>(event.entity)
            && let Some(ability) = set.get_mut(event.slot)
        {
            ability.cooldown_remaining = ability.cooldown;
        }

        // Build effect context
        let (caster_pos, caster_rotation) = world
            .get::<LocalTransform>(event.entity)
            .map(|lt| (lt.0.translation, lt.0.rotation))
            .unwrap_or((Vec3::ZERO, euca_math::Quat::IDENTITY));
        let caster_team = world.get::<Team>(event.entity).map(|t| t.0);

        let ctx = EffectContext {
            caster: event.entity,
            caster_pos,
            caster_rotation,
            caster_team,
        };

        resolve_effect(&effect, &ctx, world, &mut actions);
    }

    apply_actions(world, actions);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::Health;

    #[test]
    fn ability_cooldown_ticks() {
        let mut world = World::new();
        let mut set = AbilitySet::new();
        set.add(
            AbilitySlot::Q,
            Ability {
                name: "Fireball".into(),
                cooldown: 5.0,
                cooldown_remaining: 5.0,
                mana_cost: 50.0,
                effect: AbilityEffect::AreaDamage {
                    radius: 3.0,
                    damage: 100.0,
                },
                ..Default::default()
            },
        );
        let e = world.spawn(set);
        world.insert(e, Mana::new(200.0, 5.0));

        ability_tick_system(&mut world, 2.0);

        let set = world.get::<AbilitySet>(e).unwrap();
        let q = set.get(AbilitySlot::Q).unwrap();
        assert!((q.cooldown_remaining - 3.0).abs() < 0.01);

        let mana = world.get::<Mana>(e).unwrap();
        assert!((mana.current - 200.0).abs() < 0.01); // 200 + 5*2 = 210, capped at max 200
    }

    #[test]
    fn area_damage_ability() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut set = AbilitySet::new();
        set.add(
            AbilitySlot::Q,
            Ability {
                name: "Nova".into(),
                cooldown: 3.0,
                cooldown_remaining: 0.0,
                mana_cost: 0.0,
                effect: AbilityEffect::AreaDamage {
                    radius: 5.0,
                    damage: 50.0,
                },
                ..Default::default()
            },
        );

        let caster = world.spawn(set);
        world.insert(
            caster,
            LocalTransform(euca_math::Transform::from_translation(Vec3::ZERO)),
        );
        world.insert(caster, Team(1));

        let enemy = world.spawn(Health::new(100.0));
        world.insert(
            enemy,
            LocalTransform(euca_math::Transform::from_translation(Vec3::new(
                3.0, 0.0, 0.0,
            ))),
        );
        world.insert(enemy, Team(2));

        // Use ability
        world
            .resource_mut::<Events>()
            .unwrap()
            .send(UseAbilityEvent {
                entity: caster,
                slot: AbilitySlot::Q,
            });

        use_ability_system(&mut world);

        // Check damage event was emitted
        let events = world.resource::<Events>().unwrap();
        let dmg_events: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(dmg_events.len(), 1);
        assert_eq!(dmg_events[0].amount, 50.0);
    }

    #[test]
    fn heal_ability() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut set = AbilitySet::new();
        set.add(
            AbilitySlot::W,
            Ability {
                name: "Heal".into(),
                cooldown: 8.0,
                cooldown_remaining: 0.0,
                mana_cost: 0.0,
                effect: AbilityEffect::Heal { amount: 100.0 },
                ..Default::default()
            },
        );

        let caster = world.spawn(set);
        world.insert(
            caster,
            Health {
                current: 200.0,
                max: 500.0,
            },
        );

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(UseAbilityEvent {
                entity: caster,
                slot: AbilitySlot::W,
            });

        use_ability_system(&mut world);

        let health = world.get::<Health>(caster).unwrap();
        assert_eq!(health.current, 300.0);
    }

    // ── New composable effect tests ──

    #[test]
    fn damage_effect_single_target() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut set = AbilitySet::new();
        set.add(
            AbilitySlot::Q,
            Ability {
                name: "Smite".into(),
                cooldown: 1.0,
                cooldown_remaining: 0.0,
                mana_cost: 0.0,
                effect: AbilityEffect::Damage {
                    amount: 75.0,
                    category: "magic".into(),
                },
                ..Default::default()
            },
        );

        let caster = world.spawn(set);
        world.insert(caster, Health::new(200.0));
        world.insert(
            caster,
            LocalTransform(Transform::from_translation(Vec3::ZERO)),
        );

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(UseAbilityEvent {
                entity: caster,
                slot: AbilitySlot::Q,
            });

        use_ability_system(&mut world);

        // Standalone Damage targets the caster itself.
        let events = world.resource::<Events>().unwrap();
        let dmg: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(dmg.len(), 1);
        assert_eq!(dmg[0].amount, 75.0);
        assert_eq!(dmg[0].target.index(), caster.index());
    }

    #[test]
    fn spawn_projectile_effect() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut set = AbilitySet::new();
        set.add(
            AbilitySlot::Q,
            Ability {
                name: "Fireball".into(),
                cooldown: 2.0,
                cooldown_remaining: 0.0,
                mana_cost: 0.0,
                effect: AbilityEffect::SpawnProjectile {
                    speed: 20.0,
                    range: 40.0,
                    width: 0.3,
                    damage: 60.0,
                    category: "fire".into(),
                },
                ..Default::default()
            },
        );

        let caster = world.spawn(set);
        world.insert(
            caster,
            LocalTransform(Transform::from_translation(Vec3::new(5.0, 0.0, 5.0))),
        );

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(UseAbilityEvent {
                entity: caster,
                slot: AbilitySlot::Q,
            });

        use_ability_system(&mut world);

        // A projectile entity should have been spawned.
        let projectiles: Vec<(Entity, Vec3, f32, f32)> = {
            let query = Query::<(Entity, &LocalTransform, &Projectile)>::new(&world);
            query
                .iter()
                .map(|(e, lt, p)| (e, lt.0.translation, p.speed, p.damage))
                .collect()
        };
        assert_eq!(projectiles.len(), 1);
        let (_, pos, speed, damage) = &projectiles[0];
        assert!((pos.x - 5.0).abs() < 0.01);
        assert!((pos.z - 5.0).abs() < 0.01);
        assert!((*speed - 20.0).abs() < 0.01);
        assert!((*damage - 60.0).abs() < 0.01);
    }

    #[test]
    fn dash_displacement() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut set = AbilitySet::new();
        set.add(
            AbilitySlot::E,
            Ability {
                name: "Dash".into(),
                cooldown: 5.0,
                cooldown_remaining: 0.0,
                mana_cost: 0.0,
                effect: AbilityEffect::Dash { distance: 10.0 },
                ..Default::default()
            },
        );

        let caster = world.spawn(set);
        // Caster at origin, facing +Z (identity rotation).
        world.insert(
            caster,
            LocalTransform(Transform::from_translation(Vec3::ZERO)),
        );

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(UseAbilityEvent {
                entity: caster,
                slot: AbilitySlot::E,
            });

        use_ability_system(&mut world);

        let pos = world.get::<LocalTransform>(caster).unwrap().0.translation;
        // With identity rotation, forward is +Z, so should move 10 units in Z.
        assert!(
            (pos.z - 10.0).abs() < 0.01,
            "Expected z=10.0, got z={:.3}",
            pos.z
        );
        assert!(pos.x.abs() < 0.01);
    }

    #[test]
    fn area_effect_hits_multiple_entities() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut set = AbilitySet::new();
        set.add(
            AbilitySlot::Q,
            Ability {
                name: "Shockwave".into(),
                cooldown: 3.0,
                cooldown_remaining: 0.0,
                mana_cost: 0.0,
                effect: AbilityEffect::AreaEffect {
                    radius: 10.0,
                    effect: Box::new(AbilityEffect::Damage {
                        amount: 30.0,
                        category: "physical".into(),
                    }),
                },
                ..Default::default()
            },
        );

        let caster = world.spawn(set);
        world.insert(
            caster,
            LocalTransform(Transform::from_translation(Vec3::ZERO)),
        );

        // Spawn 3 targets within radius.
        let t1 = world.spawn(Health::new(100.0));
        world.insert(
            t1,
            LocalTransform(Transform::from_translation(Vec3::new(3.0, 0.0, 0.0))),
        );
        let t2 = world.spawn(Health::new(100.0));
        world.insert(
            t2,
            LocalTransform(Transform::from_translation(Vec3::new(0.0, 0.0, 5.0))),
        );
        let t3 = world.spawn(Health::new(100.0));
        world.insert(
            t3,
            LocalTransform(Transform::from_translation(Vec3::new(7.0, 0.0, 7.0))),
        );

        // Spawn 1 target outside radius.
        let far = world.spawn(Health::new(100.0));
        world.insert(
            far,
            LocalTransform(Transform::from_translation(Vec3::new(50.0, 0.0, 0.0))),
        );

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(UseAbilityEvent {
                entity: caster,
                slot: AbilitySlot::Q,
            });

        use_ability_system(&mut world);

        let events = world.resource::<Events>().unwrap();
        let dmg: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(dmg.len(), 3, "Should hit exactly 3 entities in radius");
        for d in &dmg {
            assert_eq!(d.amount, 30.0);
            assert_ne!(
                d.target.index(),
                far.index(),
                "Should not hit entity outside radius"
            );
        }
    }

    #[test]
    fn chain_of_effects() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut set = AbilitySet::new();
        set.add(
            AbilitySlot::Q,
            Ability {
                name: "DashStrike".into(),
                cooldown: 6.0,
                cooldown_remaining: 0.0,
                mana_cost: 0.0,
                effect: AbilityEffect::Chain(vec![
                    AbilityEffect::Dash { distance: 5.0 },
                    AbilityEffect::Heal { amount: 20.0 },
                ]),
                ..Default::default()
            },
        );

        let caster = world.spawn(set);
        world.insert(
            caster,
            LocalTransform(Transform::from_translation(Vec3::ZERO)),
        );
        world.insert(
            caster,
            Health {
                current: 50.0,
                max: 100.0,
            },
        );

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(UseAbilityEvent {
                entity: caster,
                slot: AbilitySlot::Q,
            });

        use_ability_system(&mut world);

        // Dash should have moved the caster.
        let pos = world.get::<LocalTransform>(caster).unwrap().0.translation;
        assert!((pos.z - 5.0).abs() < 0.01, "Chain: dash should move caster");

        // Heal should have restored HP.
        let health = world.get::<Health>(caster).unwrap();
        assert_eq!(health.current, 70.0, "Chain: heal should restore 20 HP");
    }

    #[test]
    fn apply_effect_creates_component() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut set = AbilitySet::new();
        set.add(
            AbilitySlot::Q,
            Ability {
                name: "Ignite".into(),
                cooldown: 4.0,
                cooldown_remaining: 0.0,
                mana_cost: 0.0,
                effect: AbilityEffect::ApplyEffect {
                    tag: "burning".into(),
                    modifiers: vec![("fire_damage".into(), "add".into(), 10.0)],
                    duration: 5.0,
                },
                ..Default::default()
            },
        );

        let caster = world.spawn(set);
        world.insert(
            caster,
            LocalTransform(Transform::from_translation(Vec3::ZERO)),
        );

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(UseAbilityEvent {
                entity: caster,
                slot: AbilitySlot::Q,
            });

        use_ability_system(&mut world);

        let applied = world
            .get::<AppliedEffect>(caster)
            .expect("AppliedEffect should be inserted");
        assert_eq!(applied.tag, "burning");
        assert_eq!(applied.modifiers.len(), 1);
        assert_eq!(applied.modifiers[0].0, "fire_damage");
        assert_eq!(applied.modifiers[0].1, "add");
        assert!((applied.modifiers[0].2 - 10.0).abs() < 0.001);
        assert!((applied.remaining - 5.0).abs() < 0.01);
    }

    #[test]
    fn backward_compat_speed_boost() {
        // Ensure the original SpeedBoost variant still works unchanged.
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut set = AbilitySet::new();
        set.add(
            AbilitySlot::W,
            Ability {
                name: "Sprint".into(),
                cooldown: 10.0,
                cooldown_remaining: 0.0,
                mana_cost: 0.0,
                effect: AbilityEffect::SpeedBoost {
                    multiplier: 2.0,
                    duration: 3.0,
                },
                ..Default::default()
            },
        );

        let caster = world.spawn(set);
        world.insert(caster, crate::combat::AutoCombat::new());

        world
            .resource_mut::<Events>()
            .unwrap()
            .send(UseAbilityEvent {
                entity: caster,
                slot: AbilitySlot::W,
            });

        use_ability_system(&mut world);

        let combat = world.get::<crate::combat::AutoCombat>(caster).unwrap();
        assert!((combat.speed - 6.0).abs() < 0.01, "Speed should be doubled");

        let buff = world
            .get::<SpeedBuff>(caster)
            .expect("SpeedBuff should be applied");
        assert!((buff.multiplier - 2.0).abs() < 0.01);
        assert!((buff.remaining - 3.0).abs() < 0.01);
    }

    // ── Advanced ability mechanic tests ──

    #[test]
    fn test_ability_behavior_types() {
        // Each behavior variant round-trips through serde.
        let behaviors = vec![
            AbilityBehavior::Active,
            AbilityBehavior::Passive,
            AbilityBehavior::Toggle {
                mana_per_second: 5.0,
            },
            AbilityBehavior::Channeled {
                channel_duration: 3.0,
            },
            AbilityBehavior::AutoCast,
        ];
        for b in &behaviors {
            let json = serde_json::to_string(b).unwrap();
            let decoded: AbilityBehavior = serde_json::from_str(&json).unwrap();
            assert_eq!(*b, decoded);
        }
    }

    #[test]
    fn test_can_level_ability() {
        let ability = Ability {
            name: "Storm Bolt".into(),
            level: Some(AbilityLevel {
                current_level: 0,
                max_level: 4,
                level_requirements: vec![1, 3, 5, 7],
            }),
            ..Default::default()
        };
        // Hero level 1 can learn rank 1.
        assert!(can_level_ability(&ability, 1));
        // Hero level 0 cannot.
        assert!(!can_level_ability(&ability, 0));
    }

    #[test]
    fn test_level_up_ability() {
        let mut ability = Ability {
            name: "Storm Bolt".into(),
            level: Some(AbilityLevel {
                current_level: 0,
                max_level: 4,
                level_requirements: vec![1, 3, 5, 7],
            }),
            scaling: Some(AbilityScaling {
                damage: vec![100.0, 175.0, 250.0, 325.0],
                mana_cost: vec![140.0, 150.0, 160.0, 170.0],
                cooldown: vec![13.0, 11.0, 9.0, 7.0],
                duration: vec![1.7, 1.8, 1.9, 2.0],
                cast_range: vec![600.0, 600.0, 600.0, 600.0],
            }),
            ..Default::default()
        };

        level_up_ability(&mut ability).unwrap();
        assert_eq!(ability.level.as_ref().unwrap().current_level, 1);
        assert!((ability.cooldown - 13.0).abs() < 0.01);
        assert!((ability.mana_cost - 140.0).abs() < 0.01);

        level_up_ability(&mut ability).unwrap();
        assert_eq!(ability.level.as_ref().unwrap().current_level, 2);
        assert!((ability.cooldown - 11.0).abs() < 0.01);
        assert!((ability.mana_cost - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_level_up_max() {
        let mut ability = Ability {
            name: "Blink".into(),
            level: Some(AbilityLevel {
                current_level: 4,
                max_level: 4,
                level_requirements: vec![1, 3, 5, 7],
            }),
            ..Default::default()
        };
        let result = level_up_ability(&mut ability);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "ability is already at max level");
    }

    #[test]
    fn test_scaled_value() {
        let table = vec![75.0, 150.0, 225.0, 300.0];
        assert!((scaled_value(&table, 1) - 75.0).abs() < 0.01);
        assert!((scaled_value(&table, 2) - 150.0).abs() < 0.01);
        assert!((scaled_value(&table, 3) - 225.0).abs() < 0.01);
        assert!((scaled_value(&table, 4) - 300.0).abs() < 0.01);
        // Beyond table length clamps to last entry.
        assert!((scaled_value(&table, 5) - 300.0).abs() < 0.01);
        // Level 0 clamps to first entry.
        assert!((scaled_value(&table, 0) - 75.0).abs() < 0.01);
        // Empty table returns 0.
        assert!((scaled_value(&[], 1) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_toggle_on_off() {
        let mut ability = Ability {
            name: "Rot".into(),
            behavior: AbilityBehavior::Toggle {
                mana_per_second: 5.0,
            },
            ..Default::default()
        };
        assert!(!ability.is_toggled_on);

        let result = toggle_ability(&mut ability).unwrap();
        assert!(result);
        assert!(ability.is_toggled_on);

        let result = toggle_ability(&mut ability).unwrap();
        assert!(!result);
        assert!(!ability.is_toggled_on);
    }

    #[test]
    fn test_toggle_non_toggle() {
        let mut ability = Ability {
            name: "Fireball".into(),
            behavior: AbilityBehavior::Active,
            ..Default::default()
        };
        let result = toggle_ability(&mut ability);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "ability is not a toggle or auto-cast type"
        );
    }

    #[test]
    fn test_channel_start() {
        let mut ability = Ability {
            name: "Dismember".into(),
            behavior: AbilityBehavior::Channeled {
                channel_duration: 3.0,
            },
            ..Default::default()
        };
        start_channel(&mut ability).unwrap();
        let state = ability.channel_state.as_ref().unwrap();
        assert!((state.remaining - 3.0).abs() < 0.01);
        assert!((state.total - 3.0).abs() < 0.01);
        assert!(state.can_be_interrupted);
    }

    #[test]
    fn test_channel_tick() {
        let mut ability = Ability {
            name: "Dismember".into(),
            behavior: AbilityBehavior::Channeled {
                channel_duration: 3.0,
            },
            ..Default::default()
        };
        start_channel(&mut ability).unwrap();

        // Tick 1 second — not done yet.
        assert!(!tick_channel(&mut ability, 1.0));
        let state = ability.channel_state.as_ref().unwrap();
        assert!((state.remaining - 2.0).abs() < 0.01);

        // Tick 2 more seconds — completes.
        assert!(tick_channel(&mut ability, 2.0));
        assert!(ability.channel_state.is_none());
    }

    #[test]
    fn test_channel_interrupt() {
        let mut ability = Ability {
            name: "Dismember".into(),
            behavior: AbilityBehavior::Channeled {
                channel_duration: 3.0,
            },
            ..Default::default()
        };
        start_channel(&mut ability).unwrap();
        assert!(ability.channel_state.is_some());

        interrupt_channel(&mut ability);
        assert!(ability.channel_state.is_none());
    }

    #[test]
    fn test_cast_time() {
        let ct = CastTime {
            cast_point: 0.3,
            backswing: 0.5,
        };
        let ability = Ability {
            name: "Storm Bolt".into(),
            cast_time: Some(ct),
            ..Default::default()
        };
        let cast_time = ability.cast_time.unwrap();
        assert!((cast_time.cast_point - 0.3).abs() < 0.01);
        assert!((cast_time.backswing - 0.5).abs() < 0.01);
        // Total animation time.
        let total = cast_time.cast_point + cast_time.backswing;
        assert!((total - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_passive_no_cooldown() {
        // Passive abilities have no cooldown — is_ready is always true.
        let ability = Ability {
            name: "Blur".into(),
            behavior: AbilityBehavior::Passive,
            cooldown: 0.0,
            cooldown_remaining: 0.0,
            ..Default::default()
        };
        assert_eq!(ability.behavior, AbilityBehavior::Passive);
        assert!(ability.is_ready());
        // Even with no mana cost, a passive doesn't need activation.
        assert_eq!(ability.mana_cost, 0.0);
    }

    #[test]
    fn test_autocast_toggle() {
        let mut ability = Ability {
            name: "Frost Arrows".into(),
            behavior: AbilityBehavior::AutoCast,
            ..Default::default()
        };
        assert!(!ability.is_toggled_on);

        // Toggle auto-cast on.
        let on = toggle_ability(&mut ability).unwrap();
        assert!(on);
        assert!(ability.is_toggled_on);

        // Toggle auto-cast off.
        let off = toggle_ability(&mut ability).unwrap();
        assert!(!off);
        assert!(!ability.is_toggled_on);
    }

    #[test]
    fn test_ult_level_requirements() {
        let mut ability = Ability {
            name: "Reaper's Scythe".into(),
            level: Some(AbilityLevel {
                current_level: 0,
                max_level: 3,
                level_requirements: vec![6, 12, 18],
            }),
            scaling: Some(AbilityScaling {
                damage: vec![400.0, 550.0, 700.0],
                mana_cost: vec![200.0, 350.0, 500.0],
                cooldown: vec![120.0, 100.0, 80.0],
                duration: vec![1.5, 1.5, 1.5],
                cast_range: vec![600.0, 600.0, 600.0],
            }),
            ..Default::default()
        };

        // Can't learn ult at hero level 5.
        assert!(!can_level_ability(&ability, 5));
        // Can learn at hero level 6.
        assert!(can_level_ability(&ability, 6));

        level_up_ability(&mut ability).unwrap();
        assert_eq!(ability.level.as_ref().unwrap().current_level, 1);

        // Can't learn rank 2 at hero level 11.
        assert!(!can_level_ability(&ability, 11));
        // Can learn rank 2 at hero level 12.
        assert!(can_level_ability(&ability, 12));

        level_up_ability(&mut ability).unwrap();
        assert_eq!(ability.level.as_ref().unwrap().current_level, 2);

        // Can't learn rank 3 at hero level 17.
        assert!(!can_level_ability(&ability, 17));
        // Can learn rank 3 at hero level 18.
        assert!(can_level_ability(&ability, 18));

        level_up_ability(&mut ability).unwrap();
        assert_eq!(ability.level.as_ref().unwrap().current_level, 3);
        assert!((ability.cooldown - 80.0).abs() < 0.01);
        assert!((ability.mana_cost - 500.0).abs() < 0.01);

        // Cannot level beyond max.
        assert!(!can_level_ability(&ability, 25));
    }

    #[test]
    fn test_target_types() {
        // All target type variants round-trip through serde.
        let targets = vec![
            TargetType::NoTarget,
            TargetType::UnitTarget,
            TargetType::PointTarget,
            TargetType::VectorTarget,
            TargetType::Aura { radius: 900.0 },
        ];
        for t in &targets {
            let json = serde_json::to_string(t).unwrap();
            let decoded: TargetType = serde_json::from_str(&json).unwrap();
            assert_eq!(*t, decoded);
        }
    }

    #[test]
    fn test_damage_type_serde() {
        let types = vec![DamageType::Physical, DamageType::Magical, DamageType::Pure];
        for dt in &types {
            let json = serde_json::to_string(dt).unwrap();
            let decoded: DamageType = serde_json::from_str(&json).unwrap();
            assert_eq!(*dt, decoded);
        }
    }

    #[test]
    fn test_channel_start_non_channeled() {
        let mut ability = Ability {
            name: "Fireball".into(),
            behavior: AbilityBehavior::Active,
            ..Default::default()
        };
        let result = start_channel(&mut ability);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "ability is not a channeled type");
    }

    #[test]
    fn test_channel_double_start() {
        let mut ability = Ability {
            name: "Black Hole".into(),
            behavior: AbilityBehavior::Channeled {
                channel_duration: 4.0,
            },
            ..Default::default()
        };
        start_channel(&mut ability).unwrap();
        let result = start_channel(&mut ability);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "already channeling");
    }

    #[test]
    fn test_level_up_no_leveling_data() {
        let mut ability = Ability {
            name: "Blink Dagger".into(),
            ..Default::default()
        };
        let result = level_up_ability(&mut ability);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "ability has no leveling data");
    }
}
