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
    /// Apply an inner effect to all entities within radius (recursive).
    AreaEffect {
        radius: f32,
        effect: Box<AbilityEffect>,
    },
    /// Apply multiple effects in sequence.
    Chain(Vec<AbilityEffect>),
}

/// A single ability with cooldown and mana cost.
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
}

impl Ability {
    /// Returns `true` when the cooldown has expired and the ability can be used.
    pub fn is_ready(&self) -> bool {
        self.cooldown_remaining <= 0.0
    }
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
                actions.push(EffectAction::SendDamage(DamageEvent {
                    target,
                    amount: *damage,
                    source: Some(ctx.caster),
                }));
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
            actions.push(EffectAction::SendDamage(DamageEvent {
                target: ctx.caster,
                amount: *amount,
                source: Some(ctx.caster),
            }));
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
}
