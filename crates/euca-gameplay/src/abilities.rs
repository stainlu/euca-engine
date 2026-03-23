//! Ability system — cooldown-based skills with effects.
//!
//! Components: `AbilitySet`, `Mana`.
//! Events: `UseAbilityEvent`.
//! Systems: `ability_tick_system`, `use_ability_system`.

use euca_ecs::{Entity, Events, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;
use serde::{Deserialize, Serialize};

use crate::health::DamageEvent;
use crate::teams::Team;

/// What an ability does when activated.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AbilityEffect {
    /// Deal damage to all enemies within radius of caster.
    AreaDamage { radius: f32, damage: f32 },
    /// Heal the caster.
    Heal { amount: f32 },
    /// Temporary speed boost.
    SpeedBoost { multiplier: f32, duration: f32 },
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

/// Process UseAbilityEvents: validate cooldown/mana, execute effect.
pub fn use_ability_system(world: &mut World) {
    let events: Vec<UseAbilityEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<UseAbilityEvent>().cloned().collect())
        .unwrap_or_default();

    let mut damage_events: Vec<DamageEvent> = Vec::new();
    let mut heals: Vec<(Entity, f32)> = Vec::new();
    let mut speed_boosts: Vec<(Entity, f32, f32)> = Vec::new();

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

        // Execute effect
        let caster_pos = world
            .get::<LocalTransform>(event.entity)
            .map(|lt| lt.0.translation)
            .unwrap_or(Vec3::ZERO);
        let caster_team = world.get::<Team>(event.entity).map(|t| t.0);

        match effect {
            AbilityEffect::AreaDamage { radius, damage } => {
                // Find all enemies within radius
                let targets: Vec<Entity> = {
                    let query = Query::<(Entity, &LocalTransform, &Team)>::new(world);
                    query
                        .iter()
                        .filter(|(e, lt, t)| {
                            *e != event.entity
                                && caster_team.is_some_and(|ct| ct != t.0)
                                && (lt.0.translation - caster_pos).length() <= radius
                        })
                        .map(|(e, _, _)| e)
                        .collect()
                };
                for target in targets {
                    damage_events.push(DamageEvent {
                        target,
                        amount: damage,
                        source: Some(event.entity),
                    });
                }
            }
            AbilityEffect::Heal { amount } => {
                heals.push((event.entity, amount));
            }
            AbilityEffect::SpeedBoost {
                multiplier,
                duration,
            } => {
                speed_boosts.push((event.entity, multiplier, duration));
            }
        }
    }

    // Apply damage events
    if let Some(events) = world.resource_mut::<Events>() {
        for dmg in damage_events {
            events.send(dmg);
        }
    }

    // Apply heals
    for (entity, amount) in heals {
        crate::health::heal(world, entity, amount);
    }

    // Apply speed boosts
    for (entity, multiplier, duration) in speed_boosts {
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
}
