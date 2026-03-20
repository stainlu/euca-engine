//! Combat — projectiles and auto-PvP melee.
//!
//! Components: `Projectile`, `AutoCombat`.
//! Systems: `projectile_system`, `auto_combat_system`.

use euca_ecs::{Entity, Events, Query, World};
use euca_math::Vec3;
use euca_physics::Velocity;
use euca_scene::LocalTransform;

use crate::health::{DamageEvent, Health};
use crate::teams::Team;

/// Entity that moves in a direction and damages what it hits.
#[derive(Clone, Debug)]
pub struct Projectile {
    pub direction: Vec3,
    pub speed: f32,
    pub damage: f32,
    pub lifetime: f32,
    pub owner: Entity,
    pub elapsed: f32,
}

impl Projectile {
    pub fn new(direction: Vec3, speed: f32, damage: f32, lifetime: f32, owner: Entity) -> Self {
        Self {
            direction: direction.normalize(),
            speed,
            damage,
            lifetime,
            owner,
            elapsed: 0.0,
        }
    }
}

/// Move projectiles, check lifetime, check collision with Health entities.
pub fn projectile_system(world: &mut World, dt: f32) {
    // Move projectiles and collect expired/hit
    let mut to_despawn: Vec<Entity> = Vec::new();
    let mut damage_events: Vec<DamageEvent> = Vec::new();

    // Collect projectile data
    let projectiles: Vec<(Entity, Vec3, f32, f32, Entity, Vec3)> = {
        let query = Query::<(Entity, &Projectile, &LocalTransform)>::new(world);
        query
            .iter()
            .map(|(e, p, lt)| (e, p.direction, p.speed, p.damage, p.owner, lt.0.translation))
            .collect()
    };

    // Collect potential targets (entities with Health and a position)
    let targets: Vec<(Entity, Vec3)> = {
        let query = Query::<(Entity, &LocalTransform)>::new(world);
        query
            .iter()
            .filter(|(e, _)| world.get::<crate::health::Health>(*e).is_some())
            .map(|(e, lt)| (e, lt.0.translation))
            .collect()
    };

    for (proj_entity, direction, speed, damage, owner, pos) in &projectiles {
        let new_pos = *pos + *direction * (*speed * dt);

        // Update position
        if let Some(lt) = world.get_mut::<LocalTransform>(*proj_entity) {
            lt.0.translation = new_pos;
        }

        // Tick lifetime
        if let Some(proj) = world.get_mut::<Projectile>(*proj_entity) {
            proj.elapsed += dt;
            if proj.elapsed >= proj.lifetime {
                to_despawn.push(*proj_entity);
                continue;
            }
        }

        // Simple sphere collision with targets (radius 0.5)
        let hit_radius = 0.5;
        for (target_entity, target_pos) in &targets {
            if *target_entity == *owner || *target_entity == *proj_entity {
                continue;
            }
            let dist = (new_pos - *target_pos).length();
            if dist < hit_radius {
                damage_events.push(DamageEvent {
                    target: *target_entity,
                    amount: *damage,
                    source: Some(*owner),
                });
                to_despawn.push(*proj_entity);
                break;
            }
        }
    }

    // Emit damage events
    if let Some(events) = world.resource_mut::<Events>() {
        for event in damage_events {
            events.send(event);
        }
    }

    // Despawn expired/hit projectiles
    for entity in to_despawn {
        world.despawn(entity);
    }
}

/// What kind of entity this is (for targeting priority and bounties).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityRole {
    Minion,
    Hero,
    Tower,
    Structure,
}

impl EntityRole {
    /// Target priority depends on who is attacking.
    /// Lower = higher priority (attacked first).
    pub fn target_priority_for(&self, attacker: &EntityRole) -> u8 {
        match attacker {
            // Heroes: Hero > Minion > Tower > Structure
            EntityRole::Hero => match self {
                EntityRole::Hero => 0,
                EntityRole::Minion => 1,
                EntityRole::Tower => 2,
                EntityRole::Structure => 3,
            },
            // Minions: Minion > Hero > Tower > Structure
            EntityRole::Minion => match self {
                EntityRole::Minion => 0,
                EntityRole::Hero => 1,
                EntityRole::Tower => 2,
                EntityRole::Structure => 3,
            },
            // Towers: Minion > Hero > Structure
            EntityRole::Tower => match self {
                EntityRole::Minion => 0,
                EntityRole::Hero => 1,
                EntityRole::Structure => 2,
                EntityRole::Tower => 3,
            },
            // Structures don't attack
            EntityRole::Structure => 3,
        }
    }
}

/// Tracks the current combat target. Persists across ticks until invalidated.
#[derive(Clone, Debug)]
pub struct CurrentTarget(pub Entity);

/// Default movement direction when not in combat (march toward enemy base).
#[derive(Clone, Debug)]
pub struct MarchDirection(pub Vec3);

// ── Auto-PvP Combat ──

/// How the entity attacks.
#[derive(Clone, Debug, PartialEq)]
pub enum AttackStyle {
    /// Chase enemies and melee attack in range.
    Melee,
    /// Stay in place, only attack enemies that enter range (towers).
    Stationary,
}

impl Default for AttackStyle {
    fn default() -> Self {
        Self::Melee
    }
}

/// Entity automatically detects nearby enemies, chases, and attacks.
/// Just add this + Health + Team to make an entity fight.
#[derive(Clone, Debug)]
pub struct AutoCombat {
    pub damage: f32,
    pub range: f32,
    pub cooldown: f32,
    pub elapsed: f32,
    pub detect_range: f32,
    pub speed: f32,
    pub attack_style: AttackStyle,
}

impl AutoCombat {
    pub fn new() -> Self {
        Self {
            damage: 10.0,
            range: 1.5,
            cooldown: 1.0,
            elapsed: 0.0,
            detect_range: 20.0,
            speed: 3.0,
            attack_style: AttackStyle::Melee,
        }
    }

    /// Create a stationary combatant (tower).
    pub fn stationary(damage: f32, range: f32, cooldown: f32) -> Self {
        Self {
            damage,
            range,
            cooldown,
            elapsed: 0.0,
            detect_range: range, // detect = attack range for towers
            speed: 0.0,
            attack_style: AttackStyle::Stationary,
        }
    }
}

impl Default for AutoCombat {
    fn default() -> Self {
        Self::new()
    }
}

/// Auto-PvP: entities with AutoCombat + Health + Team detect enemies, chase, and attack.
///
/// Behavior per entity each tick:
/// 1. Dead → zero velocity, skip.
/// 2. Validate CurrentTarget (alive? in detect_range?). Remove if invalid.
/// 3. If no CurrentTarget → scan for best enemy using role-aware priority.
/// 4. If CurrentTarget exists → chase or attack.
/// 5. If no target at all → march in MarchDirection (or stop).
pub fn auto_combat_system(world: &mut World, dt: f32) {
    // Collect all combat entities: position, team, alive, role
    let fighters: Vec<(Entity, Vec3, u8, bool, EntityRole)> = {
        let query = Query::<(Entity, &LocalTransform, &Team, &Health)>::new(world);
        query
            .iter()
            .filter(|(e, _, _, _)| world.get::<AutoCombat>(*e).is_some())
            .map(|(e, lt, team, health)| {
                let role = world
                    .get::<EntityRole>(e)
                    .copied()
                    .unwrap_or(EntityRole::Hero);
                (e, lt.0.translation, team.0, health.is_dead(), role)
            })
            .collect()
    };

    let mut velocity_updates: Vec<(Entity, Vec3)> = Vec::new();
    let mut damage_events: Vec<DamageEvent> = Vec::new();
    let mut cooldown_resets: Vec<Entity> = Vec::new();
    let mut target_inserts: Vec<(Entity, Entity)> = Vec::new();
    let mut target_removes: Vec<Entity> = Vec::new();

    for &(entity, pos, team, dead, my_role) in &fighters {
        if dead {
            if let Some(v) = world.get_mut::<Velocity>(entity) {
                v.linear = Vec3::ZERO;
            }
            continue;
        }

        let combat = match world.get::<AutoCombat>(entity) {
            Some(c) => c.clone(),
            None => continue,
        };

        // --- Step 2: Validate CurrentTarget ---
        let mut current_target: Option<(Entity, Vec3, f32)> = None;
        if let Some(ct) = world.get::<CurrentTarget>(entity) {
            let target_entity = ct.0;
            // Check: alive, in detect_range, still an enemy
            let valid = fighters
                .iter()
                .find(|(e, _, _, _, _)| *e == target_entity)
                .and_then(|&(_, tpos, tteam, tdead, _)| {
                    if tdead || tteam == team {
                        return None;
                    }
                    let dist = (tpos - pos).length();
                    if dist >= combat.detect_range {
                        return None;
                    }
                    Some((target_entity, tpos, dist))
                });
            match valid {
                Some(v) => current_target = Some(v),
                None => target_removes.push(entity),
            }
        }

        // --- Step 3: Acquire new target if none ---
        if current_target.is_none() {
            let mut best: Option<(Entity, Vec3, f32)> = None;
            let mut best_priority: u8 = u8::MAX;
            for &(other, other_pos, other_team, other_dead, other_role) in &fighters {
                if other == entity || other_team == team || other_dead {
                    continue;
                }
                let dist = (other_pos - pos).length();
                if dist >= combat.detect_range {
                    continue;
                }
                let priority = other_role.target_priority_for(&my_role);
                let better = match best {
                    None => true,
                    Some((_, _, best_dist)) => {
                        priority < best_priority
                            || (priority == best_priority && dist < best_dist)
                    }
                };
                if better {
                    best = Some((other, other_pos, dist));
                    best_priority = priority;
                }
            }
            if let Some((target_entity, _, _)) = best {
                target_inserts.push((entity, target_entity));
                current_target = best;
            }
        }

        // --- Step 4: Act on target ---
        if let Some((target, target_pos, dist)) = current_target {
            if dist <= combat.range {
                // In attack range — deal damage if cooldown ready
                if combat.elapsed >= combat.cooldown {
                    damage_events.push(DamageEvent {
                        target,
                        amount: combat.damage,
                        source: Some(entity),
                    });
                    cooldown_resets.push(entity);
                }
                velocity_updates.push((entity, Vec3::ZERO));
            } else if combat.attack_style == AttackStyle::Stationary {
                velocity_updates.push((entity, Vec3::ZERO));
            } else {
                // Chase target
                let dir = (target_pos - pos).normalize();
                velocity_updates.push((
                    entity,
                    Vec3::new(dir.x * combat.speed, 0.0, dir.z * combat.speed),
                ));
            }
        } else {
            // --- Step 5: No target — march toward enemy base ---
            if let Some(march) = world.get::<MarchDirection>(entity) {
                let dir = march.0;
                velocity_updates.push((
                    entity,
                    Vec3::new(dir.x * combat.speed, 0.0, dir.z * combat.speed),
                ));
            } else {
                velocity_updates.push((entity, Vec3::ZERO));
            }
        }
    }

    // Apply target changes
    for entity in target_removes {
        world.remove::<CurrentTarget>(entity);
    }
    for (entity, target) in target_inserts {
        world.insert(entity, CurrentTarget(target));
    }

    // Apply velocity updates
    for (entity, vel) in velocity_updates {
        if let Some(v) = world.get_mut::<Velocity>(entity) {
            v.linear.x = vel.x;
            v.linear.z = vel.z;
        }
    }

    // Emit damage events
    if let Some(events) = world.resource_mut::<Events>() {
        for event in damage_events {
            events.send(event);
        }
    }

    // Reset cooldowns for entities that attacked
    for entity in cooldown_resets {
        if let Some(combat) = world.get_mut::<AutoCombat>(entity) {
            combat.elapsed = 0.0;
        }
    }

    // Tick cooldowns for all fighters
    for &(entity, _, _, dead, _) in &fighters {
        if dead {
            continue;
        }
        if let Some(combat) = world.get_mut::<AutoCombat>(entity) {
            combat.elapsed += dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::Health;
    use euca_math::Transform;

    #[test]
    fn projectile_moves_forward() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let owner = world.spawn(Health::new(100.0));

        let proj = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(
            proj,
            Projectile::new(Vec3::new(1.0, 0.0, 0.0), 10.0, 25.0, 5.0, owner),
        );

        projectile_system(&mut world, 1.0);

        let pos = world.get::<LocalTransform>(proj).unwrap().0.translation;
        assert!((pos.x - 10.0).abs() < 0.01);
    }

    #[test]
    fn projectile_despawns_on_lifetime() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let owner = world.spawn(Health::new(100.0));

        let proj = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(
            proj,
            Projectile::new(Vec3::new(1.0, 0.0, 0.0), 10.0, 25.0, 0.5, owner),
        );

        projectile_system(&mut world, 1.0); // elapsed > lifetime

        assert!(!world.is_alive(proj));
    }

    #[test]
    fn projectile_damages_on_hit() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let owner = world.spawn(Health::new(100.0));
        world.insert(
            owner,
            LocalTransform(Transform::from_translation(Vec3::new(-5.0, 0.0, 0.0))),
        );

        let target = world.spawn(Health::new(100.0));
        world.insert(
            target,
            LocalTransform(Transform::from_translation(Vec3::new(0.3, 0.0, 0.0))),
        );

        let proj = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(
            proj,
            Projectile::new(Vec3::new(1.0, 0.0, 0.0), 10.0, 50.0, 5.0, owner),
        );

        // Move projectile close to target
        projectile_system(&mut world, 0.01);

        // Check DamageEvent was emitted
        let events = world.resource::<Events>().unwrap();
        let damage_events: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(damage_events.len(), 1);
        assert_eq!(damage_events[0].target.index(), target.index());
        assert_eq!(damage_events[0].amount, 50.0);
    }
}
