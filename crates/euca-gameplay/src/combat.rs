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
    /// Lower = higher targeting priority (towers prefer minions).
    pub fn target_priority(&self) -> u8 {
        match self {
            Self::Minion => 0, // attacked first
            Self::Hero => 1,   // attacked second
            Self::Tower => 2,
            Self::Structure => 3, // attacked last
        }
    }
}

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
pub fn auto_combat_system(world: &mut World, dt: f32) {
    // Collect all combat entities: position, team, alive
    let fighters: Vec<(Entity, Vec3, u8, bool)> = {
        let query = Query::<(Entity, &LocalTransform, &Team, &Health)>::new(world);
        query
            .iter()
            .filter(|(e, _, _, _)| world.get::<AutoCombat>(*e).is_some())
            .map(|(e, lt, team, health)| (e, lt.0.translation, team.0, health.is_dead()))
            .collect()
    };

    // For each alive fighter, find nearest enemy and decide: chase or attack
    let mut velocity_updates: Vec<(Entity, Vec3)> = Vec::new();
    let mut damage_events: Vec<DamageEvent> = Vec::new();
    let mut cooldown_resets: Vec<Entity> = Vec::new();

    for &(entity, pos, team, dead) in &fighters {
        if dead {
            continue;
        }

        let combat = match world.get::<AutoCombat>(entity) {
            Some(c) => c.clone(),
            None => continue,
        };

        // Find best target: sort by (role_priority, distance)
        let mut nearest: Option<(Entity, Vec3, f32)> = None;
        let mut nearest_priority: u8 = u8::MAX;
        for &(other, other_pos, other_team, other_dead) in &fighters {
            if other == entity || other_team == team || other_dead {
                continue;
            }
            let dist = (other_pos - pos).length();
            if dist >= combat.detect_range {
                continue;
            }
            let priority = world
                .get::<EntityRole>(other)
                .map(|r| r.target_priority())
                .unwrap_or(1); // default: hero-level priority
            let better = match nearest {
                None => true,
                Some((_, _, best_dist)) => {
                    priority < nearest_priority
                        || (priority == nearest_priority && dist < best_dist)
                }
            };
            if better {
                nearest = Some((other, other_pos, dist));
                nearest_priority = priority;
            }
        }

        if let Some((target, _target_pos, dist)) = nearest {
            if dist <= combat.range {
                // In attack range
                let dir = (_target_pos - pos).normalize();

                if combat.elapsed >= combat.cooldown {
                    // Attack: lunge toward target
                    damage_events.push(DamageEvent {
                        target,
                        amount: combat.damage,
                        source: Some(entity),
                    });
                    cooldown_resets.push(entity);
                    velocity_updates.push((
                        entity,
                        Vec3::new(dir.x * combat.speed * 1.5, 0.0, dir.z * combat.speed * 1.5),
                    ));
                } else if dist < combat.range * 0.5 {
                    // Too close — back away to maintain spacing
                    velocity_updates.push((
                        entity,
                        Vec3::new(-dir.x * combat.speed, 0.0, -dir.z * combat.speed),
                    ));
                } else {
                    // In range, waiting for cooldown — hold
                    velocity_updates.push((entity, Vec3::ZERO));
                }
            } else if combat.attack_style == AttackStyle::Stationary {
                // Stationary: enemy in detect range but not attack range — do nothing
                velocity_updates.push((entity, Vec3::ZERO));
            } else {
                // Chase: move toward target
                let dir = (_target_pos - pos).normalize();
                velocity_updates.push((
                    entity,
                    Vec3::new(dir.x * combat.speed, 0.0, dir.z * combat.speed),
                ));
            }
        } else {
            // No enemy found — check if entity has patrol waypoints to follow
            let patrol_vel = world.get::<crate::ai::AiGoal>(entity).and_then(|goal| {
                if matches!(goal.behavior, crate::ai::AiBehavior::Patrol)
                    && !goal.waypoints.is_empty()
                {
                    let wp = goal.waypoints[goal.waypoint_index % goal.waypoints.len()];
                    let to_wp = wp - pos;
                    let dist = Vec3::new(to_wp.x, 0.0, to_wp.z).length();
                    if dist > 0.5 {
                        let dir = Vec3::new(to_wp.x, 0.0, to_wp.z).normalize();
                        Some(Vec3::new(dir.x * combat.speed, 0.0, dir.z * combat.speed))
                    } else {
                        None // at waypoint, advance index below
                    }
                } else {
                    None
                }
            });

            if let Some(vel) = patrol_vel {
                velocity_updates.push((entity, vel));
            } else {
                // Advance patrol waypoint if close enough
                if let Some(goal) = world.get_mut::<crate::ai::AiGoal>(entity)
                    && matches!(goal.behavior, crate::ai::AiBehavior::Patrol)
                    && !goal.waypoints.is_empty()
                {
                    goal.waypoint_index = (goal.waypoint_index + 1) % goal.waypoints.len();
                }
                velocity_updates.push((entity, Vec3::ZERO));
            }
        }
    }

    // Apply velocity updates + ground snapping
    for (entity, vel) in velocity_updates {
        if let Some(v) = world.get_mut::<Velocity>(entity) {
            v.linear.x = vel.x;
            v.linear.z = vel.z;
        }
        // Snap gameplay entities to ground level (Y ≈ half their height)
        if let Some(lt) = world.get_mut::<LocalTransform>(entity)
            && lt.0.translation.y > 0.5
        {
            lt.0.translation.y = 0.5;
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
    for &(entity, _, _, dead) in &fighters {
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
