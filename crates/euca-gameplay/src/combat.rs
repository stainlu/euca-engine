//! Combat — projectiles and auto-PvP melee.
//!
//! Components: `Projectile`, `AutoCombat`.
//! Systems: `projectile_system`, `auto_combat_system`.

use std::collections::HashMap;

use euca_ecs::{Entity, Events, Query, World};
use euca_math::Vec3;
use euca_physics::Velocity;
use euca_scene::{LocalTransform, SpatialIndex};

use crate::health::{DamageEvent, Dead, Health};
use crate::player::PlayerHero;
use crate::teams::Team;
use crate::tower_aggro::TowerAggroOverride;

/// Entity that moves in a direction and damages what it hits.
#[derive(Clone, Debug)]
pub struct Projectile {
    /// Normalized movement direction.
    pub direction: Vec3,
    /// Movement speed in units per second.
    pub speed: f32,
    /// Damage dealt on hit.
    pub damage: f32,
    /// Maximum time alive in seconds before auto-despawn.
    pub lifetime: f32,
    /// Entity that fired this projectile (excluded from hit detection).
    pub owner: Entity,
    /// Time elapsed since spawn.
    pub elapsed: f32,
    /// Collision radius for hit detection.
    pub radius: f32,
    /// Damage category (e.g. "physical", "magical", "pure").
    /// Used to determine [`DamageType`] when the projectile hits.
    pub category: String,
}

impl Projectile {
    /// Create a projectile with the given trajectory, damage, and lifetime.
    /// Defaults to `"physical"` damage category.
    pub fn new(direction: Vec3, speed: f32, damage: f32, lifetime: f32, owner: Entity) -> Self {
        Self {
            direction: direction.normalize(),
            speed,
            damage,
            lifetime,
            owner,
            elapsed: 0.0,
            radius: 0.5,
            category: "physical".to_string(),
        }
    }

    /// Set the damage category (e.g. `"magical"`, `"pure"`).
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }
}

/// Move projectiles, check lifetime, check collision with Health entities.
pub fn projectile_system(world: &mut World, dt: f32) {
    // Move projectiles and collect expired/hit
    let mut to_despawn: Vec<Entity> = Vec::new();
    let mut damage_events: Vec<DamageEvent> = Vec::new();

    // Collect projectile data
    struct ProjSnapshot {
        entity: Entity,
        direction: Vec3,
        speed: f32,
        damage: f32,
        owner: Entity,
        pos: Vec3,
        radius: f32,
        category: String,
    }
    let projectiles: Vec<ProjSnapshot> = {
        let query = Query::<(Entity, &Projectile, &LocalTransform)>::new(world);
        query
            .iter()
            .map(|(e, p, lt)| ProjSnapshot {
                entity: e,
                direction: p.direction,
                speed: p.speed,
                damage: p.damage,
                owner: p.owner,
                pos: lt.0.translation,
                radius: p.radius,
                category: p.category.clone(),
            })
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

    for proj in &projectiles {
        let new_pos = proj.pos + proj.direction * (proj.speed * dt);

        // Update position
        if let Some(lt) = world.get_mut::<LocalTransform>(proj.entity) {
            lt.0.translation = new_pos;
        }

        // Tick lifetime
        if let Some(p) = world.get_mut::<Projectile>(proj.entity) {
            p.elapsed += dt;
            if p.elapsed >= p.lifetime {
                to_despawn.push(proj.entity);
                continue;
            }
        }

        // Simple sphere collision with targets
        for (target_entity, target_pos) in &targets {
            if *target_entity == proj.owner || *target_entity == proj.entity {
                continue;
            }
            let dist = (new_pos - *target_pos).length();
            if dist < proj.radius {
                damage_events.push(DamageEvent::with_category(
                    *target_entity,
                    proj.damage,
                    Some(proj.owner),
                    proj.category.as_str(),
                ));
                to_despawn.push(proj.entity);
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug)]
pub struct AutoCombat {
    /// Damage dealt per attack.
    pub damage: f32,
    /// Maximum distance at which this entity can attack.
    pub range: f32,
    /// Seconds between attacks.
    pub cooldown: f32,
    /// Time elapsed since last attack (attacks when `elapsed >= cooldown`).
    pub elapsed: f32,
    /// Maximum distance to detect and acquire enemies.
    pub detect_range: f32,
    /// Movement speed when chasing a target (units/s).
    pub speed: f32,
    /// Whether the entity chases enemies or stays in place.
    pub attack_style: AttackStyle,
}

impl AutoCombat {
    /// Create a melee combatant with sensible defaults.
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
/// 1. Dead -> zero velocity, skip.
/// 2. Validate CurrentTarget (alive? in detect_range?). Remove if invalid.
/// 3. If no CurrentTarget -> scan for best enemy using role-aware priority.
///    Uses `SpatialIndex` for O(k) queries when available; falls back to O(n).
/// 4. If CurrentTarget exists -> chase or attack.
/// 5. If no target at all -> march in MarchDirection (or stop).
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

    // Build a lookup map for O(1) access to fighter data by entity.
    let fighter_map: HashMap<Entity, (Vec3, u8, bool, EntityRole)> = fighters
        .iter()
        .map(|&(e, pos, team, dead, role)| (e, (pos, team, dead, role)))
        .collect();

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

        // Skip player-controlled heroes — they use PlayerCommandQueue instead.
        if world.get::<PlayerHero>(entity).is_some() {
            continue;
        }

        let combat = match world.get::<AutoCombat>(entity) {
            Some(c) => *c,
            None => continue,
        };

        // --- Step 2: Validate CurrentTarget (O(1) via fighter_map) ---
        let mut current_target: Option<(Entity, Vec3, f32)> = None;
        if let Some(ct) = world.get::<CurrentTarget>(entity) {
            let target_entity = ct.0;
            let valid = fighter_map
                .get(&target_entity)
                .and_then(|&(tpos, tteam, tdead, _)| {
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

        // --- Step 2b: Tower aggro override ---
        // If this entity has a TowerAggroOverride and the override target is
        // alive, not dead, and within attack range, force it as the current
        // target. This takes priority over both the existing target and the
        // normal priority scan.
        if let Some(ovr) = world.get::<TowerAggroOverride>(entity) {
            let ovr_target = ovr.target;
            let valid = fighter_map
                .get(&ovr_target)
                .and_then(|&(tpos, _tteam, tdead, _)| {
                    if tdead {
                        return None;
                    }
                    // Also skip if the override target has a Dead marker.
                    if world.get::<Dead>(ovr_target).is_some() {
                        return None;
                    }
                    let dist = (tpos - pos).length();
                    if dist > combat.range {
                        return None;
                    }
                    Some((ovr_target, tpos, dist))
                });
            if let Some(v) = valid {
                current_target = Some(v);
                target_inserts.push((entity, v.0));
            }
        }

        // --- Step 3: Acquire new target if none ---
        // Use SpatialIndex for O(k) candidate scan when available,
        // falling back to O(n) full-fighter scan otherwise.
        if current_target.is_none() {
            let mut best: Option<(Entity, Vec3, f32)> = None;
            let mut best_priority: u8 = u8::MAX;

            if let Some(spatial) = world.resource::<SpatialIndex>() {
                // O(k): only examine entities within detect_range.
                let nearby = spatial.query_radius(pos, combat.detect_range);
                for other in nearby {
                    if other == entity {
                        continue;
                    }
                    if let Some(&(other_pos, other_team, other_dead, other_role)) =
                        fighter_map.get(&other)
                    {
                        if other_team == team || other_dead {
                            continue;
                        }
                        let dist = (other_pos - pos).length();
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
                }
            } else {
                // Fallback O(n): scan all fighters when no SpatialIndex.
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
                    damage_events.push(DamageEvent::new(target, combat.damage, Some(entity)));
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
    use euca_physics::Velocity;

    // ── Helpers ──

    fn setup_world() -> World {
        let mut world = World::new();
        world.insert_resource(Events::default());
        world
    }

    fn spawn_fighter(world: &mut World, pos: Vec3, team: u8, role: EntityRole) -> Entity {
        let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(e, Health::new(500.0));
        world.insert(e, Team(team));
        world.insert(e, role);
        world.insert(e, AutoCombat::new());
        world.insert(e, Velocity::default());
        e
    }

    /// Spawn a fighter with both LocalTransform and GlobalTransform so the
    /// spatial index can discover it via GlobalTransform while the combat
    /// system reads positions from LocalTransform.
    fn spawn_fighter_with_global(
        world: &mut World,
        pos: Vec3,
        team: u8,
        role: EntityRole,
    ) -> Entity {
        let e = spawn_fighter(world, pos, team, role);
        world.insert(
            e,
            euca_scene::GlobalTransform(Transform::from_translation(pos)),
        );
        e
    }

    // ── Projectile tests ──

    #[test]
    fn projectile_moves_forward() {
        let mut world = setup_world();
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
        let mut world = setup_world();
        let owner = world.spawn(Health::new(100.0));
        let proj = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(
            proj,
            Projectile::new(Vec3::new(1.0, 0.0, 0.0), 10.0, 25.0, 0.5, owner),
        );
        projectile_system(&mut world, 1.0);
        assert!(!world.is_alive(proj));
    }

    #[test]
    fn projectile_damages_on_hit() {
        let mut world = setup_world();
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
        projectile_system(&mut world, 0.01);
        let events = world.resource::<Events>().unwrap();
        let damage_events: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(damage_events.len(), 1);
        assert_eq!(damage_events[0].target.index(), target.index());
        assert_eq!(damage_events[0].amount, 50.0);
    }

    // ── Targeting priority tests ──

    #[test]
    fn hero_targets_hero_over_minion() {
        let mut world = setup_world();
        // Team 1 hero at origin
        let hero = spawn_fighter(&mut world, Vec3::ZERO, 1, EntityRole::Hero);
        // Team 2 minion nearby
        let _minion = spawn_fighter(&mut world, Vec3::new(3.0, 0.0, 0.0), 2, EntityRole::Minion);
        // Team 2 hero slightly further
        let enemy_hero = spawn_fighter(&mut world, Vec3::new(5.0, 0.0, 0.0), 2, EntityRole::Hero);

        auto_combat_system(&mut world, 0.016);

        let target = world
            .get::<CurrentTarget>(hero)
            .expect("should have target");
        assert_eq!(
            target.0.index(),
            enemy_hero.index(),
            "Hero should target enemy hero over minion"
        );
    }

    #[test]
    fn minion_targets_minion_over_hero() {
        let mut world = setup_world();
        // Team 1 minion at origin
        let minion = spawn_fighter(&mut world, Vec3::ZERO, 1, EntityRole::Minion);
        // Team 2 hero nearby
        let _enemy_hero = spawn_fighter(&mut world, Vec3::new(3.0, 0.0, 0.0), 2, EntityRole::Hero);
        // Team 2 minion slightly further
        let enemy_minion =
            spawn_fighter(&mut world, Vec3::new(5.0, 0.0, 0.0), 2, EntityRole::Minion);

        auto_combat_system(&mut world, 0.016);

        let target = world
            .get::<CurrentTarget>(minion)
            .expect("should have target");
        assert_eq!(
            target.0.index(),
            enemy_minion.index(),
            "Minion should target enemy minion over hero"
        );
    }

    #[test]
    fn tower_targets_minion_over_hero() {
        let mut world = setup_world();
        let tower_pos = Vec3::new(0.0, 0.0, 0.0);
        let tower = world.spawn(LocalTransform(Transform::from_translation(tower_pos)));
        world.insert(tower, Health::new(800.0));
        world.insert(tower, Team(1));
        world.insert(tower, EntityRole::Tower);
        world.insert(tower, AutoCombat::stationary(40.0, 5.0, 1.5));
        world.insert(tower, Velocity::default());

        let _enemy_hero = spawn_fighter(&mut world, Vec3::new(3.0, 0.0, 0.0), 2, EntityRole::Hero);
        let enemy_minion =
            spawn_fighter(&mut world, Vec3::new(4.0, 0.0, 0.0), 2, EntityRole::Minion);

        auto_combat_system(&mut world, 0.016);

        let target = world
            .get::<CurrentTarget>(tower)
            .expect("tower should have target");
        assert_eq!(
            target.0.index(),
            enemy_minion.index(),
            "Tower should target minion over hero"
        );
    }

    // ── CurrentTarget persistence tests ──

    #[test]
    fn current_target_persists_across_ticks() {
        let mut world = setup_world();
        let hero = spawn_fighter(&mut world, Vec3::ZERO, 1, EntityRole::Hero);
        let enemy = spawn_fighter(&mut world, Vec3::new(3.0, 0.0, 0.0), 2, EntityRole::Hero);

        // Tick 1: acquire target
        auto_combat_system(&mut world, 0.016);
        let target1 = world.get::<CurrentTarget>(hero).unwrap().0.index();

        // Tick 2: target should persist (enemy still alive and in range)
        auto_combat_system(&mut world, 0.016);
        let target2 = world.get::<CurrentTarget>(hero).unwrap().0.index();

        assert_eq!(target1, target2, "Target should persist across ticks");
        assert_eq!(target1, enemy.index());
    }

    #[test]
    fn current_target_cleared_on_death() {
        let mut world = setup_world();
        let hero = spawn_fighter(&mut world, Vec3::ZERO, 1, EntityRole::Hero);
        let enemy = spawn_fighter(&mut world, Vec3::new(1.0, 0.0, 0.0), 2, EntityRole::Hero);

        // Acquire target
        auto_combat_system(&mut world, 0.016);
        assert!(world.get::<CurrentTarget>(hero).is_some());

        // Kill enemy
        world.get_mut::<Health>(enemy).unwrap().current = 0.0;

        // Target should be cleared
        auto_combat_system(&mut world, 0.016);
        assert!(
            world.get::<CurrentTarget>(hero).is_none(),
            "Target should be cleared when enemy dies"
        );
    }

    #[test]
    fn current_target_cleared_when_out_of_range() {
        let mut world = setup_world();
        let hero = spawn_fighter(&mut world, Vec3::ZERO, 1, EntityRole::Hero);
        let enemy = spawn_fighter(&mut world, Vec3::new(5.0, 0.0, 0.0), 2, EntityRole::Hero);

        // Acquire target (within default detect_range=20)
        auto_combat_system(&mut world, 0.016);
        assert!(world.get::<CurrentTarget>(hero).is_some());

        // Move enemy far away (beyond detect_range)
        world
            .get_mut::<LocalTransform>(enemy)
            .unwrap()
            .0
            .translation = Vec3::new(100.0, 0.0, 0.0);

        // Target should be cleared
        auto_combat_system(&mut world, 0.016);
        assert!(
            world.get::<CurrentTarget>(hero).is_none(),
            "Target should be cleared when enemy leaves detect range"
        );
    }

    // ── MarchDirection tests ──

    #[test]
    fn march_direction_when_no_target() {
        let mut world = setup_world();
        let hero = spawn_fighter(&mut world, Vec3::ZERO, 1, EntityRole::Hero);
        world.insert(hero, MarchDirection(Vec3::new(1.0, 0.0, 0.0)));
        // No enemies — should march

        auto_combat_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(hero).unwrap();
        assert!(
            vel.linear.x > 0.0,
            "Should march in +X direction when no target"
        );
    }

    #[test]
    fn no_march_when_fighting() {
        let mut world = setup_world();
        let hero = spawn_fighter(&mut world, Vec3::ZERO, 1, EntityRole::Hero);
        world.insert(hero, MarchDirection(Vec3::new(1.0, 0.0, 0.0)));
        // Enemy in attack range
        let _enemy = spawn_fighter(&mut world, Vec3::new(1.0, 0.0, 0.0), 2, EntityRole::Hero);

        auto_combat_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(hero).unwrap();
        assert!(
            vel.linear.x.abs() < 0.01,
            "Should stop marching when fighting"
        );
    }

    #[test]
    fn no_march_direction_means_stop() {
        let mut world = setup_world();
        let hero = spawn_fighter(&mut world, Vec3::ZERO, 1, EntityRole::Hero);
        // No MarchDirection, no enemies

        auto_combat_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(hero).unwrap();
        assert!(
            vel.linear.x.abs() < 0.01 && vel.linear.z.abs() < 0.01,
            "Should stop when no target and no march direction"
        );
    }

    // ── Priority function tests ──

    #[test]
    fn hero_priority_order() {
        assert_eq!(EntityRole::Hero.target_priority_for(&EntityRole::Hero), 0);
        assert_eq!(EntityRole::Minion.target_priority_for(&EntityRole::Hero), 1);
        assert_eq!(EntityRole::Tower.target_priority_for(&EntityRole::Hero), 2);
        assert_eq!(
            EntityRole::Structure.target_priority_for(&EntityRole::Hero),
            3
        );
    }

    #[test]
    fn minion_priority_order() {
        assert_eq!(
            EntityRole::Minion.target_priority_for(&EntityRole::Minion),
            0
        );
        assert_eq!(EntityRole::Hero.target_priority_for(&EntityRole::Minion), 1);
        assert_eq!(
            EntityRole::Tower.target_priority_for(&EntityRole::Minion),
            2
        );
        assert_eq!(
            EntityRole::Structure.target_priority_for(&EntityRole::Minion),
            3
        );
    }

    #[test]
    fn tower_priority_order() {
        assert_eq!(
            EntityRole::Minion.target_priority_for(&EntityRole::Tower),
            0
        );
        assert_eq!(EntityRole::Hero.target_priority_for(&EntityRole::Tower), 1);
    }

    // ── SpatialIndex tests ──

    #[test]
    fn spatial_index_finds_target() {
        let mut world = setup_world();
        // Spawn fighters with GlobalTransform so the spatial index indexes them.
        let hero = spawn_fighter_with_global(&mut world, Vec3::ZERO, 1, EntityRole::Hero);
        let enemy =
            spawn_fighter_with_global(&mut world, Vec3::new(5.0, 0.0, 0.0), 2, EntityRole::Hero);
        // Distant enemy beyond detect_range (default 20) -- should NOT be found.
        let _far_enemy =
            spawn_fighter_with_global(&mut world, Vec3::new(100.0, 0.0, 0.0), 2, EntityRole::Hero);

        // Build the spatial index (mirrors what editor does before combat).
        euca_scene::spatial_index_update_system(&mut world);

        auto_combat_system(&mut world, 0.016);

        let target = world
            .get::<CurrentTarget>(hero)
            .expect("hero should acquire a target via spatial index");
        assert_eq!(
            target.0.index(),
            enemy.index(),
            "spatial query should find the nearby enemy, not the distant one"
        );
    }

    #[test]
    fn fallback_without_spatial_index() {
        let mut world = setup_world();
        // No SpatialIndex resource -- system must fall back to O(n) scan.
        assert!(
            world.resource::<SpatialIndex>().is_none(),
            "precondition: no SpatialIndex resource"
        );

        let hero = spawn_fighter(&mut world, Vec3::ZERO, 1, EntityRole::Hero);
        let enemy = spawn_fighter(&mut world, Vec3::new(5.0, 0.0, 0.0), 2, EntityRole::Hero);

        auto_combat_system(&mut world, 0.016);

        let target = world
            .get::<CurrentTarget>(hero)
            .expect("hero should acquire a target via fallback scan");
        assert_eq!(
            target.0.index(),
            enemy.index(),
            "fallback scan should find the nearby enemy"
        );
    }

    // ── PlayerHero skip tests ──

    #[test]
    fn player_hero_skipped_by_auto_combat() {
        let mut world = setup_world();

        // Spawn a player-controlled hero on team 1.
        let player = spawn_fighter(&mut world, Vec3::ZERO, 1, EntityRole::Hero);
        world.insert(player, PlayerHero);

        // Spawn an enemy within detect range so auto_combat would normally acquire it.
        let _enemy = spawn_fighter(&mut world, Vec3::new(3.0, 0.0, 0.0), 2, EntityRole::Hero);

        // Run several ticks of auto combat.
        for _ in 0..5 {
            auto_combat_system(&mut world, 0.016);
        }

        // The player hero must NOT have a target assigned by auto_combat.
        assert!(
            world.get::<CurrentTarget>(player).is_none(),
            "PlayerHero should not acquire targets via auto_combat_system"
        );

        // The player hero's velocity must remain at zero (no chase/march).
        let vel = world.get::<Velocity>(player).unwrap();
        assert!(
            vel.linear.x.abs() < 0.001 && vel.linear.z.abs() < 0.001,
            "PlayerHero velocity should not be modified by auto_combat_system"
        );
    }
}
