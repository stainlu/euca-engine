//! Player hero — marker, command queue, and input-driven command execution.
//!
//! Components: `PlayerHero`, `PlayerCommandQueue`.
//! Systems: `player_command_system`.

use euca_ecs::{Entity, Events, Query, World};
use euca_math::Vec3;
use euca_physics::Velocity;
use euca_scene::LocalTransform;

use crate::abilities::AbilitySlot;
use crate::combat::AutoCombat;
use crate::health::{DamageEvent, Health};

/// Marker component that identifies the player's hero entity.
/// Entities with this marker are driven by `PlayerCommandQueue` rather than AI.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerHero;

/// What a player command targets (for abilities).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AbilityTarget {
    SelfCast,
    Point(Vec3),
    Unit(Entity),
    None,
}

/// A single player-issued command.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlayerCommand {
    MoveTo(Vec3),
    AttackTarget(Entity),
    UseAbility {
        slot: AbilitySlot,
        target: AbilityTarget,
    },
    Stop,
    HoldPosition,
}

/// Queued commands for a player hero. The system processes `current` first,
/// then pops the next command from `commands` when `current` completes.
#[derive(Clone, Debug)]
pub struct PlayerCommandQueue {
    pub commands: Vec<PlayerCommand>,
    pub current: Option<PlayerCommand>,
}

impl PlayerCommandQueue {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            current: None,
        }
    }

    pub fn push(&mut self, command: PlayerCommand) {
        self.commands.push(command);
    }

    pub fn drain(&mut self) -> Vec<PlayerCommand> {
        std::mem::take(&mut self.commands)
    }

    pub fn is_empty(&self) -> bool {
        self.current.is_none() && self.commands.is_empty()
    }
}

impl Default for PlayerCommandQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Threshold distance for considering a MoveTo command complete.
const ARRIVAL_THRESHOLD: f32 = 0.5;

/// Process player commands for all entities with `PlayerHero` + `PlayerCommandQueue`.
///
/// Each tick:
/// 1. If `current` is `None` and `commands` is not empty, pop the next command.
/// 2. Execute the current command:
///    - **MoveTo**: set velocity toward target; clear when within threshold.
///    - **AttackTarget**: if target alive and in attack range, deal damage; if not in range, move
///      toward target; if target dead, clear.
///    - **Stop**: zero velocity, clear immediately.
///    - **HoldPosition**: zero velocity, auto-attack enemies in range (stationary behavior).
///    - **UseAbility**: emit `UseAbilityEvent` and clear (ability system handles execution).
pub fn player_command_system(world: &mut World, dt: f32) {
    // Collect entities that have both PlayerHero and PlayerCommandQueue.
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &PlayerHero, &PlayerCommandQueue)>::new(world);
        query.iter().map(|(e, _, _)| e).collect()
    };

    let mut velocity_updates: Vec<(Entity, Vec3)> = Vec::new();
    let mut damage_events: Vec<DamageEvent> = Vec::new();
    let mut cooldown_resets: Vec<Entity> = Vec::new();
    let mut ability_events: Vec<crate::abilities::UseAbilityEvent> = Vec::new();
    let mut command_clears: Vec<Entity> = Vec::new();

    for entity in &entities {
        let entity = *entity;

        // Step 1: If no current command, pop the next one from the queue.
        let queue = match world.get_mut::<PlayerCommandQueue>(entity) {
            Some(q) => q,
            None => continue,
        };
        if queue.current.is_none() && !queue.commands.is_empty() {
            queue.current = Some(queue.commands.remove(0));
        }
        let current = match queue.current {
            Some(cmd) => cmd,
            None => continue,
        };

        // Read this entity's position and combat stats.
        let my_pos = world
            .get::<LocalTransform>(entity)
            .map(|lt| lt.0.translation)
            .unwrap_or(Vec3::ZERO);

        let combat = world.get::<AutoCombat>(entity).copied();

        // Step 2: Execute the current command.
        match current {
            PlayerCommand::MoveTo(target) => {
                // Use XZ-only distance — hero Y may differ from ground target Y.
                let diff_xz = Vec3::new(target.x - my_pos.x, 0.0, target.z - my_pos.z);
                let dist = diff_xz.length();
                if dist <= ARRIVAL_THRESHOLD {
                    // Arrived — stop and clear command.
                    velocity_updates.push((entity, Vec3::ZERO));
                    command_clears.push(entity);
                } else {
                    // Move toward target using combat speed (or a default).
                    let speed = combat.map(|c| c.speed).unwrap_or(3.0);
                    let dir = diff_xz.normalize();
                    velocity_updates.push((entity, Vec3::new(dir.x * speed, 0.0, dir.z * speed)));
                }
            }

            PlayerCommand::AttackTarget(target_entity) => {
                // Check target is alive.
                let target_alive = world
                    .get::<Health>(target_entity)
                    .is_some_and(|h| !h.is_dead());

                if !target_alive {
                    // Target dead — clear command, stop moving.
                    velocity_updates.push((entity, Vec3::ZERO));
                    command_clears.push(entity);
                    continue;
                }

                let target_pos = world
                    .get::<LocalTransform>(target_entity)
                    .map(|lt| lt.0.translation)
                    .unwrap_or(Vec3::ZERO);
                // XZ-only distance for range checks.
                let diff_xz = Vec3::new(target_pos.x - my_pos.x, 0.0, target_pos.z - my_pos.z);
                let dist = diff_xz.length();

                let ac = match combat {
                    Some(c) => c,
                    None => continue,
                };

                if dist <= ac.range {
                    // In attack range — deal damage if cooldown ready.
                    velocity_updates.push((entity, Vec3::ZERO));
                    if ac.elapsed >= ac.cooldown {
                        damage_events.push(DamageEvent {
                            target: target_entity,
                            amount: ac.damage,
                            source: Some(entity),
                        });
                        cooldown_resets.push(entity);
                    }
                } else {
                    // Move toward target on XZ plane.
                    let dir = diff_xz.normalize();
                    velocity_updates
                        .push((entity, Vec3::new(dir.x * ac.speed, 0.0, dir.z * ac.speed)));
                }
            }

            PlayerCommand::UseAbility { slot, .. } => {
                ability_events.push(crate::abilities::UseAbilityEvent { entity, slot });
                command_clears.push(entity);
            }

            PlayerCommand::Stop => {
                velocity_updates.push((entity, Vec3::ZERO));
                command_clears.push(entity);
            }

            PlayerCommand::HoldPosition => {
                // Zero velocity — don't move.
                velocity_updates.push((entity, Vec3::ZERO));

                // Auto-attack nearest enemy in attack range (stationary behavior).
                if let Some(ac) = combat
                    && ac.elapsed >= ac.cooldown
                {
                    let my_team = world.get::<crate::teams::Team>(entity).map(|t| t.0);
                    let target =
                        find_nearest_enemy_in_range(world, entity, my_pos, ac.range, my_team);
                    if let Some(target_entity) = target {
                        damage_events.push(DamageEvent {
                            target: target_entity,
                            amount: ac.damage,
                            source: Some(entity),
                        });
                        cooldown_resets.push(entity);
                    }
                }
                // HoldPosition persists — never auto-clears.
            }
        }
    }

    // Apply velocity updates.
    for (entity, vel) in velocity_updates {
        if let Some(v) = world.get_mut::<Velocity>(entity) {
            v.linear = vel;
        }
    }

    // Emit damage events.
    if let Some(events) = world.resource_mut::<Events>() {
        for event in damage_events {
            events.send(event);
        }
    }

    // Emit ability events.
    if let Some(events) = world.resource_mut::<Events>() {
        for event in ability_events {
            events.send(event);
        }
    }

    // Reset cooldowns for entities that attacked.
    for entity in cooldown_resets {
        if let Some(ac) = world.get_mut::<AutoCombat>(entity) {
            ac.elapsed = 0.0;
        }
    }

    // Tick cooldowns for all player heroes.
    for entity in &entities {
        if let Some(ac) = world.get_mut::<AutoCombat>(*entity) {
            ac.elapsed += dt;
        }
    }

    // Clear completed commands.
    for entity in command_clears {
        if let Some(queue) = world.get_mut::<PlayerCommandQueue>(entity) {
            queue.current = None;
        }
    }
}

/// Find the nearest alive enemy within `range` of `pos`.
fn find_nearest_enemy_in_range(
    world: &World,
    self_entity: Entity,
    pos: Vec3,
    range: f32,
    my_team: Option<u8>,
) -> Option<Entity> {
    let query = Query::<(Entity, &LocalTransform, &crate::teams::Team, &Health)>::new(world);
    let mut best: Option<(Entity, f32)> = None;

    for (e, lt, team, health) in query.iter() {
        if e == self_entity || health.is_dead() {
            continue;
        }
        if my_team.is_some_and(|t| t == team.0) {
            continue;
        }
        let dist = (lt.0.translation - pos).length();
        if dist > range {
            continue;
        }
        let closer = best.is_none_or(|(_, best_dist)| dist < best_dist);
        if closer {
            best = Some((e, dist));
        }
    }

    best.map(|(e, _)| e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combat::AutoCombat;
    use crate::health::Health;
    use crate::teams::Team;
    use euca_math::Transform;

    fn setup_world() -> World {
        let mut world = World::new();
        world.insert_resource(Events::default());
        world
    }

    /// Spawn a player hero at `pos` with standard combat stats.
    fn spawn_player_hero(world: &mut World, pos: Vec3) -> Entity {
        let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(e, PlayerHero);
        world.insert(e, PlayerCommandQueue::new());
        world.insert(e, Health::new(500.0));
        world.insert(e, Team(1));
        world.insert(e, Velocity::default());
        let mut ac = AutoCombat::new();
        // Pre-charge cooldown so attacks fire immediately in tests.
        ac.elapsed = ac.cooldown;
        world.insert(e, ac);
        e
    }

    fn spawn_enemy(world: &mut World, pos: Vec3) -> Entity {
        let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(e, Health::new(100.0));
        world.insert(e, Team(2));
        world.insert(e, Velocity::default());
        e
    }

    // ── Test 1: MoveTo reaches target and stops ──

    #[test]
    fn move_to_reaches_target() {
        let mut world = setup_world();
        let hero = spawn_player_hero(&mut world, Vec3::ZERO);

        let target = Vec3::new(3.0, 0.0, 0.0);
        world.get_mut::<PlayerCommandQueue>(hero).unwrap().current =
            Some(PlayerCommand::MoveTo(target));

        // Tick 1: should be moving toward target.
        player_command_system(&mut world, 0.016);
        let vel = world.get::<Velocity>(hero).unwrap();
        assert!(vel.linear.x > 0.0, "Should be moving toward target");

        // Simulate arriving by teleporting close to target.
        world.get_mut::<LocalTransform>(hero).unwrap().0.translation = Vec3::new(2.8, 0.0, 0.0);

        player_command_system(&mut world, 0.016);
        let vel = world.get::<Velocity>(hero).unwrap();
        assert!(
            vel.linear.x.abs() < 0.01,
            "Should stop when within arrival threshold"
        );
        let queue = world.get::<PlayerCommandQueue>(hero).unwrap();
        assert!(
            queue.current.is_none(),
            "Command should be cleared on arrival"
        );
    }

    // ── Test 2: AttackTarget deals damage ──

    #[test]
    fn attack_target_damages() {
        let mut world = setup_world();
        let hero = spawn_player_hero(&mut world, Vec3::ZERO);
        // Place enemy within attack range (AutoCombat default range = 1.5).
        let enemy = spawn_enemy(&mut world, Vec3::new(1.0, 0.0, 0.0));

        world.get_mut::<PlayerCommandQueue>(hero).unwrap().current =
            Some(PlayerCommand::AttackTarget(enemy));

        player_command_system(&mut world, 0.016);

        let events = world.resource::<Events>().unwrap();
        let dmg: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(dmg.len(), 1, "Should emit exactly one damage event");
        assert_eq!(dmg[0].target.index(), enemy.index());
        assert_eq!(dmg[0].amount, 10.0); // AutoCombat default damage
    }

    // ── Test 3: Stop zeroes velocity ──

    #[test]
    fn stop_zeros_velocity() {
        let mut world = setup_world();
        let hero = spawn_player_hero(&mut world, Vec3::ZERO);

        // Give hero some velocity first.
        world.get_mut::<Velocity>(hero).unwrap().linear = Vec3::new(5.0, 0.0, 3.0);

        world.get_mut::<PlayerCommandQueue>(hero).unwrap().current = Some(PlayerCommand::Stop);

        player_command_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(hero).unwrap();
        assert!(
            vel.linear.x.abs() < 0.01 && vel.linear.z.abs() < 0.01,
            "Stop should zero velocity"
        );
        let queue = world.get::<PlayerCommandQueue>(hero).unwrap();
        assert!(queue.current.is_none(), "Stop should clear immediately");
    }

    // ── Test 4: Command queue ordering ──

    #[test]
    fn command_queue_ordering() {
        let mut world = setup_world();
        let hero = spawn_player_hero(&mut world, Vec3::ZERO);

        // Queue: MoveTo(far away), then Stop.
        {
            let queue = world.get_mut::<PlayerCommandQueue>(hero).unwrap();
            queue
                .commands
                .push(PlayerCommand::MoveTo(Vec3::new(100.0, 0.0, 0.0)));
            queue.commands.push(PlayerCommand::Stop);
        }

        // Tick 1: should pop MoveTo as current.
        player_command_system(&mut world, 0.016);
        let queue = world.get::<PlayerCommandQueue>(hero).unwrap();
        assert!(
            matches!(queue.current, Some(PlayerCommand::MoveTo(_))),
            "First command should be MoveTo"
        );
        assert_eq!(queue.commands.len(), 1, "Stop should still be queued");

        let vel = world.get::<Velocity>(hero).unwrap();
        assert!(vel.linear.x > 0.0, "Should be moving from MoveTo command");

        // Simulate arrival to clear MoveTo.
        world.get_mut::<LocalTransform>(hero).unwrap().0.translation = Vec3::new(100.0, 0.0, 0.0);
        player_command_system(&mut world, 0.016);

        // MoveTo cleared. Next tick should pop Stop.
        player_command_system(&mut world, 0.016);
        let queue = world.get::<PlayerCommandQueue>(hero).unwrap();
        // Stop executes and clears immediately, so current should be None.
        assert!(
            queue.current.is_none(),
            "Stop should have executed and cleared"
        );
        assert!(queue.commands.is_empty(), "Queue should be empty");
    }

    // ── Test 5: HoldPosition auto-attacks in range ──

    #[test]
    fn hold_position_attacks_in_range() {
        let mut world = setup_world();
        let hero = spawn_player_hero(&mut world, Vec3::ZERO);
        let enemy = spawn_enemy(&mut world, Vec3::new(1.0, 0.0, 0.0));

        world.get_mut::<PlayerCommandQueue>(hero).unwrap().current =
            Some(PlayerCommand::HoldPosition);

        player_command_system(&mut world, 0.016);

        // Should not move.
        let vel = world.get::<Velocity>(hero).unwrap();
        assert!(
            vel.linear.x.abs() < 0.01 && vel.linear.z.abs() < 0.01,
            "HoldPosition should not move"
        );

        // Should have attacked the nearby enemy.
        let events = world.resource::<Events>().unwrap();
        let dmg: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(dmg.len(), 1, "Should attack enemy in range");
        assert_eq!(dmg[0].target.index(), enemy.index());

        // HoldPosition should persist (not cleared).
        let queue = world.get::<PlayerCommandQueue>(hero).unwrap();
        assert!(
            matches!(queue.current, Some(PlayerCommand::HoldPosition)),
            "HoldPosition should persist"
        );
    }

    // ── Test 6: AttackTarget clears when target dies ──

    #[test]
    fn attack_target_clears_on_dead_target() {
        let mut world = setup_world();
        let hero = spawn_player_hero(&mut world, Vec3::ZERO);
        let enemy = spawn_enemy(&mut world, Vec3::new(1.0, 0.0, 0.0));

        // Kill the enemy before issuing command.
        world.get_mut::<Health>(enemy).unwrap().current = 0.0;

        world.get_mut::<PlayerCommandQueue>(hero).unwrap().current =
            Some(PlayerCommand::AttackTarget(enemy));

        player_command_system(&mut world, 0.016);

        let queue = world.get::<PlayerCommandQueue>(hero).unwrap();
        assert!(
            queue.current.is_none(),
            "AttackTarget should clear when target is dead"
        );
    }

    // ── Test 7: AttackTarget chases out-of-range enemy ──

    #[test]
    fn attack_target_chases_out_of_range() {
        let mut world = setup_world();
        let hero = spawn_player_hero(&mut world, Vec3::ZERO);
        // Place enemy beyond attack range (default 1.5) but alive.
        let enemy = spawn_enemy(&mut world, Vec3::new(10.0, 0.0, 0.0));

        world.get_mut::<PlayerCommandQueue>(hero).unwrap().current =
            Some(PlayerCommand::AttackTarget(enemy));

        player_command_system(&mut world, 0.016);

        let vel = world.get::<Velocity>(hero).unwrap();
        assert!(vel.linear.x > 0.0, "Should chase toward enemy");

        // Command should still be active (not cleared).
        let queue = world.get::<PlayerCommandQueue>(hero).unwrap();
        assert!(
            matches!(queue.current, Some(PlayerCommand::AttackTarget(_))),
            "AttackTarget should persist while chasing"
        );
    }
}
