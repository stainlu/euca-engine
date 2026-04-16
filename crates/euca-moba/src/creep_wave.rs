//! Creep wave spawning, lane routing, aggro, denial, and last-hit mechanics.
//!
//! Implements Dota 2-style creep waves: periodic spawning of melee, ranged,
//! siege, and super creeps across three lanes, with configurable aggro
//! priority, denial rules, and gold bounty scaling.
//!
//! Reuses [`euca_gameplay::economy::CreepType`] for creep variants. Adds stats,
//! composition, aggro, denial, and wave spawning on top.
//!
//! Types: `Lane`, `WaveConfig`, `LaneWaypoints`, `CreepAggro`,
//! `WaveSpawner`, `SpawnWaveEvent`, `CreepStats`.
//!
//! Pure functions: `creep_stats`, `wave_composition`, `can_deny`,
//! `denial_xp`, `last_hit_gold`.

use euca_ecs::{Events, World};
use euca_math::Vec3;

use euca_gameplay::economy::CreepType;
use euca_gameplay::rules::RuleSpawnEvent;

/// Map a `CreepType` to the string tag used in procedural mesh names.
fn creep_type_tag(ct: CreepType) -> &'static str {
    match ct {
        CreepType::Melee => "melee",
        CreepType::Ranged => "ranged",
        CreepType::Siege => "siege",
        CreepType::Super => "super",
    }
}

// ── Creep stats ──

/// Base stats for a creep, determined by its type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CreepStats {
    pub hp: f32,
    pub damage: f32,
    pub armor: f32,
}

/// Return the base combat stats for a creep type.
pub fn creep_stats(creep_type: CreepType) -> CreepStats {
    match creep_type {
        CreepType::Melee => CreepStats {
            hp: 550.0,
            damage: 21.0,
            armor: 2.0,
        },
        CreepType::Ranged => CreepStats {
            hp: 300.0,
            damage: 27.0,
            armor: 0.0,
        },
        CreepType::Siege => CreepStats {
            hp: 800.0,
            damage: 40.0,
            armor: 0.0,
        },
        CreepType::Super => CreepStats {
            hp: 1100.0,
            damage: 36.0,
            armor: 0.0,
        },
    }
}

/// Base gold bounty for last-hitting a creep (before time scaling).
fn base_bounty(creep_type: CreepType) -> u32 {
    match creep_type {
        CreepType::Melee => 38,
        CreepType::Ranged => 44,
        CreepType::Siege => 66,
        CreepType::Super => 80,
    }
}

// ── Lane routing ──

/// The three standard MOBA lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lane {
    Top,
    Mid,
    Bot,
}

/// Ordered waypoints that creeps follow along a lane.
#[derive(Clone, Debug)]
pub struct LaneWaypoints {
    pub lane: Lane,
    pub points: Vec<Vec3>,
}

// ── Wave composition ──

/// Configuration for wave spawning.
#[derive(Clone, Debug)]
pub struct WaveConfig {
    /// Seconds between wave spawns.
    pub spawn_interval: f32,
    /// Current wave number (1-indexed, increments each spawn).
    pub wave_number: u32,
    /// Elapsed game time in seconds.
    pub game_time: f32,
}

impl Default for WaveConfig {
    fn default() -> Self {
        Self {
            spawn_interval: 30.0,
            wave_number: 1,
            game_time: 0.0,
        }
    }
}

/// Determine the creep composition for a given wave.
///
/// - Every wave: 3 melee + 1 ranged.
/// - Every 5th wave: adds 1 siege creep.
/// - If barracks destroyed: adds 1 super creep.
pub fn wave_composition(wave_number: u32, barracks_destroyed: bool) -> Vec<CreepType> {
    let mut creeps = vec![
        CreepType::Melee,
        CreepType::Melee,
        CreepType::Melee,
        CreepType::Ranged,
    ];

    if wave_number % 5 == 0 {
        creeps.push(CreepType::Siege);
    }

    if barracks_destroyed {
        creeps.push(CreepType::Super);
    }

    creeps
}

// ── Aggro ──

/// Aggro state for a creep, implementing Dota 2's priority system.
///
/// Priority:
/// 1. Enemy hero attacking an allied hero within `hero_defend_range`.
/// 2. Closest enemy unit.
///
/// Aggro resets after `reset_timeout` seconds of no re-triggers.
#[derive(Clone, Debug)]
pub struct CreepAggro {
    /// The entity currently being targeted due to aggro override.
    pub override_target: Option<u64>,
    /// Time remaining before aggro override resets (seconds).
    pub override_timer: f32,
    /// Range within which a hero-on-hero attack triggers aggro (units).
    pub hero_defend_range: f32,
    /// Duration before aggro override expires (seconds).
    pub reset_timeout: f32,
}

impl Default for CreepAggro {
    fn default() -> Self {
        Self {
            override_target: None,
            override_timer: 0.0,
            hero_defend_range: 500.0,
            reset_timeout: 2.5,
        }
    }
}

impl CreepAggro {
    /// Tick the aggro timer by `dt` seconds. Clears override when expired.
    pub fn update(&mut self, dt: f32) {
        if self.override_target.is_some() {
            self.override_timer -= dt;
            if self.override_timer <= 0.0 {
                self.override_target = None;
                self.override_timer = 0.0;
            }
        }
    }

    /// Set an aggro override (enemy hero attacked an allied hero nearby).
    /// Resets the timer to `reset_timeout`.
    pub fn set_override(&mut self, attacker_id: u64) {
        self.override_target = Some(attacker_id);
        self.override_timer = self.reset_timeout;
    }

    /// Whether the creep currently has an active aggro override.
    pub fn has_override(&self) -> bool {
        self.override_target.is_some()
    }
}

// ── Denial ──

/// Check whether `attacker` is allowed to deny `target`.
///
/// Denial rules (Dota 2):
/// - Attacker and target must be on the same team.
/// - Target must be below 50% HP.
pub fn can_deny(attacker_team: u32, target_team: u32, target_hp_percent: f32) -> bool {
    attacker_team == target_team && target_hp_percent < 50.0
}

/// XP awarded to the enemy team when a creep is denied.
///
/// A denied creep gives 50% of the normal XP bounty to nearby enemies.
pub fn denial_xp(normal_xp: u32) -> u32 {
    normal_xp / 2
}

// ── Last hit gold ──

/// Gold awarded for last-hitting a creep, scaling with game time.
///
/// Formula: `base_bounty + floor(game_time_minutes)` (1 gold per minute).
/// This models Dota 2's gradual creep gold increase over the match.
pub fn last_hit_gold(creep_type: CreepType, game_time_minutes: f32) -> u32 {
    let base = base_bounty(creep_type);
    let time_bonus = game_time_minutes.floor() as u32;
    base + time_bonus
}

// ── Wave spawner ──

/// Per-lane configuration within the spawner.
#[derive(Clone, Debug)]
pub struct LaneConfig {
    pub lane: Lane,
    pub waypoints: LaneWaypoints,
    /// Whether the enemy barracks for this lane has been destroyed.
    pub barracks_destroyed: bool,
    /// Team that owns this lane's creeps (1 = Radiant, 2 = Dire).
    pub team: u8,
    /// Mesh path for creeps spawned in this lane (e.g. "assets/generated/radiant_minion.glb").
    pub mesh: String,
    /// Material color name for creeps (e.g. "cyan", "red").
    pub color: String,
}

/// Event emitted when a wave should be spawned.
#[derive(Clone, Debug)]
pub struct SpawnWaveEvent {
    /// Which lane this wave belongs to.
    pub lane: Lane,
    /// The wave number (1-indexed).
    pub wave_number: u32,
    /// The creep types to spawn in this wave.
    pub composition: Vec<CreepType>,
    /// Team that owns these creeps.
    pub team: u8,
    /// Mesh path for these creeps.
    pub mesh: String,
    /// Material color name.
    pub color: String,
    /// Waypoints for marching.
    pub waypoints: Vec<Vec3>,
}

/// Manages periodic creep wave spawning across all lanes.
///
/// Call `tick(dt)` each frame. When the internal timer exceeds
/// `spawn_interval`, it returns spawn events for every configured lane
/// and advances the wave counter.
#[derive(Clone, Debug)]
pub struct WaveSpawner {
    /// Time accumulated since last spawn (seconds).
    pub timer: f32,
    /// Seconds between wave spawns.
    pub spawn_interval: f32,
    /// Current wave number (incremented after each spawn cycle).
    pub wave_number: u32,
    /// Lane configurations.
    pub lanes: Vec<LaneConfig>,
}

impl WaveSpawner {
    /// Create a spawner for the given lanes with a 30-second interval.
    pub fn new(lanes: Vec<LaneConfig>) -> Self {
        Self {
            timer: 0.0,
            spawn_interval: 30.0,
            wave_number: 0,
            lanes,
        }
    }

    /// Advance the spawner by `dt` seconds. Returns spawn events if a wave
    /// triggers this tick.
    pub fn tick(&mut self, dt: f32) -> Vec<SpawnWaveEvent> {
        self.timer += dt;

        if self.timer < self.spawn_interval {
            return Vec::new();
        }

        // Consume one interval (does not accumulate multiple spawns per tick).
        self.timer -= self.spawn_interval;
        self.wave_number += 1;

        self.lanes
            .iter()
            .map(|lane_cfg| {
                let composition = wave_composition(self.wave_number, lane_cfg.barracks_destroyed);
                SpawnWaveEvent {
                    lane: lane_cfg.lane,
                    wave_number: self.wave_number,
                    composition,
                    team: lane_cfg.team,
                    mesh: lane_cfg.mesh.clone(),
                    color: lane_cfg.color.clone(),
                    waypoints: lane_cfg.waypoints.points.clone(),
                }
            })
            .collect()
    }
}

// ── ECS system ──

/// Tick the `WaveSpawner` resource and create creep entities for each
/// triggered wave. Emits `RuleSpawnEvent` for each creep so the rendering
/// layer can attach mesh/material.
///
/// Creep entities get: Health, Team, AutoCombat, MarchDirection,
/// GoldBounty, EntityRole::Minion, Velocity, PhysicsBody, and transforms.
pub fn wave_spawn_system(world: &mut World, dt: f32) {
    // Take the spawner out of the world to avoid borrow conflicts.
    let mut spawner = match world.remove_resource::<WaveSpawner>() {
        Some(s) => s,
        None => return,
    };

    let game_time_minutes = world
        .resource::<euca_gameplay::game_state::GameState>()
        .map(|gs| gs.elapsed / 60.0)
        .unwrap_or(0.0);

    let wave_events = spawner.tick(dt);

    for event in &wave_events {
        let spawn_pos = event.waypoints.first().copied().unwrap_or(Vec3::ZERO);

        // March direction: from first waypoint toward last.
        let march_dir = if event.waypoints.len() >= 2 {
            let last = event.waypoints.last().unwrap();
            (*last - spawn_pos).normalize()
        } else {
            // Default: team 1 marches +X, team 2 marches -X.
            if event.team == 1 {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::new(-1.0, 0.0, 0.0)
            }
        };

        let creep_scale = Vec3::new(0.4, 0.4, 0.4);
        let z_spacing = 1.0_f32;
        let z_offset_base = -z_spacing * (event.composition.len() as f32 - 1.0) / 2.0;

        for (i, &creep_type) in event.composition.iter().enumerate() {
            let stats = creep_stats(creep_type);
            let bounty = euca_gameplay::economy::creep_bounty(creep_type, game_time_minutes);
            let z_offset = z_offset_base + z_spacing * i as f32;

            let mut transform = euca_math::Transform::from_translation(Vec3::new(
                spawn_pos.x,
                spawn_pos.y,
                spawn_pos.z + z_offset,
            ));
            transform.scale = creep_scale;

            let entity = world.spawn(euca_scene::LocalTransform(transform));
            world.insert(entity, euca_scene::GlobalTransform::default());
            world.insert(entity, euca_gameplay::health::Health::new(stats.hp));
            world.insert(entity, euca_gameplay::teams::Team(event.team));
            world.insert(entity, euca_gameplay::combat::EntityRole::Minion);
            world.insert(entity, euca_gameplay::economy::GoldBounty(bounty as i32));

            let mut combat = euca_gameplay::combat::AutoCombat::new();
            combat.damage = stats.damage;
            combat.speed = 3.0;
            world.insert(entity, combat);

            world.insert(entity, euca_physics::Velocity::default());
            world.insert(
                entity,
                euca_physics::PhysicsBody {
                    body_type: euca_physics::RigidBodyType::Kinematic,
                },
            );
            world.insert(entity, euca_gameplay::combat::MarchDirection(march_dir));

            // Emit RuleSpawnEvent with a per-creep-type procedural mesh name.
            let mesh_name = euca_render::creep_mesh_name(creep_type_tag(creep_type), event.team);
            if let Some(events) = world.resource_mut::<Events>() {
                events.send(RuleSpawnEvent {
                    entity,
                    mesh: mesh_name,
                    color: Some(event.color.clone()),
                    scale: Some([creep_scale.x, creep_scale.y, creep_scale.z]),
                });
            }

            log::debug!(
                "Wave {} spawned {:?} creep (team {}) at ({:.0}, {:.0}, {:.0})",
                event.wave_number,
                creep_type,
                event.team,
                spawn_pos.x,
                spawn_pos.y,
                spawn_pos.z + z_offset
            );
        }

        log::info!(
            "Wave {} spawned {} creeps for {:?} lane (team {})",
            event.wave_number,
            event.composition.len(),
            event.lane,
            event.team
        );
    }

    // Put the spawner back.
    world.insert_resource(spawner);
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──

    fn default_waypoints(lane: Lane) -> LaneWaypoints {
        LaneWaypoints {
            lane,
            points: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(100.0, 0.0, 0.0)],
        }
    }

    fn default_lane_config(lane: Lane) -> LaneConfig {
        LaneConfig {
            lane,
            waypoints: default_waypoints(lane),
            barracks_destroyed: false,
            team: 1,
            mesh: "cube".to_string(),
            color: "cyan".to_string(),
        }
    }

    fn three_lane_spawner() -> WaveSpawner {
        WaveSpawner::new(vec![
            default_lane_config(Lane::Top),
            default_lane_config(Lane::Mid),
            default_lane_config(Lane::Bot),
        ])
    }

    // ── Wave composition tests ──

    #[test]
    fn test_wave_composition_normal() {
        let comp = wave_composition(1, false);
        assert_eq!(comp.len(), 4);
        assert_eq!(
            comp.iter().filter(|c| **c == CreepType::Melee).count(),
            3,
            "normal wave has 3 melee"
        );
        assert_eq!(
            comp.iter().filter(|c| **c == CreepType::Ranged).count(),
            1,
            "normal wave has 1 ranged"
        );
    }

    #[test]
    fn test_wave_composition_siege() {
        // Waves 5, 10, 15, ... include a siege creep.
        for wave in [5, 10, 15] {
            let comp = wave_composition(wave, false);
            assert!(
                comp.contains(&CreepType::Siege),
                "wave {wave} should contain a siege creep"
            );
            assert_eq!(comp.len(), 5, "wave {wave}: 3 melee + 1 ranged + 1 siege");
        }

        // Non-5th waves should NOT have siege.
        let comp = wave_composition(3, false);
        assert!(
            !comp.contains(&CreepType::Siege),
            "wave 3 should not contain a siege creep"
        );
    }

    #[test]
    fn test_wave_composition_super() {
        let comp = wave_composition(1, true);
        assert!(
            comp.contains(&CreepType::Super),
            "barracks destroyed should add super creep"
        );
        assert_eq!(comp.len(), 5, "3 melee + 1 ranged + 1 super");
    }

    // ── Creep stats tests ──

    #[test]
    fn test_creep_stats() {
        let melee = creep_stats(CreepType::Melee);
        assert_eq!(melee.hp, 550.0);
        assert_eq!(melee.damage, 21.0);
        assert_eq!(melee.armor, 2.0);

        let ranged = creep_stats(CreepType::Ranged);
        assert_eq!(ranged.hp, 300.0);
        assert_eq!(ranged.damage, 27.0);
        assert_eq!(ranged.armor, 0.0);

        let siege = creep_stats(CreepType::Siege);
        assert_eq!(siege.hp, 800.0);
        assert_eq!(siege.damage, 40.0);
        assert_eq!(siege.armor, 0.0);

        let super_creep = creep_stats(CreepType::Super);
        assert_eq!(super_creep.hp, 1100.0);
        assert_eq!(super_creep.damage, 36.0);
        assert_eq!(super_creep.armor, 0.0);
    }

    // ── Aggro tests ──

    #[test]
    fn test_aggro_priority_hero() {
        let mut aggro = CreepAggro::default();
        assert!(!aggro.has_override(), "no override initially");

        // Enemy hero (ID 42) attacks allied hero within range.
        aggro.set_override(42);
        assert!(aggro.has_override());
        assert_eq!(aggro.override_target, Some(42));
    }

    #[test]
    fn test_aggro_reset() {
        let mut aggro = CreepAggro::default();
        aggro.set_override(42);
        assert!(aggro.has_override());

        // Tick 2.0s -- not yet expired (timeout is 2.5s).
        aggro.update(2.0);
        assert!(aggro.has_override(), "aggro should persist before timeout");

        // Tick another 0.6s -- total 2.6s, past the 2.5s timeout.
        aggro.update(0.6);
        assert!(!aggro.has_override(), "aggro should reset after timeout");
        assert_eq!(aggro.override_target, None);
    }

    // ── Denial tests ──

    #[test]
    fn test_denial_same_team() {
        // Same team, below 50% HP -- can deny.
        assert!(can_deny(1, 1, 40.0));
        assert!(can_deny(2, 2, 10.0));
        assert!(can_deny(1, 1, 49.9));
    }

    #[test]
    fn test_denial_above_threshold() {
        // Same team but at or above 50% HP -- cannot deny.
        assert!(!can_deny(1, 1, 50.0));
        assert!(!can_deny(1, 1, 75.0));
        assert!(!can_deny(1, 1, 100.0));
    }

    #[test]
    fn test_denial_different_team() {
        // Different team -- cannot deny (that is a normal kill).
        assert!(!can_deny(1, 2, 30.0));
    }

    #[test]
    fn test_denial_xp_penalty() {
        // Denied creep gives 50% XP.
        assert_eq!(denial_xp(100), 50);
        assert_eq!(denial_xp(60), 30);
        // Integer division truncates.
        assert_eq!(denial_xp(1), 0);
    }

    // ── Last hit gold tests ──

    #[test]
    fn test_last_hit_bounty() {
        // At minute 0, should return base bounty.
        assert_eq!(last_hit_gold(CreepType::Melee, 0.0), 38);
        assert_eq!(last_hit_gold(CreepType::Ranged, 0.0), 44);
        assert_eq!(last_hit_gold(CreepType::Siege, 0.0), 66);

        // At minute 10, adds 10 gold.
        assert_eq!(last_hit_gold(CreepType::Melee, 10.0), 48);
        assert_eq!(last_hit_gold(CreepType::Ranged, 10.0), 54);

        // At minute 30.5, adds 30 gold (floor).
        assert_eq!(last_hit_gold(CreepType::Melee, 30.5), 68);
    }

    // ── Spawner tests ──

    #[test]
    fn test_spawner_timing() {
        let mut spawner = three_lane_spawner();

        // 29 seconds -- no spawn yet.
        let events = spawner.tick(29.0);
        assert!(events.is_empty(), "should not spawn before 30s");

        // 1 more second (total 30s) -- should spawn.
        let events = spawner.tick(1.0);
        assert_eq!(events.len(), 3, "should spawn one wave per lane");
        assert_eq!(spawner.wave_number, 1);

        // Next 29 seconds -- no spawn.
        let events = spawner.tick(29.0);
        assert!(events.is_empty());

        // 1 more second -- second wave.
        let events = spawner.tick(1.0);
        assert_eq!(events.len(), 3);
        assert_eq!(spawner.wave_number, 2);
    }

    #[test]
    fn test_three_lanes() {
        let mut spawner = three_lane_spawner();

        let events = spawner.tick(30.0);
        assert_eq!(events.len(), 3);

        let lanes: Vec<Lane> = events.iter().map(|e| e.lane).collect();
        assert!(lanes.contains(&Lane::Top), "Top lane should get a wave");
        assert!(lanes.contains(&Lane::Mid), "Mid lane should get a wave");
        assert!(lanes.contains(&Lane::Bot), "Bot lane should get a wave");

        // Each wave should have the correct composition.
        for event in &events {
            assert_eq!(event.wave_number, 1);
            assert_eq!(event.composition.len(), 4, "wave 1: 3 melee + 1 ranged");
        }
    }

    #[test]
    fn test_spawner_siege_wave() {
        let mut spawner = three_lane_spawner();
        let mut last_events = Vec::new();

        // Advance to wave 5.
        for _ in 0..5 {
            last_events = spawner.tick(30.0);
        }

        assert_eq!(spawner.wave_number, 5);

        for event in &last_events {
            assert!(
                event.composition.contains(&CreepType::Siege),
                "wave 5 should include siege creep"
            );
        }
    }

    #[test]
    fn test_spawner_barracks_destroyed() {
        let mut spawner = WaveSpawner::new(vec![
            LaneConfig {
                lane: Lane::Mid,
                waypoints: default_waypoints(Lane::Mid),
                barracks_destroyed: true,
                team: 1,
                mesh: "cube".to_string(),
                color: "cyan".to_string(),
            },
            default_lane_config(Lane::Top),
        ]);

        let events = spawner.tick(30.0);
        assert_eq!(events.len(), 2);

        let mid_event = events.iter().find(|e| e.lane == Lane::Mid).unwrap();
        assert!(
            mid_event.composition.contains(&CreepType::Super),
            "mid lane with destroyed barracks should spawn super creep"
        );

        let top_event = events.iter().find(|e| e.lane == Lane::Top).unwrap();
        assert!(
            !top_event.composition.contains(&CreepType::Super),
            "top lane without destroyed barracks should not spawn super creep"
        );
    }

    #[test]
    fn test_lane_waypoints() {
        let wp = LaneWaypoints {
            lane: Lane::Bot,
            points: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(50.0, 0.0, 50.0),
                Vec3::new(100.0, 0.0, 100.0),
            ],
        };
        assert_eq!(wp.lane, Lane::Bot);
        assert_eq!(wp.points.len(), 3);
    }
}
