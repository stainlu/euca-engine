//! Roshan — the main boss objective. Drops Aegis of the Immortal.
//!
//! Roshan is the linchpin objective in a MOBA match. He spawns with base stats
//! that scale with game time, is spell-immune (magical damage has no effect),
//! and grows stronger with each death. Killing Roshan drops the Aegis (always),
//! Cheese (2nd+ kill), and a Refresher Shard (3rd+ kill).
//!
//! This module is pure data + logic — no ECS dependency. Game systems integrate
//! by calling these functions and mapping results onto ECS entities.

use serde::{Deserialize, Serialize};

// ── Constants ────────────────────────────────────────────────────────────────

/// Base HP at game minute 0.
const BASE_HP: f32 = 6000.0;
/// HP gained per minute of game time.
const HP_PER_MINUTE: f32 = 115.0;
/// Base armor value.
const BASE_ARMOR: f32 = 20.0;
/// Base attack damage.
const BASE_DAMAGE: f32 = 75.0;

/// HP bonus Roshan gains per previous death.
const BONUS_HP_PER_DEATH: f32 = 500.0;
/// Damage bonus per previous death.
const BONUS_DAMAGE_PER_DEATH: f32 = 10.0;
/// Armor bonus per previous death.
const BONUS_ARMOR_PER_DEATH: f32 = 1.0;

/// Minimum respawn time in seconds (8 minutes).
const RESPAWN_MIN_SECONDS: f32 = 480.0;
/// Maximum respawn time in seconds (11 minutes).
const RESPAWN_MAX_SECONDS: f32 = 660.0;

/// Aegis duration before it expires (5 minutes).
const AEGIS_DURATION: f32 = 300.0;
/// Resurrection delay after Aegis triggers (5 seconds).
const AEGIS_RESURRECT_DELAY: f32 = 5.0;

/// Cheese HP restoration.
const CHEESE_HP_RESTORE: f32 = 2500.0;
/// Cheese mana restoration.
const CHEESE_MANA_RESTORE: f32 = 1500.0;

/// Gold bounty for killing Roshan.
const ROSHAN_GOLD_BOUNTY: u32 = 225;

/// Slam AoE damage.
const SLAM_DAMAGE: f32 = 70.0;
/// Slam radius.
const SLAM_RADIUS: f32 = 3.5;
/// Slam cooldown in seconds.
const SLAM_COOLDOWN: f32 = 10.0;

/// Damage category that Roshan is immune to.
const MAGICAL_CATEGORY: &str = "magical";

// ── Data types ───────────────────────────────────────────────────────────────

/// Roshan's current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Roshan {
    pub hp: f32,
    pub max_hp: f32,
    pub armor: f32,
    pub damage: f32,
    pub spell_immune: bool,
    pub alive: bool,
    /// How many times Roshan has been killed this game.
    pub kill_count: u32,
    /// `None` if alive, `Some(remaining_seconds)` if dead and waiting to respawn.
    pub respawn_timer: Option<f32>,
    /// HP bonus Roshan gains per previous death.
    pub bonus_hp_per_death: f32,
    /// Damage bonus per previous death.
    pub bonus_damage_per_death: f32,
    /// Armor bonus per previous death.
    pub bonus_armor_per_death: f32,
}

/// Roshan's slam ability (AoE damage around him).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoshanSlam {
    pub damage: f32,
    pub radius: f32,
    pub cooldown: f32,
    pub remaining_cooldown: f32,
}

/// Aegis of the Immortal — resurrects the holder on death.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aegis {
    /// Entity ID of the hero carrying the Aegis. `None` if on the ground.
    pub holder: Option<u64>,
    /// Time remaining before the Aegis expires (starts at 300 s).
    pub remaining_duration: f32,
    /// Whether the Aegis has been consumed (holder died and resurrected).
    pub consumed: bool,
}

/// Data returned when the Aegis triggers (holder dies).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AegisResurrection {
    /// Entity that will be resurrected.
    pub entity: u64,
    /// Seconds before the resurrection completes.
    pub delay: f32,
    /// Fraction of max HP and mana restored (1.0 = 100%).
    pub heal_percent: f32,
}

/// Cheese — instant HP + mana restore consumable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cheese {
    pub hp_restore: f32,
    pub mana_restore: f32,
    /// Entity ID of the holder.
    pub holder: Option<u64>,
}

/// Refresher Shard — dropped on 3rd+ Roshan kill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefresherShard {
    pub holder: Option<u64>,
}

/// What drops when Roshan dies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoshanDrops {
    /// Aegis always drops.
    pub aegis: bool,
    /// Cheese drops on 2nd+ kill.
    pub cheese: bool,
    /// Refresher Shard drops on 3rd+ kill.
    pub refresher_shard: bool,
    /// Gold bounty split among the killing team.
    pub gold_bounty: u32,
}

// ── Core functions ───────────────────────────────────────────────────────────

/// Create a fresh Roshan with base stats scaled by current game time.
///
/// HP formula: `6000 + 115 * game_time_minutes`.
pub fn spawn_roshan(game_time_minutes: f32) -> Roshan {
    let hp = BASE_HP + HP_PER_MINUTE * game_time_minutes;
    Roshan {
        hp,
        max_hp: hp,
        armor: BASE_ARMOR,
        damage: BASE_DAMAGE,
        spell_immune: true,
        alive: true,
        kill_count: 0,
        respawn_timer: None,
        bonus_hp_per_death: BONUS_HP_PER_DEATH,
        bonus_damage_per_death: BONUS_DAMAGE_PER_DEATH,
        bonus_armor_per_death: BONUS_ARMOR_PER_DEATH,
    }
}

/// Apply damage to Roshan. Returns `true` if Roshan dies from this hit.
///
/// Spell immunity: damage with category `"magical"` is completely ignored.
/// Physical and true damage pass through normally.
pub fn roshan_takes_damage(roshan: &mut Roshan, damage: f32, damage_category: &str) -> bool {
    if !roshan.alive {
        return false;
    }

    // Spell immunity blocks magical damage entirely.
    if roshan.spell_immune && damage_category == MAGICAL_CATEGORY {
        return false;
    }

    roshan.hp = (roshan.hp - damage).max(0.0);

    if roshan.hp <= 0.0 {
        roshan.alive = false;
        return true;
    }

    false
}

/// Called when Roshan's HP reaches 0. Increments kill count, starts respawn
/// timer, and returns the drops for this kill.
pub fn roshan_dies(roshan: &mut Roshan) -> RoshanDrops {
    roshan.alive = false;
    roshan.kill_count += 1;

    // Respawn timer: random between 8-11 minutes. We pick the midpoint here;
    // the game system should randomize within the range returned by
    // `roshan_respawn_time()`.
    let (min, max) = roshan_respawn_time();
    let midpoint = (min + max) / 2.0;
    roshan.respawn_timer = Some(midpoint);

    RoshanDrops {
        aegis: true,
        cheese: roshan.kill_count >= 2,
        refresher_shard: roshan.kill_count >= 3,
        gold_bounty: ROSHAN_GOLD_BOUNTY,
    }
}

/// Tick Roshan's respawn timer. Returns `true` if Roshan is ready to respawn
/// (timer reached 0).
pub fn tick_roshan(roshan: &mut Roshan, dt: f32) -> bool {
    if let Some(timer) = roshan.respawn_timer.as_mut() {
        *timer -= dt;
        if *timer <= 0.0 {
            roshan.respawn_timer = None;
            return true;
        }
    }
    false
}

/// Respawn Roshan with increased stats based on kill count and current game
/// time.
pub fn respawn_roshan(roshan: &mut Roshan, game_time_minutes: f32) {
    let base_hp = BASE_HP + HP_PER_MINUTE * game_time_minutes;
    let bonus_hp = roshan.bonus_hp_per_death * roshan.kill_count as f32;
    let hp = base_hp + bonus_hp;

    roshan.hp = hp;
    roshan.max_hp = hp;
    roshan.armor = BASE_ARMOR + roshan.bonus_armor_per_death * roshan.kill_count as f32;
    roshan.damage = BASE_DAMAGE + roshan.bonus_damage_per_death * roshan.kill_count as f32;
    roshan.alive = true;
    roshan.spell_immune = true;
    roshan.respawn_timer = None;
}

/// Returns the valid respawn time range `(min_seconds, max_seconds)`.
///
/// Roshan respawns between 8 and 11 minutes after death.
pub fn roshan_respawn_time() -> (f32, f32) {
    (RESPAWN_MIN_SECONDS, RESPAWN_MAX_SECONDS)
}

// ── Aegis ────────────────────────────────────────────────────────────────────

/// Create a fresh Aegis (not yet picked up).
pub fn new_aegis() -> Aegis {
    Aegis {
        holder: None,
        remaining_duration: AEGIS_DURATION,
        consumed: false,
    }
}

/// Assign the Aegis to an entity.
pub fn pick_up_aegis(aegis: &mut Aegis, entity: u64) {
    aegis.holder = Some(entity);
}

/// Called when the Aegis holder dies. If valid, consumes the Aegis and returns
/// resurrection data. Returns `None` if already consumed or no holder.
pub fn aegis_trigger(aegis: &mut Aegis) -> Option<AegisResurrection> {
    if aegis.consumed {
        return None;
    }

    let entity = aegis.holder?;

    aegis.consumed = true;

    Some(AegisResurrection {
        entity,
        delay: AEGIS_RESURRECT_DELAY,
        heal_percent: 1.0,
    })
}

/// Tick the Aegis duration. Returns `true` if the Aegis has expired (5 minutes
/// elapsed since Roshan's death).
pub fn tick_aegis(aegis: &mut Aegis, dt: f32) -> bool {
    if aegis.consumed {
        return false;
    }

    aegis.remaining_duration -= dt;
    aegis.remaining_duration <= 0.0
}

// ── Cheese ───────────────────────────────────────────────────────────────────

/// Create a new Cheese drop.
pub fn new_cheese() -> Cheese {
    Cheese {
        hp_restore: CHEESE_HP_RESTORE,
        mana_restore: CHEESE_MANA_RESTORE,
        holder: None,
    }
}

/// Consume Cheese. Returns `(hp_restored, mana_restored)`.
pub fn use_cheese(cheese: &Cheese) -> (f32, f32) {
    (cheese.hp_restore, cheese.mana_restore)
}

// ── Refresher Shard ──────────────────────────────────────────────────────────

/// Create a new Refresher Shard drop.
pub fn new_refresher_shard() -> RefresherShard {
    RefresherShard { holder: None }
}

// ── Slam ─────────────────────────────────────────────────────────────────────

/// Create Roshan's slam ability with default values.
pub fn new_roshan_slam() -> RoshanSlam {
    RoshanSlam {
        damage: SLAM_DAMAGE,
        radius: SLAM_RADIUS,
        cooldown: SLAM_COOLDOWN,
        remaining_cooldown: 0.0,
    }
}

/// Tick slam cooldown. Returns `true` if the slam is ready to fire.
pub fn roshan_slam_tick(slam: &mut RoshanSlam, dt: f32) -> bool {
    if slam.remaining_cooldown > 0.0 {
        slam.remaining_cooldown = (slam.remaining_cooldown - dt).max(0.0);
    }

    if slam.remaining_cooldown <= 0.0 {
        slam.remaining_cooldown = slam.cooldown;
        return true;
    }

    false
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_roshan_base_stats() {
        let rosh = spawn_roshan(0.0);
        assert_eq!(rosh.hp, 6000.0);
        assert_eq!(rosh.max_hp, 6000.0);
        assert_eq!(rosh.armor, 20.0);
        assert_eq!(rosh.damage, 75.0);
        assert!(rosh.alive);
        assert!(rosh.spell_immune);
        assert_eq!(rosh.kill_count, 0);
        assert!(rosh.respawn_timer.is_none());
    }

    #[test]
    fn test_roshan_hp_scaling() {
        let rosh_10 = spawn_roshan(10.0);
        let rosh_30 = spawn_roshan(30.0);

        // 6000 + 115 * 10 = 7150
        assert_eq!(rosh_10.hp, 6000.0 + 115.0 * 10.0);
        // 6000 + 115 * 30 = 9450
        assert_eq!(rosh_30.hp, 6000.0 + 115.0 * 30.0);
        assert!(rosh_30.hp > rosh_10.hp);
    }

    #[test]
    fn test_roshan_spell_immune() {
        let mut rosh = spawn_roshan(0.0);
        let original_hp = rosh.hp;

        // Magical damage should be completely blocked.
        let died = roshan_takes_damage(&mut rosh, 1000.0, "magical");

        assert!(!died);
        assert_eq!(rosh.hp, original_hp, "magical damage should be blocked");
    }

    #[test]
    fn test_roshan_physical_damage() {
        let mut rosh = spawn_roshan(0.0);

        let died = roshan_takes_damage(&mut rosh, 100.0, "physical");

        assert!(!died);
        assert_eq!(rosh.hp, 5900.0);
    }

    #[test]
    fn test_roshan_death_drops_aegis() {
        let mut rosh = spawn_roshan(0.0);
        rosh.hp = 0.0;
        rosh.alive = false;

        let drops = roshan_dies(&mut rosh);

        assert!(drops.aegis, "first kill always drops Aegis");
        assert!(!drops.cheese, "first kill should not drop Cheese");
        assert!(
            !drops.refresher_shard,
            "first kill should not drop Refresher Shard"
        );
        assert_eq!(drops.gold_bounty, ROSHAN_GOLD_BOUNTY);
        assert_eq!(rosh.kill_count, 1);
    }

    #[test]
    fn test_roshan_second_kill_cheese() {
        let mut rosh = spawn_roshan(0.0);

        // Simulate first kill.
        rosh.alive = false;
        let _ = roshan_dies(&mut rosh);
        respawn_roshan(&mut rosh, 10.0);

        // Second kill.
        rosh.alive = false;
        let drops = roshan_dies(&mut rosh);

        assert!(drops.aegis);
        assert!(drops.cheese, "second kill should drop Cheese");
        assert!(
            !drops.refresher_shard,
            "second kill should not drop Refresher Shard"
        );
        assert_eq!(rosh.kill_count, 2);
    }

    #[test]
    fn test_roshan_third_kill_refresher() {
        let mut rosh = spawn_roshan(0.0);

        // Kill 1.
        rosh.alive = false;
        let _ = roshan_dies(&mut rosh);
        respawn_roshan(&mut rosh, 10.0);

        // Kill 2.
        rosh.alive = false;
        let _ = roshan_dies(&mut rosh);
        respawn_roshan(&mut rosh, 20.0);

        // Kill 3.
        rosh.alive = false;
        let drops = roshan_dies(&mut rosh);

        assert!(drops.aegis);
        assert!(drops.cheese);
        assert!(
            drops.refresher_shard,
            "third kill should drop Refresher Shard"
        );
        assert_eq!(rosh.kill_count, 3);
    }

    #[test]
    fn test_roshan_respawn_timer() {
        let (min, max) = roshan_respawn_time();

        assert_eq!(min, 480.0, "minimum respawn = 8 minutes");
        assert_eq!(max, 660.0, "maximum respawn = 11 minutes");
    }

    #[test]
    fn test_aegis_pickup() {
        let mut aegis = new_aegis();
        assert!(aegis.holder.is_none());

        pick_up_aegis(&mut aegis, 42);

        assert_eq!(aegis.holder, Some(42));
        assert!(!aegis.consumed);
    }

    #[test]
    fn test_aegis_trigger_on_death() {
        let mut aegis = new_aegis();
        pick_up_aegis(&mut aegis, 99);

        let res = aegis_trigger(&mut aegis);

        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res.entity, 99);
        assert_eq!(res.delay, 5.0);
        assert_eq!(res.heal_percent, 1.0);
        assert!(aegis.consumed, "Aegis should be consumed after triggering");

        // Second trigger should return None (already consumed).
        assert!(aegis_trigger(&mut aegis).is_none());
    }

    #[test]
    fn test_aegis_expiry() {
        let mut aegis = new_aegis();
        pick_up_aegis(&mut aegis, 1);

        // Tick 299 seconds — should not expire yet.
        let expired = tick_aegis(&mut aegis, 299.0);
        assert!(!expired);

        // Tick the remaining 1+ second — should expire.
        let expired = tick_aegis(&mut aegis, 2.0);
        assert!(expired, "Aegis should expire after 300 seconds total");
    }

    #[test]
    fn test_cheese_restore() {
        let cheese = new_cheese();

        let (hp, mana) = use_cheese(&cheese);

        assert_eq!(hp, 2500.0);
        assert_eq!(mana, 1500.0);
    }

    #[test]
    fn test_roshan_gets_stronger() {
        let mut rosh = spawn_roshan(0.0);
        let initial_armor = rosh.armor;
        let initial_damage = rosh.damage;

        // Kill and respawn at minute 0 to isolate the per-death bonus.
        rosh.alive = false;
        let _ = roshan_dies(&mut rosh);
        respawn_roshan(&mut rosh, 0.0);

        assert_eq!(
            rosh.max_hp,
            BASE_HP + BONUS_HP_PER_DEATH,
            "HP should increase by bonus_hp_per_death after one death"
        );
        assert_eq!(
            rosh.armor,
            initial_armor + BONUS_ARMOR_PER_DEATH,
            "armor should increase after death"
        );
        assert_eq!(
            rosh.damage,
            initial_damage + BONUS_DAMAGE_PER_DEATH,
            "damage should increase after death"
        );
    }

    #[test]
    fn test_roshan_takes_damage_kills() {
        let mut rosh = spawn_roshan(0.0);

        let died = roshan_takes_damage(&mut rosh, 6000.0, "physical");

        assert!(died, "lethal damage should return true");
        assert!(!rosh.alive);
        assert_eq!(rosh.hp, 0.0);
    }

    #[test]
    fn test_tick_roshan_respawn() {
        let mut rosh = spawn_roshan(0.0);
        rosh.alive = false;
        rosh.respawn_timer = Some(10.0);

        // Not ready yet.
        assert!(!tick_roshan(&mut rosh, 5.0));
        assert!(rosh.respawn_timer.is_some());

        // Ready now.
        assert!(tick_roshan(&mut rosh, 6.0));
        assert!(rosh.respawn_timer.is_none());
    }

    #[test]
    fn test_slam_cooldown_tick() {
        let mut slam = new_roshan_slam();
        assert_eq!(slam.remaining_cooldown, 0.0);

        // First tick: ready immediately, resets cooldown.
        let ready = roshan_slam_tick(&mut slam, 0.016);
        assert!(ready);
        assert_eq!(slam.remaining_cooldown, SLAM_COOLDOWN);

        // Mid-cooldown tick: not ready.
        let ready = roshan_slam_tick(&mut slam, 5.0);
        assert!(!ready);

        // Finish cooldown: ready again.
        let ready = roshan_slam_tick(&mut slam, 6.0);
        assert!(ready);
    }

    #[test]
    fn test_damage_to_dead_roshan_is_noop() {
        let mut rosh = spawn_roshan(0.0);
        rosh.alive = false;
        rosh.hp = 0.0;

        let died = roshan_takes_damage(&mut rosh, 100.0, "physical");

        assert!(!died, "damage to dead Roshan should not return true");
        assert_eq!(rosh.hp, 0.0, "HP should not change");
    }

    #[test]
    fn test_true_damage_bypasses_spell_immunity() {
        let mut rosh = spawn_roshan(0.0);
        let original_hp = rosh.hp;

        let died = roshan_takes_damage(&mut rosh, 500.0, "true");

        assert!(!died);
        assert_eq!(
            rosh.hp,
            original_hp - 500.0,
            "true damage should bypass spell immunity"
        );
    }
}
