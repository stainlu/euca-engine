//! Dota 2 economy system — gold management, bounties, buyback, respawn.
//!
//! Gold is split into **reliable** (hero kills, tower bounties, Roshan) and
//! **unreliable** (creep kills, passive income). Reliable gold is never lost
//! on death. Spending always consumes unreliable gold first.
//!
//! Components: `Gold` (backward-compatible wrapper), `GoldBounty`, `HeroEconomy`.
//! Systems: `gold_on_kill_system`, `tick_passive_income`, `tick_buyback_cooldown`.
//! Free functions: `apply_death_penalty`, `attempt_buyback`, `award_kill`.

use serde::{Deserialize, Serialize};

use euca_ecs::{Events, World};

use crate::health::DeathEvent;

// ── Gold wallet ──

/// Gold split into reliable (hero kills, tower bounty, Roshan)
/// and unreliable (creep kills, passive income).
/// Reliable gold is not lost on death.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct GoldWallet {
    pub reliable: u32,
    pub unreliable: u32,
}

impl GoldWallet {
    pub fn total(&self) -> u32 {
        self.reliable + self.unreliable
    }

    /// Spend gold, consuming unreliable first.
    pub fn spend(&mut self, amount: u32) -> Result<(), EconomyError> {
        if self.total() < amount {
            return Err(EconomyError::InsufficientGold);
        }
        let from_unreliable = amount.min(self.unreliable);
        self.unreliable -= from_unreliable;
        self.reliable -= amount - from_unreliable;
        Ok(())
    }

    /// Add reliable gold (hero kills, towers, Roshan).
    pub fn add_reliable(&mut self, amount: u32) {
        self.reliable += amount;
    }

    /// Add unreliable gold (creep kills, passive income).
    pub fn add_unreliable(&mut self, amount: u32) {
        self.unreliable += amount;
    }
}

// ── Error ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EconomyError {
    InsufficientGold,
    BuybackOnCooldown,
    BuybackNotDead,
    AlreadyAlive,
}

impl std::fmt::Display for EconomyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientGold => write!(f, "insufficient gold"),
            Self::BuybackOnCooldown => write!(f, "buyback on cooldown"),
            Self::BuybackNotDead => write!(f, "cannot buyback while alive"),
            Self::AlreadyAlive => write!(f, "already alive"),
        }
    }
}

impl std::error::Error for EconomyError {}

// ── Constants ──

/// Passive gold income rate (gold per second).
pub const PASSIVE_GOLD_PER_SECOND: f32 = 1.0;

/// Starting gold for heroes.
pub const STARTING_GOLD: u32 = 600;

/// Buyback cooldown in seconds (480 = 8 minutes).
pub const BUYBACK_COOLDOWN: f32 = 480.0;

// ── Backward-compatible Gold component ──

/// How much gold this entity carries.
///
/// This is a backward-compatible wrapper. For Dota 2-depth economy, use
/// `HeroEconomy` instead. `Gold` is still useful for non-hero entities
/// (minions, structures) that have a simple gold pool.
#[derive(Clone, Copy, Debug)]
pub struct Gold(pub i32);

impl Gold {
    pub fn new(amount: i32) -> Self {
        Self(amount)
    }
}

/// How much gold the killer receives when this entity dies.
#[derive(Clone, Copy, Debug)]
pub struct GoldBounty(pub i32);

// ── Creep types and bounty formulas ──

/// Creep bounty based on type and game time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CreepType {
    Melee,
    Ranged,
    Siege,
    Super,
}

/// Creep bounty formula: base + scaling * game_time_minutes.
pub fn creep_bounty(creep_type: CreepType, game_time_minutes: f32) -> u32 {
    let base = match creep_type {
        CreepType::Melee => 38,
        CreepType::Ranged => 46,
        CreepType::Siege => 74,
        CreepType::Super => 36,
    };
    let scaling = match creep_type {
        CreepType::Melee => 0.7,
        CreepType::Ranged => 0.7,
        CreepType::Siege => 1.5,
        CreepType::Super => 0.0,
    };
    (base as f32 + scaling * game_time_minutes) as u32
}

/// Hero kill bounty.
/// Base: 110 + streak * 60 + victim_level * 8
pub fn hero_kill_bounty(victim_level: u32, victim_streak: u32) -> u32 {
    110 + victim_streak * 60 + victim_level * 8
}

/// Assist gold split.
/// Kill gold goes to killer. Assist gold = kill_bounty * 0.3 / num_assists.
pub fn assist_gold(kill_bounty: u32, num_assists: u32) -> u32 {
    if num_assists == 0 {
        return 0;
    }
    (kill_bounty as f32 * 0.3 / num_assists as f32) as u32
}

/// Gold lost on death (unreliable only).
/// Formula: 30 * level.
pub fn gold_loss_on_death(level: u32) -> u32 {
    30 * level
}

// ── Buyback ──

/// Buyback cost formula: 100 + level^2 * 1.5 + game_time_seconds * 0.25.
pub fn buyback_cost(level: u32, game_time_seconds: f32) -> u32 {
    (100.0 + (level * level) as f32 * 1.5 + game_time_seconds * 0.25) as u32
}

/// Buyback state for a hero.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BuybackState {
    pub cooldown_remaining: f32,
    pub available: bool,
}

// ── Respawn ──

/// Respawn timer formula: 5 + level * 3.8 seconds.
pub fn respawn_time(level: u32) -> f32 {
    5.0 + level as f32 * 3.8
}

/// Tower bounty (split among team).
/// T1=175, T2=200, T3=225, etc.
pub fn tower_bounty(tier: u32) -> u32 {
    150 + 25 * tier
}

// ── Hero economy state ──

/// Full economy state for one hero entity. Attach this as an ECS component
/// for Dota 2-depth gold management.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeroEconomy {
    pub wallet: GoldWallet,
    pub kill_streak: u32,
    pub buyback: BuybackState,
    pub passive_gold_accumulator: f32,
}

impl HeroEconomy {
    pub fn new() -> Self {
        Self {
            wallet: GoldWallet {
                reliable: 0,
                unreliable: STARTING_GOLD,
            },
            buyback: BuybackState {
                cooldown_remaining: 0.0,
                available: true,
            },
            ..Default::default()
        }
    }
}

// ── Free functions operating on HeroEconomy ──

/// Accumulate passive gold (unreliable) over `dt` seconds.
pub fn tick_passive_income(economy: &mut HeroEconomy, dt: f32) {
    economy.passive_gold_accumulator += PASSIVE_GOLD_PER_SECOND * dt;
    let whole = economy.passive_gold_accumulator as u32;
    if whole > 0 {
        economy.wallet.add_unreliable(whole);
        economy.passive_gold_accumulator -= whole as f32;
    }
}

/// Apply death penalty: lose unreliable gold based on level.
pub fn apply_death_penalty(economy: &mut HeroEconomy, level: u32) {
    let loss = gold_loss_on_death(level);
    let actual_loss = loss.min(economy.wallet.unreliable);
    economy.wallet.unreliable -= actual_loss;
    // Death resets kill streak.
    economy.kill_streak = 0;
}

/// Attempt buyback. Returns the respawn time that would have remained
/// (caller uses this to revive the hero). On success, starts the buyback
/// cooldown and deducts gold.
pub fn attempt_buyback(
    economy: &mut HeroEconomy,
    level: u32,
    game_time_seconds: f32,
) -> Result<f32, EconomyError> {
    if economy.buyback.cooldown_remaining > 0.0 {
        return Err(EconomyError::BuybackOnCooldown);
    }

    let cost = buyback_cost(level, game_time_seconds);
    economy.wallet.spend(cost)?;
    economy.buyback.cooldown_remaining = BUYBACK_COOLDOWN;
    economy.buyback.available = false;

    Ok(respawn_time(level))
}

/// Tick buyback cooldown by `dt` seconds.
pub fn tick_buyback_cooldown(economy: &mut HeroEconomy, dt: f32) {
    if economy.buyback.cooldown_remaining > 0.0 {
        economy.buyback.cooldown_remaining = (economy.buyback.cooldown_remaining - dt).max(0.0);
        if economy.buyback.cooldown_remaining == 0.0 {
            economy.buyback.available = true;
        }
    }
}

/// Award kill bounty to killer and assist gold to assisters.
/// Kill gold is reliable. Assist gold is reliable.
/// Killer's streak increments. Victim's streak is used for bounty calculation.
pub fn award_kill(
    killer: &mut HeroEconomy,
    victim_level: u32,
    victim_streak: u32,
    assisters: &mut [&mut HeroEconomy],
) {
    let bounty = hero_kill_bounty(victim_level, victim_streak);
    killer.wallet.add_reliable(bounty);
    killer.kill_streak += 1;

    let num_assists = assisters.len() as u32;
    let per_assist = assist_gold(bounty, num_assists);
    for assister in assisters.iter_mut() {
        assister.wallet.add_reliable(per_assist);
    }
}

// ── ECS systems for HeroEconomy ──

/// Tick passive gold income for all entities with `HeroEconomy`.
pub fn passive_income_system(world: &mut World, dt: f32) {
    let entities: Vec<euca_ecs::Entity> = {
        let query = euca_ecs::Query::<(euca_ecs::Entity, &HeroEconomy)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };
    for entity in entities {
        if let Some(econ) = world.get_mut::<HeroEconomy>(entity) {
            tick_passive_income(econ, dt);
        }
    }
}

/// Tick buyback cooldowns for all entities with `HeroEconomy`.
pub fn buyback_cooldown_system(world: &mut World, dt: f32) {
    let entities: Vec<euca_ecs::Entity> = {
        let query = euca_ecs::Query::<(euca_ecs::Entity, &HeroEconomy)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };
    for entity in entities {
        if let Some(econ) = world.get_mut::<HeroEconomy>(entity) {
            tick_buyback_cooldown(econ, dt);
        }
    }
}

/// Handle hero death economy: apply death penalty to victim and award
/// kill bounty to killer (if the killer also has `HeroEconomy`).
///
/// Non-hero kills (entities without `HeroEconomy`) are handled by the
/// legacy `gold_on_kill_system`. This system only fires for hero victims.
pub fn economy_death_system(world: &mut World) {
    let events: Vec<DeathEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<DeathEvent>().cloned().collect())
        .unwrap_or_default();

    // Collect victim data first, then mutate.
    struct DeathInfo {
        victim: euca_ecs::Entity,
        victim_level: u32,
        victim_streak: u32,
        killer: euca_ecs::Entity,
    }

    let mut deaths = Vec::new();
    for event in &events {
        let killer = match event.killer {
            Some(k) => k,
            None => continue,
        };

        // Only process heroes (entities with HeroEconomy).
        let (victim_level, victim_streak) = match world.get::<HeroEconomy>(event.entity) {
            Some(econ) => {
                let level = world
                    .get::<crate::leveling::Level>(event.entity)
                    .map(|l| l.level)
                    .unwrap_or(1);
                (level, econ.kill_streak)
            }
            None => continue,
        };

        deaths.push(DeathInfo {
            victim: event.entity,
            victim_level,
            victim_streak,
            killer,
        });
    }

    for death in deaths {
        // Apply death penalty to victim.
        if let Some(victim_econ) = world.get_mut::<HeroEconomy>(death.victim) {
            apply_death_penalty(victim_econ, death.victim_level);
        }

        // Award kill bounty to killer (if killer is a hero with HeroEconomy).
        let bounty = hero_kill_bounty(death.victim_level, death.victim_streak);
        if let Some(killer_econ) = world.get_mut::<HeroEconomy>(death.killer) {
            killer_econ.wallet.add_reliable(bounty);
            killer_econ.kill_streak += 1;
            log::info!(
                "Hero {} earned {} reliable gold for hero kill (streak: {})",
                death.killer.index(),
                bounty,
                killer_econ.kill_streak
            );
        }
    }
}

// ── Legacy system (backward compatible) ──

/// Award gold to killers when entities with GoldBounty die.
///
/// This is the simple/legacy system using `Gold(i32)` and `GoldBounty`.
/// For full Dota 2 economy, use `HeroEconomy` + `award_kill`.
pub fn gold_on_kill_system(world: &mut World) {
    let events: Vec<DeathEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<DeathEvent>().cloned().collect())
        .unwrap_or_default();

    for event in events {
        let killer = match event.killer {
            Some(k) => k,
            None => continue,
        };

        // Get bounty from victim
        let bounty = world.get::<GoldBounty>(event.entity).map(|b| b.0);
        let bounty = match bounty {
            Some(b) => b,
            None => continue,
        };

        // Award gold to killer
        if let Some(gold) = world.get_mut::<Gold>(killer) {
            gold.0 += bounty;
            log::info!(
                "Entity {} earned {} gold (total: {})",
                killer.index(),
                bounty,
                gold.0
            );
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::{DeathEvent, Health};

    // -- Legacy system tests (kept for backward compatibility) --

    #[test]
    fn gold_on_kill_awards_bounty() {
        let mut world = euca_ecs::World::new();
        world.insert_resource(Events::default());

        let killer = world.spawn(Gold(0));
        let victim = world.spawn(GoldBounty(100));

        world.resource_mut::<Events>().unwrap().send(DeathEvent {
            entity: victim,
            killer: Some(killer),
        });

        gold_on_kill_system(&mut world);

        assert_eq!(world.get::<Gold>(killer).unwrap().0, 100);
    }

    #[test]
    fn no_gold_without_bounty() {
        let mut world = euca_ecs::World::new();
        world.insert_resource(Events::default());

        let killer = world.spawn(Gold(50));
        let victim = world.spawn(Health::new(100.0));

        world.resource_mut::<Events>().unwrap().send(DeathEvent {
            entity: victim,
            killer: Some(killer),
        });

        gold_on_kill_system(&mut world);

        assert_eq!(world.get::<Gold>(killer).unwrap().0, 50);
    }

    #[test]
    fn no_gold_without_killer() {
        let mut world = euca_ecs::World::new();
        world.insert_resource(Events::default());

        let victim = world.spawn(GoldBounty(100));

        world.resource_mut::<Events>().unwrap().send(DeathEvent {
            entity: victim,
            killer: None,
        });

        gold_on_kill_system(&mut world);
        // Should not panic
    }

    // -- Wallet tests --

    #[test]
    fn test_wallet_spend_unreliable_first() {
        let mut wallet = GoldWallet {
            reliable: 200,
            unreliable: 300,
        };
        assert!(wallet.spend(250).is_ok());
        // Should consume all 250 from unreliable (300 - 250 = 50), reliable untouched.
        assert_eq!(wallet.unreliable, 50);
        assert_eq!(wallet.reliable, 200);
    }

    #[test]
    fn test_wallet_spend_overflows_to_reliable() {
        let mut wallet = GoldWallet {
            reliable: 200,
            unreliable: 100,
        };
        // Spend 250: 100 from unreliable, 150 from reliable.
        assert!(wallet.spend(250).is_ok());
        assert_eq!(wallet.unreliable, 0);
        assert_eq!(wallet.reliable, 50);
    }

    #[test]
    fn test_wallet_insufficient() {
        let mut wallet = GoldWallet {
            reliable: 100,
            unreliable: 100,
        };
        let result = wallet.spend(300);
        assert_eq!(result, Err(EconomyError::InsufficientGold));
        // Wallet unchanged.
        assert_eq!(wallet.reliable, 100);
        assert_eq!(wallet.unreliable, 100);
    }

    // -- Passive income --

    #[test]
    fn test_passive_income() {
        let mut econ = HeroEconomy::new();
        let starting = econ.wallet.total();
        // 2.5 seconds of passive income = 2 gold (fractional accumulates).
        tick_passive_income(&mut econ, 2.5);
        assert_eq!(econ.wallet.total(), starting + 2);
        assert!((econ.passive_gold_accumulator - 0.5).abs() < 0.01);
        // Another 0.5 seconds tips it over.
        tick_passive_income(&mut econ, 0.5);
        assert_eq!(econ.wallet.total(), starting + 3);
    }

    // -- Creep bounty --

    #[test]
    fn test_creep_bounty_melee() {
        // At minute 0: base 38.
        assert_eq!(creep_bounty(CreepType::Melee, 0.0), 38);
        // At minute 10: 38 + 0.7 * 10 = 45.
        assert_eq!(creep_bounty(CreepType::Melee, 10.0), 45);
    }

    #[test]
    fn test_creep_bounty_siege() {
        // At minute 0: base 74.
        assert_eq!(creep_bounty(CreepType::Siege, 0.0), 74);
        // At minute 20: 74 + 1.5 * 20 = 104.
        assert_eq!(creep_bounty(CreepType::Siege, 20.0), 104);
    }

    // -- Hero kill bounty --

    #[test]
    fn test_hero_kill_bounty_base() {
        // Level 1, no streak: 110 + 0*60 + 1*8 = 118.
        assert_eq!(hero_kill_bounty(1, 0), 118);
    }

    #[test]
    fn test_hero_kill_bounty_streak() {
        // Level 5, streak 3: 110 + 3*60 + 5*8 = 110 + 180 + 40 = 330.
        assert_eq!(hero_kill_bounty(5, 3), 330);
    }

    // -- Assist gold --

    #[test]
    fn test_assist_gold_split() {
        let bounty = 300;
        // 2 assisters: 300 * 0.3 / 2 = 45 each.
        assert_eq!(assist_gold(bounty, 2), 45);
        // 0 assisters: 0.
        assert_eq!(assist_gold(bounty, 0), 0);
    }

    // -- Death penalty --

    #[test]
    fn test_gold_loss_on_death() {
        assert_eq!(gold_loss_on_death(1), 30);
        assert_eq!(gold_loss_on_death(10), 300);
        assert_eq!(gold_loss_on_death(25), 750);
    }

    #[test]
    fn test_apply_death_penalty_caps_at_unreliable() {
        let mut econ = HeroEconomy::new();
        econ.wallet.reliable = 500;
        econ.wallet.unreliable = 20; // less than 30*10=300
        econ.kill_streak = 5;
        apply_death_penalty(&mut econ, 10);
        // Should only lose 20 (all unreliable), reliable untouched.
        assert_eq!(econ.wallet.unreliable, 0);
        assert_eq!(econ.wallet.reliable, 500);
        assert_eq!(econ.kill_streak, 0);
    }

    // -- Buyback --

    #[test]
    fn test_buyback_cost() {
        // Level 10, 1200 seconds: 100 + 100*1.5 + 1200*0.25 = 100+150+300 = 550.
        assert_eq!(buyback_cost(10, 1200.0), 550);
    }

    #[test]
    fn test_buyback_cooldown() {
        let mut econ = HeroEconomy::new();
        econ.buyback.cooldown_remaining = BUYBACK_COOLDOWN;
        econ.buyback.available = false;

        // Tick 400 seconds.
        tick_buyback_cooldown(&mut econ, 400.0);
        assert!(!econ.buyback.available);
        assert!((econ.buyback.cooldown_remaining - 80.0).abs() < 0.01);

        // Tick remaining 80 seconds.
        tick_buyback_cooldown(&mut econ, 80.0);
        assert!(econ.buyback.available);
        assert_eq!(econ.buyback.cooldown_remaining, 0.0);
    }

    #[test]
    fn test_attempt_buyback_success() {
        let mut econ = HeroEconomy::new();
        econ.wallet.reliable = 1000;
        econ.wallet.unreliable = 500;
        let result = attempt_buyback(&mut econ, 10, 1200.0);
        assert!(result.is_ok());
        // Cost was 550, consumed from unreliable first (500) then reliable (50).
        assert_eq!(econ.wallet.unreliable, 0);
        assert_eq!(econ.wallet.reliable, 950);
        // Cooldown started.
        assert_eq!(econ.buyback.cooldown_remaining, BUYBACK_COOLDOWN);
        assert!(!econ.buyback.available);
    }

    #[test]
    fn test_attempt_buyback_on_cooldown() {
        let mut econ = HeroEconomy::new();
        econ.wallet.reliable = 9999;
        econ.buyback.cooldown_remaining = 100.0;
        let result = attempt_buyback(&mut econ, 1, 0.0);
        assert_eq!(result, Err(EconomyError::BuybackOnCooldown));
    }

    #[test]
    fn test_attempt_buyback_insufficient_gold() {
        let mut econ = HeroEconomy::new();
        econ.wallet.reliable = 0;
        econ.wallet.unreliable = 10; // not enough
        let result = attempt_buyback(&mut econ, 10, 1200.0);
        assert_eq!(result, Err(EconomyError::InsufficientGold));
    }

    // -- Respawn time --

    #[test]
    fn test_respawn_time() {
        // Level 1: 5 + 1*3.8 = 8.8.
        assert!((respawn_time(1) - 8.8).abs() < 0.01);
        // Level 25: 5 + 25*3.8 = 100.0.
        assert!((respawn_time(25) - 100.0).abs() < 0.01);
    }

    // -- Tower bounty --

    #[test]
    fn test_tower_bounty() {
        assert_eq!(tower_bounty(1), 175); // T1
        assert_eq!(tower_bounty(2), 200); // T2
        assert_eq!(tower_bounty(3), 225); // T3
    }

    // -- Starting gold --

    #[test]
    fn test_starting_gold() {
        let econ = HeroEconomy::new();
        assert_eq!(econ.wallet.total(), STARTING_GOLD);
        assert_eq!(econ.wallet.unreliable, 600);
        assert_eq!(econ.wallet.reliable, 0);
    }

    // -- Full kill flow --

    #[test]
    fn test_award_kill_full_flow() {
        let mut killer = HeroEconomy::new();
        let mut assister1 = HeroEconomy::new();
        let mut assister2 = HeroEconomy::new();

        let victim_level = 10;
        let victim_streak = 3;
        // Bounty: 110 + 3*60 + 10*8 = 110 + 180 + 80 = 370.
        let expected_bounty = hero_kill_bounty(victim_level, victim_streak);
        assert_eq!(expected_bounty, 370);

        {
            let assisters: &mut [&mut HeroEconomy] = &mut [&mut assister1, &mut assister2];
            award_kill(&mut killer, victim_level, victim_streak, assisters);
        }

        // Killer receives full bounty as reliable gold.
        assert_eq!(killer.wallet.reliable, expected_bounty);
        assert_eq!(killer.kill_streak, 1);

        // Each assister: 370 * 0.3 / 2 = 55 reliable gold.
        let expected_assist = assist_gold(expected_bounty, 2);
        assert_eq!(expected_assist, 55);
        assert_eq!(assister1.wallet.reliable, expected_assist);
        assert_eq!(assister2.wallet.reliable, expected_assist);
    }
}
