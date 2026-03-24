//! ECS-native game logic — composable components, systems, events, and resources.
//!
//! This is a **library, not a framework**. Pick the components and systems you need.
//! Game logic emerges from composition:
//!
//! ```text
//! entity + Health + Team + Velocity + MeshRenderer
//! + GameState resource
//! + damage_system + death_system + respawn_system
//! = a deathmatch game
//! ```

/// Cooldown-based abilities (Q/W/E/R) with mana costs and effects.
pub mod abilities;
/// AI behaviors and goal-driven entity logic.
pub mod ai;
/// Camera modes and follow systems.
pub mod camera;
/// Timed corpse/entity cleanup after death.
pub mod cleanup;
/// Projectiles and auto-PvP melee combat.
pub mod combat;
/// Tabular game data loaded from config files.
pub mod data_table;
/// Gold currency, bounties, and kill rewards.
pub mod economy;
/// Match lifecycle: lobby, countdown, playing, post-match.
pub mod game_state;
/// Hit points, damage events, death detection, and healing.
pub mod health;
/// Inventory, equipment, items, and stat modifiers.
pub mod inventory;
/// Experience points, levels, and XP bounties.
pub mod leveling;
/// Player hero marker, command queue, and command execution.
pub mod player;
/// Mouse/keyboard input translation to player commands.
pub mod player_input;
/// Data-driven game rules: "when X happens, do Y" without code.
pub mod rules;
/// Team assignment, spawn points, and respawn timers.
pub mod teams;
/// Spatial trigger zones that fire actions on overlap.
pub mod triggers;

// Re-export key types at crate root for convenience
pub use ai::{AiBehavior, AiGoal, ai_system};
pub use combat::{
    AttackStyle, AutoCombat, CurrentTarget, EntityRole, MarchDirection, Projectile,
    auto_combat_system, projectile_system,
};
pub use data_table::DataTable;
pub use game_state::{GamePhase, GameState, MatchConfig, ScoreEvent, game_state_system};
pub use health::{
    DamageEvent, Dead, DeathEvent, Health, LastAttacker, apply_damage_system, death_check_system,
};
pub use rules::{
    ActionTarget, GameAction, HealthBelowRule, OnDeathRule, OnPhaseRule, OnScoreRule,
    RuleCondition, RuleFilter, RuleSpawnEvent, TimerRule, health_below_rule_system,
    on_death_rule_system, on_phase_rule_system, on_score_rule_system, parse_action, parse_filter,
    parse_when, timer_rule_system,
};
pub use teams::{RespawnTimer, SpawnPoint, Team, respawn_system, start_respawn_on_death};
pub use triggers::{TriggerAction, TriggerZone, trigger_system};

pub use abilities::{
    Ability, AbilityEffect, AbilitySet, AbilitySlot, Mana, SpeedBuff, UseAbilityEvent,
    ability_tick_system, use_ability_system,
};
pub use cleanup::{CorpseTimer, corpse_cleanup_system};
pub use economy::{Gold, GoldBounty, gold_on_kill_system};
pub use inventory::{
    Equipment, Inventory, ItemDef, ItemRegistry, ItemStack, StatModifiers, add_item, equip,
    equipment_stat_system, find_item, has_space, remove_item, unequip,
};
pub use leveling::{Level, XpBounty, xp_on_kill_system};
pub use player::{PlayerCommand, PlayerCommandQueue};
pub use player_input::{ViewportSize, player_input_system, ray_ground_intersection};
