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

pub mod ai;
pub mod combat;
pub mod data_table;
pub mod game_state;
pub mod health;
pub mod rules;
pub mod teams;
pub mod triggers;

// Re-export key types at crate root for convenience
pub use ai::{AiBehavior, AiGoal, ai_system};
pub use combat::{AttackStyle, AutoCombat, Projectile, auto_combat_system, projectile_system};
pub use data_table::DataTable;
pub use game_state::{GamePhase, GameState, MatchConfig, ScoreEvent, game_state_system};
pub use health::{DamageEvent, Dead, DeathEvent, Health, apply_damage_system, death_check_system};
pub use rules::{
    ActionTarget, GameAction, HealthBelowRule, OnDeathRule, OnPhaseRule, OnScoreRule,
    RuleCondition, RuleFilter, RuleSpawnEvent, TimerRule, health_below_rule_system,
    on_death_rule_system, on_phase_rule_system, on_score_rule_system, parse_action, parse_filter,
    parse_when, timer_rule_system,
};
pub use teams::{RespawnTimer, SpawnPoint, Team, respawn_system, start_respawn_on_death};
pub use triggers::{TriggerAction, TriggerZone, trigger_system};
