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

/// Engine-level assertions — testable expectations as ECS entities.
pub mod assertions;
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
/// Hero definitions, per-hero stat growth, and hero registry.
pub mod hero;
/// Data-driven inventory, equipment, and stat aggregation.
pub mod inventory;
/// Experience points, levels, and XP bounties.
pub mod leveling;
/// Jungle neutral camp monsters with leash behavior.
pub mod neutral_camp;
/// Player hero marker, command queue, and command execution.
pub mod player;
/// Mouse/keyboard input translation to player commands.
pub mod player_input;
/// Data-driven game rules: "when X happens, do Y" without code.
pub mod rules;
/// Shop system — buy/sell items with gold, recipe combining.
pub mod shop;
/// Stat block and damage resistance — data-driven entity attributes.
pub mod stats;
/// Genre-agnostic status effects (modifiers) with tick effects and cleanse.
pub mod status_effects;
/// Team assignment, spawn points, and respawn timers.
pub mod teams;
/// Genre-agnostic tile maps with square and hex topologies.
pub mod tilemap;
/// Tower aggro override — forces towers to target heroes attacking allied heroes.
pub mod tower_aggro;
/// Spatial trigger zones that fire actions on overlap.
pub mod triggers;
/// Turn & phase management for turn-based games.
pub mod turns;
/// Dynamic entity visibility — per-observer filtering with composable rules.
pub mod visibility;
/// Spatial zones with continuous effects (damage, healing, status effects, shrinking).
pub mod zones;

// Re-export key types at crate root for convenience
pub use assertions::{
    Assertion, AssertCondition, AssertResult, CompareOp, EntityFilter, EvaluationReport, Severity,
    evaluate_assertions, parse_entity_filter,
};
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
pub use visibility::{
    Tags, ViewFilter, VisibilityRule, VisibleTo, ZoneRadius, parse_rule as parse_visibility_rule,
    visibility_system,
};

pub use abilities::{
    Ability, AbilityEffect, AbilitySet, AbilitySlot, AppliedEffect, Mana, SpeedBuff,
    UseAbilityEvent, ability_tick_system, use_ability_system,
};
pub use cleanup::{CorpseTimer, corpse_cleanup_system};
pub use economy::{Gold, GoldBounty, gold_on_kill_system};
pub use hero::{AbilityDef, HeroDef, HeroName, HeroRegistry, StatGrowth, spawn_hero};
pub use inventory::{
    Equipment, Inventory, ItemDef, ItemRegistry, ItemStack, StatModifiers, add_item, equip,
    equipment_stat_system, find_item, has_space, remove_item, unequip,
};
pub use leveling::{Level, XpBounty, XpShareRadius, xp_on_kill_system};
pub use neutral_camp::{NeutralCamp, neutral_camp_system};
pub use player::{PlayerCommand, PlayerCommandQueue};
pub use player_input::{ViewportSize, player_input_system, ray_ground_intersection};
pub use shop::{RecipeDef, RecipeRegistry, ShopError, buy_item, sell_item};
pub use stats::{BaseStats, DamageResistance, ResolvedStats, stat_resolution_system};
pub use status_effects::{
    ModifierOp, StackPolicy, StatModifier, StatusEffect, StatusEffectExpired, StatusEffects,
    TickEffect, apply_status_effect, cleanse, effective_stat, status_effect_tick_system,
};
pub use tilemap::{
    ResourcePool, TileCoord, TileData, TileMap, TileOwnerTable, Topology, tile_income_system,
};
pub use turns::{
    TurnConfig, TurnEvent, TurnState, advance_phase, spend_action_points, turn_system,
};
pub use zones::{Zone, ZoneDynamic, ZoneEffect, ZoneShape, zone_dynamic_system, zone_system};

pub use tower_aggro::{TowerAggroOverride, tower_aggro_system};
