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
/// Engine-level assertions — testable expectations as ECS entities.
pub mod assertions;
/// Dota 2 hero attribute system — STR/AGI/INT with per-level growth and stat conversions.
pub mod attributes;
/// Dota 2 tower and building system — types, backdoor, fortification, aggro, bounties.
pub mod building;
/// ECS systems for buildings — backdoor protection, fortification, barracks death.
pub mod building_systems;
/// Camera modes and follow systems.
pub mod camera;
/// Timed corpse/entity cleanup after death.
pub mod cleanup;
/// Projectiles and auto-PvP melee combat.
pub mod combat;
/// Dota 2-accurate combat math formulas (armor, magic resistance, crits, evasion, etc.).
pub mod combat_math;
/// Dota 2-style creep wave spawning, lane routing, aggro, denial, and last-hit.
pub mod creep_wave;
/// Dota 2 crowd control — stun, silence, root, hex, disarm, break, mute, dispel, spell immunity.
pub mod crowd_control;
/// Tabular game data loaded from config files.
pub mod data_table;
/// Dota 2 economy — reliable/unreliable gold, bounties, buyback, respawn.
pub mod economy;
/// Fog of war, day/night cycle, and ward system.
pub mod fog_of_war;
/// Match lifecycle: lobby, countdown, playing, post-match.
pub mod game_state;
/// Hit points, damage events, death detection, and healing.
pub mod health;
/// Hero definitions, per-hero stat growth, and hero registry.
pub mod hero;
/// Data-driven inventory, equipment, and stat aggregation.
pub mod inventory;
/// Active item abilities, cooldowns, charges, backpack, and neutral item slot.
pub mod item_active;
/// Experience points, levels, and XP bounties.
pub mod leveling;
/// Jungle neutral camp monsters with leash behavior.
pub mod neutral_camp;
/// Player hero marker, command queue, and command execution.
pub mod player;
/// Mouse/keyboard input translation to player commands.
pub mod player_input;
/// Game replay recording and playback.
pub mod replay;
/// Roshan — the main boss objective, drops Aegis/Cheese/Refresher Shard.
pub mod roshan;
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
pub use ai::{AiBehavior, AiGoal, ai_system};
pub use assertions::{
    AssertCondition, AssertResult, Assertion, CompareOp, EntityFilter, EvaluationReport, Severity,
    evaluate_assertions, parse_entity_filter,
};
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
    Ability, AbilityBehavior, AbilityEffect, AbilityLevel, AbilityScaling, AbilitySet, AbilitySlot,
    AppliedEffect, CastTime, ChannelState, DamageType, Mana, SpeedBuff, TargetType,
    UseAbilityEvent, ability_tick_system, can_level_ability, interrupt_channel, level_up_ability,
    scaled_value, start_channel, tick_channel, toggle_ability, use_ability_system,
};
pub use cleanup::{CorpseTimer, corpse_cleanup_system};
pub use creep_wave::{
    CreepAggro, CreepStats, LaneConfig, LaneWaypoints, SpawnWaveEvent, WaveConfig, WaveSpawner,
    can_deny, creep_stats, denial_xp, last_hit_gold, wave_composition, wave_spawn_system,
};
pub use crowd_control::{
    CcState, CcType, CrowdControl, DisableFlags, DispelType, SpellImmunity, StatusResistance,
};

/// Tick crowd control durations and spell immunity for all entities with [`CcState`].
///
/// Run this **before** `player_command_system` and `auto_combat_system` each frame
/// so that CC queries reflect up-to-date state.
pub fn cc_tick_system(world: &mut euca_ecs::World, dt: f32) {
    let entities: Vec<euca_ecs::Entity> = {
        let query = euca_ecs::Query::<(euca_ecs::Entity, &CcState)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };
    for entity in entities {
        if let Some(cc) = world.get_mut::<CcState>(entity) {
            cc.remove_expired(dt);
        }
    }
}
pub use economy::{
    BUYBACK_COOLDOWN, BuybackState, CreepType, EconomyError, Gold, GoldBounty, GoldWallet,
    HeroEconomy, PASSIVE_GOLD_PER_SECOND, STARTING_GOLD, apply_death_penalty, assist_gold,
    attempt_buyback, award_kill, buyback_cooldown_system, buyback_cost, creep_bounty,
    economy_death_system, gold_loss_on_death, gold_on_kill_system, hero_kill_bounty,
    passive_income_system, respawn_time, tick_buyback_cooldown, tick_passive_income, tower_bounty,
};
pub use hero::{AbilityDef, HeroDef, HeroName, HeroRegistry, StatGrowth, spawn_hero};
pub use inventory::{
    Equipment, Inventory, ItemDef, ItemRegistry, ItemStack, StatModifiers, add_item, equip,
    equipment_stat_system, find_item, has_space, remove_item, unequip,
};
pub use item_active::{
    Backpack, BackpackItem, CooldownGroup, ItemActive, ItemCharges, ItemError, ItemState,
    NeutralItemSlot, can_use_active, consume_charge, swap_to_backpack, tick_charges,
    tick_cooldowns, use_item_active,
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

pub use attributes::{
    AttributeGrowth as AttrGrowth, BaseAttributes, ComputedAttributes, DerivedStats,
    HeroAttributes, HeroTimings, PrimaryAttribute, attack_interval, attribute_update_system,
    compute_attributes, derive_stats, total_armor, total_attack_speed, total_damage, total_hp,
    total_mana, turn_time,
};

pub use building::{
    BackdoorProtection, BuildingStats, BuildingType, CreepEffect, Fortification, Lane, TowerAggro,
    activate_fortification, backdoor_damage_modifier, barracks_destroyed_effect, building_stats,
    is_building_invulnerable, tick_fortification, tower_bounty as building_tower_bounty,
    update_backdoor_protection, update_tower_aggro,
};

pub use building_systems::{
    DestroyedBarracks, TeamFortifications, backdoor_protection_system, barracks_death_system,
    building_damage_multiplier, building_tower_aggro_system, fortification_tick_system,
};

pub use fog_of_war::{
    CellVisibility, DayNightCycle, VisionMap, VisionSource, Ward, WardStock, WardType,
    hero_vision_radius, is_unit_visible, place_ward, tick_ward_stock, tick_wards, update_vision,
};

pub use roshan::{
    Aegis, AegisHolder, AegisResurrectTimer, AegisResurrection, Cheese, RefresherShard, Roshan,
    RoshanDrops, RoshanManager, RoshanPickup, RoshanSlam, aegis_system, aegis_trigger, new_aegis,
    new_cheese, new_refresher_shard, new_roshan_slam, pick_up_aegis, respawn_roshan, roshan_dies,
    roshan_respawn_time, roshan_slam_tick, roshan_system, roshan_takes_damage, spawn_roshan,
    tick_aegis, tick_roshan, use_cheese,
};
