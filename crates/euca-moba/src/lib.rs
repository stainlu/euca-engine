//! MOBA genre kit for euca engine.
//!
//! This crate contains MOBA-specific gameplay modules that build on the
//! genre-neutral primitives in [`euca_gameplay`]. An agent or game that
//! doesn't need MOBA mechanics should depend on `euca-gameplay` alone.
//!
//! # Modules
//!
//! - **Heroes** — hero definitions, stat growth, hero registry
//! - **Buildings** — towers, barracks, backdoor protection, fortification
//! - **Creeps** — lane wave spawning, routing, denial, last-hit
//! - **Roshan** — boss objective with Aegis / Cheese / Refresher drops
//! - **Shop** — buy/sell items with gold, recipe combining
//! - **Items** — active item abilities, charges, backpack, neutral slot
//! - **Attributes** — STR/AGI/INT per-level growth system
//! - **Fog of war** — vision, day/night cycle, wards
//! - **Tower aggro** — tower target override when enemy attacks ally hero
//! - **Neutral camps** — jungle monsters with leash behavior

/// Dota 2 hero attribute system — STR/AGI/INT with per-level growth and stat conversions.
pub mod attributes;
/// Dota 2 tower and building system — types, backdoor, fortification, aggro, bounties.
pub mod building;
/// ECS systems for buildings — backdoor protection, fortification, barracks death.
pub mod building_systems;
/// Dota 2-style creep wave spawning, lane routing, aggro, denial, and last-hit.
pub mod creep_wave;
/// Fog of war, day/night cycle, and ward system.
pub mod fog_of_war;
/// Hero definitions, per-hero stat growth, and hero registry.
pub mod hero;
/// Active item abilities, cooldowns, charges, backpack, and neutral item slot.
pub mod item_active;
/// Jungle neutral camp monsters with leash behavior.
pub mod neutral_camp;
/// Roshan — the main boss objective, drops Aegis/Cheese/Refresher Shard.
pub mod roshan;
/// Shop system — buy/sell items with gold, recipe combining.
pub mod shop;
/// Tower aggro override — forces towers to target heroes attacking allied heroes.
pub mod tower_aggro;

// Re-export key types at crate root for convenience.

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

pub use creep_wave::{
    CreepAggro, CreepStats, LaneConfig, LaneWaypoints, SpawnWaveEvent, WaveConfig, WaveSpawner,
    can_deny, creep_stats, denial_xp, last_hit_gold, wave_composition, wave_spawn_system,
};

pub use fog_of_war::{
    CellVisibility, DayNightCycle, VisionMap, VisionSource, Ward, WardStock, WardType,
    hero_vision_radius, is_unit_visible, place_ward, tick_ward_stock, tick_wards, update_vision,
};

pub use hero::{AbilityDef, HeroDef, HeroName, HeroRegistry, spawn_hero};

pub use item_active::{
    Backpack, BackpackItem, CooldownGroup, ItemActive, ItemCharges, ItemError, ItemState,
    NeutralItemSlot, can_use_active, consume_charge, swap_to_backpack, tick_charges,
    tick_cooldowns, use_item_active,
};

pub use neutral_camp::{NeutralCamp, neutral_camp_system};

pub use roshan::{
    Aegis, AegisHolder, AegisResurrectTimer, AegisResurrection, Cheese, RefresherShard, Roshan,
    RoshanDrops, RoshanManager, RoshanPickup, RoshanSlam, aegis_system, aegis_trigger, new_aegis,
    new_cheese, new_refresher_shard, new_roshan_slam, pick_up_aegis, respawn_roshan, roshan_dies,
    roshan_respawn_time, roshan_slam_tick, roshan_system, roshan_takes_damage, spawn_roshan,
    tick_aegis, tick_roshan, use_cheese,
};

pub use shop::{RecipeDef, RecipeRegistry, ShopError, buy_item, sell_item};

pub use tower_aggro::{TowerAggroOverride, tower_aggro_system};
