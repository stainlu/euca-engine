//! Hero definition, selection, and listing endpoints.

use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

// ── Request types ──

/// Request body for `POST /hero/define`.
#[derive(Deserialize)]
pub struct HeroDefRequest {
    /// Hero name (e.g. "Juggernaut").
    pub name: String,
    /// Starting health.
    #[serde(default = "default_health")]
    pub health: f32,
    /// Starting mana.
    #[serde(default = "default_mana")]
    pub mana: f32,
    /// Starting gold.
    #[serde(default = "default_gold")]
    pub gold: i32,
    /// Base attack damage.
    #[serde(default = "default_damage")]
    pub damage: f32,
    /// Attack range.
    #[serde(default = "default_range")]
    pub range: f32,
    /// Base stats at level 1 (e.g. {"max_health": 600, "attack_damage": 55}).
    #[serde(default)]
    pub base_stats: std::collections::HashMap<String, f64>,
    /// Stat growth per level (e.g. {"max_health": 80, "attack_damage": 3}).
    #[serde(default)]
    pub stat_growth: std::collections::HashMap<String, f64>,
    /// Ability definitions (Q/W/E/R).
    #[serde(default)]
    pub abilities: Vec<AbilityDefRequest>,
}

fn default_health() -> f32 {
    600.0
}
fn default_mana() -> f32 {
    300.0
}
fn default_gold() -> i32 {
    625
}
fn default_damage() -> f32 {
    50.0
}
fn default_range() -> f32 {
    1.5
}

/// A single ability definition within a hero define request.
#[derive(Deserialize)]
pub struct AbilityDefRequest {
    /// Slot: "Q", "W", "E", or "R".
    pub slot: euca_gameplay::AbilitySlot,
    /// Ability display name.
    pub name: String,
    /// Cooldown in seconds.
    #[serde(default = "default_cooldown")]
    pub cooldown: f32,
    /// Mana cost per use.
    #[serde(default = "default_mana_cost")]
    pub mana_cost: f32,
    /// What the ability does when activated.
    pub effect: euca_gameplay::AbilityEffect,
}

fn default_cooldown() -> f32 {
    10.0
}
fn default_mana_cost() -> f32 {
    100.0
}

// ── Handlers ──

/// POST /hero/define — register a hero definition in the HeroRegistry.
///
/// If the registry resource doesn't exist yet, creates it. If a hero with
/// the same name already exists, overwrites it.
pub async fn hero_define(
    State(world): State<SharedWorld>,
    Json(req): Json<HeroDefRequest>,
) -> Json<MessageResponse> {
    let name = req.name.clone();
    world.with(|w, _| {
        let def = euca_gameplay::HeroDef {
            name: req.name,
            base_stats: req.base_stats,
            growth: req.stat_growth,
            health: req.health,
            mana: req.mana,
            gold: req.gold,
            damage: req.damage,
            range: req.range,
            abilities: req
                .abilities
                .into_iter()
                .map(|a| euca_gameplay::AbilityDef {
                    slot: a.slot,
                    name: a.name,
                    cooldown: a.cooldown,
                    mana_cost: a.mana_cost,
                    effect: a.effect,
                })
                .collect(),
            primary_attribute: None,
            base_attributes: None,
            attribute_growth: None,
            hero_timings: None,
        };

        if let Some(registry) = w.resource_mut::<euca_gameplay::HeroRegistry>() {
            registry.register(def);
        } else {
            let mut registry = euca_gameplay::HeroRegistry::new();
            registry.register(def);
            w.insert_resource(registry);
        }
    });

    Json(MessageResponse {
        ok: true,
        message: Some(format!("Hero '{name}' registered")),
    })
}

/// POST /hero/select — apply a hero template to an existing entity.
///
/// Request body: `{ "entity_id": 0, "hero_name": "Dragon Knight" }`
///
/// Looks up the hero in `HeroRegistry` and applies all hero components
/// (Health, BaseStats, StatGrowth, HeroName, AutoCombat, AbilitySet, Mana,
/// EntityRole::Hero) to the specified entity.
pub async fn hero_select(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let hero_name = req
        .get("hero_name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if hero_name.is_empty() {
        return Json(MessageResponse {
            ok: false,
            message: Some("Missing 'hero_name' field".to_string()),
        });
    }

    let result = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return Err(format!("Entity {entity_id} not found")),
        };

        let def = {
            let registry = match w.resource::<euca_gameplay::HeroRegistry>() {
                Some(r) => r.clone(),
                None => return Err("No HeroRegistry resource".to_string()),
            };
            match registry.get(&hero_name) {
                Some(d) => d.clone(),
                None => return Err(format!("Hero '{hero_name}' not found")),
            }
        };

        // Apply hero template components to the existing entity.
        w.insert(entity, euca_gameplay::HeroName(hero_name.clone()));
        w.insert(entity, euca_gameplay::Health::new(def.health));
        w.insert(entity, euca_gameplay::Mana::new(def.mana, 5.0));
        w.insert(entity, euca_gameplay::Gold::new(def.gold));
        w.insert(entity, euca_gameplay::Level::new(1));
        w.insert(entity, euca_gameplay::BaseStats(def.base_stats.clone()));
        w.insert(entity, euca_gameplay::StatGrowth(def.growth.clone()));
        w.insert(entity, euca_gameplay::EntityRole::Hero);

        let mut combat = euca_gameplay::AutoCombat::new();
        combat.damage = def.damage;
        combat.range = def.range;
        w.insert(entity, combat);

        let mut ability_set = euca_gameplay::AbilitySet::new();
        for ability_def in &def.abilities {
            ability_set.add(
                ability_def.slot,
                euca_gameplay::Ability {
                    name: ability_def.name.clone(),
                    cooldown: ability_def.cooldown,
                    cooldown_remaining: 0.0,
                    mana_cost: ability_def.mana_cost,
                    effect: ability_def.effect.clone(),
                    ..Default::default()
                },
            );
        }
        w.insert(entity, ability_set);

        Ok(format!(
            "Applied hero template '{hero_name}' to entity {entity_id}"
        ))
    });

    match result {
        Ok(msg) => Json(MessageResponse {
            ok: true,
            message: Some(msg),
        }),
        Err(msg) => Json(MessageResponse {
            ok: false,
            message: Some(msg),
        }),
    }
}

/// GET /hero/list — list all available heroes from the HeroRegistry.
pub async fn hero_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let data = world.with_world(|w| {
        let registry = match w.resource::<euca_gameplay::HeroRegistry>() {
            Some(r) => r,
            None => return serde_json::json!({ "heroes": [] }),
        };

        let mut heroes: Vec<serde_json::Value> = registry
            .heroes
            .values()
            .map(|def| {
                serde_json::json!({
                    "name": def.name,
                    "health": def.health,
                    "mana": def.mana,
                    "gold": def.gold,
                    "damage": def.damage,
                })
            })
            .collect();

        // Sort for deterministic output.
        heroes.sort_by(|a, b| {
            a.get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .cmp(b.get("name").and_then(|v| v.as_str()).unwrap_or(""))
        });

        serde_json::json!({ "heroes": heroes })
    });

    Json(data)
}
