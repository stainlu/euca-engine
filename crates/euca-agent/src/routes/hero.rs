//! Hero selection and listing endpoints.

use axum::Json;
use axum::extract::State;

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

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
