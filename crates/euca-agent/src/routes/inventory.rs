use axum::Json;
use axum::extract::{Path, State};

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

/// POST /item/define — register an item definition in the ItemRegistry.
pub async fn item_define(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let id = req.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let name = req
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unnamed")
        .to_string();

    let mut item = euca_gameplay::ItemDef::new(id, &name);

    if let Some(props) = req.get("properties").and_then(|v| v.as_object()) {
        for (k, v) in props {
            if let Some(val) = v.as_f64() {
                item.properties.insert(k.clone(), val);
            }
        }
    }

    world.with(|w, _| {
        let registry = w
            .resource_mut::<euca_gameplay::ItemRegistry>()
            .expect("ItemRegistry not found");
        registry.register(item);
    });

    Json(MessageResponse {
        ok: true,
        message: Some(format!("Defined item {id}: {name}")),
    })
}

/// POST /item/give — add an item to an entity's inventory.
pub async fn item_give(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let item_id = req.get("item_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let count = req.get("count").and_then(|v| v.as_u64()).unwrap_or(1) as u32;

    let result = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => {
                return Err(format!("Entity {entity_id} not found"));
            }
        };

        if euca_gameplay::add_item(w, entity, item_id, count) {
            Ok(format!(
                "Gave {count}x item {item_id} to entity {entity_id}"
            ))
        } else {
            Err("Failed to give item: entity has no Inventory or inventory is full".to_string())
        }
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

/// POST /item/equip — equip an item from inventory into a named slot.
pub async fn item_equip(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let item_id = req.get("item_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let slot = req
        .get("slot")
        .and_then(|v| v.as_str())
        .unwrap_or("weapon")
        .to_string();

    let result = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return Err(format!("Entity {entity_id} not found")),
        };

        if euca_gameplay::equip(w, entity, &slot, item_id) {
            Ok(format!(
                "Equipped item {item_id} in slot '{slot}' on entity {entity_id}"
            ))
        } else {
            Err("Failed to equip: item not in inventory or no Equipment component".to_string())
        }
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

/// GET /item/list/{entity_id} — list an entity's inventory and equipment.
pub async fn item_list(
    State(world): State<SharedWorld>,
    Path(entity_id): Path<u32>,
) -> Json<serde_json::Value> {
    let data = world.with_world(|w| {
        let entity = find_entity(w, entity_id)?;
        let registry = w.resource::<euca_gameplay::ItemRegistry>();

        let inventory = w.get::<euca_gameplay::Inventory>(entity).map(|inv| {
            let slots: Vec<serde_json::Value> = inv
                .slots
                .iter()
                .enumerate()
                .map(|(i, slot)| match slot {
                    Some(stack) => {
                        let item_name = registry
                            .and_then(|r| r.get(stack.item_id))
                            .map(|d| d.name.as_str())
                            .unwrap_or("unknown");
                        serde_json::json!({
                            "slot": i,
                            "item_id": stack.item_id,
                            "name": item_name,
                            "count": stack.count,
                        })
                    }
                    None => serde_json::json!({"slot": i, "empty": true}),
                })
                .collect();
            serde_json::json!({
                "max_slots": inv.max_slots,
                "slots": slots,
            })
        });

        let equipment = w.get::<euca_gameplay::Equipment>(entity).map(|eq| {
            let equipped: serde_json::Map<String, serde_json::Value> = eq
                .equipped
                .iter()
                .map(|(slot_name, item_id)| {
                    let item_name = registry
                        .and_then(|r| r.get(*item_id))
                        .map(|d| d.name.as_str())
                        .unwrap_or("unknown");
                    (
                        slot_name.clone(),
                        serde_json::json!({"item_id": item_id, "name": item_name}),
                    )
                })
                .collect();
            serde_json::Value::Object(equipped)
        });

        let stat_modifiers = w
            .get::<euca_gameplay::StatModifiers>(entity)
            .map(|sm| serde_json::json!(sm.modifiers));

        Some(serde_json::json!({
            "entity_id": entity_id,
            "inventory": inventory,
            "equipment": equipment,
            "stat_modifiers": stat_modifiers,
        }))
    });

    Json(data.unwrap_or(serde_json::json!({"error": "Entity not found"})))
}
