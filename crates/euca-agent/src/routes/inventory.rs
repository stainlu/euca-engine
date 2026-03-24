use axum::Json;
use axum::extract::State;

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

/// POST /item/define — register a new item definition
pub async fn item_define(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let id = req.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let name = req
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("Unnamed")
        .to_string();

    let properties: std::collections::HashMap<String, f64> = req
        .get("properties")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_f64().map(|val| (k.clone(), val)))
                .collect()
        })
        .unwrap_or_default();

    world.with(|w, _| {
        let mut registry = w
            .resource::<euca_gameplay::ItemRegistry>()
            .cloned()
            .unwrap_or_default();
        registry.register(euca_gameplay::ItemDef {
            id,
            name: name.clone(),
            properties,
        });
        w.insert_resource(registry);
    });

    Json(MessageResponse {
        ok: true,
        message: Some(format!("Defined item {id}: {name}")),
    })
}

/// POST /item/give — add items to an entity's inventory
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
            None => return Err(format!("Entity {entity_id} not found")),
        };

        // Auto-create Inventory if missing (8 default slots).
        if w.get::<euca_gameplay::Inventory>(entity).is_none() {
            w.insert(entity, euca_gameplay::Inventory::new(8));
        }

        let inventory = match w.get_mut::<euca_gameplay::Inventory>(entity) {
            Some(inv) => inv,
            None => return Err("Failed to access inventory".into()),
        };

        let leftover = euca_gameplay::add_item(inventory, item_id, count);
        if leftover > 0 {
            Ok(format!(
                "Added {} of item {item_id} (inventory full, {leftover} not added)",
                count - leftover
            ))
        } else {
            Ok(format!("Added {count} of item {item_id}"))
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

/// POST /item/equip — equip an item from inventory into a named slot
pub async fn item_equip(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let slot = req
        .get("slot")
        .and_then(|v| v.as_str())
        .unwrap_or("weapon")
        .to_string();
    let item_id = req.get("item_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let result = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return Err(format!("Entity {entity_id} not found")),
        };

        // Auto-create Equipment if missing.
        if w.get::<euca_gameplay::Equipment>(entity).is_none() {
            w.insert(entity, euca_gameplay::Equipment::default());
        }
        // Auto-create StatModifiers if missing.
        if w.get::<euca_gameplay::StatModifiers>(entity).is_none() {
            w.insert(entity, euca_gameplay::StatModifiers::default());
        }

        // Clone-mutate-writeback: ECS allows only one mutable borrow at a time.
        let mut inventory = match w.get::<euca_gameplay::Inventory>(entity).cloned() {
            Some(inv) => inv,
            None => return Err("Entity has no inventory".into()),
        };
        let mut equipment = w
            .get::<euca_gameplay::Equipment>(entity)
            .cloned()
            .unwrap_or_default();

        let ok = euca_gameplay::equip(&mut inventory, &mut equipment, &slot, item_id);

        // Write back.
        if let Some(inv) = w.get_mut::<euca_gameplay::Inventory>(entity) {
            *inv = inventory;
        }
        if let Some(eq) = w.get_mut::<euca_gameplay::Equipment>(entity) {
            *eq = equipment;
        }

        if ok {
            Ok(format!("Equipped item {item_id} in slot '{slot}'"))
        } else {
            Err(format!("Item {item_id} not found in inventory"))
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

/// GET /item/list/{entity_id} — list entity's inventory and equipment
pub async fn item_list(
    State(world): State<SharedWorld>,
    axum::extract::Path(entity_id): axum::extract::Path<u32>,
) -> Json<serde_json::Value> {
    let data = world.with_world(|w| {
        let entity = find_entity(w, entity_id)?;
        let registry = w.resource::<euca_gameplay::ItemRegistry>();

        let resolve_name = |id: u32| -> &str {
            registry
                .and_then(|reg| reg.get(id))
                .map(|def| def.name.as_str())
                .unwrap_or("unknown")
        };

        let inventory = w.get::<euca_gameplay::Inventory>(entity).map(|inv| {
            inv.slots
                .iter()
                .enumerate()
                .filter_map(|(i, slot)| {
                    slot.map(|s| {
                        serde_json::json!({
                            "slot": i,
                            "item_id": s.item_id,
                            "name": resolve_name(s.item_id),
                            "count": s.count,
                        })
                    })
                })
                .collect::<Vec<_>>()
        });

        let equipment = w.get::<euca_gameplay::Equipment>(entity).map(|eq| {
            eq.equipped
                .iter()
                .map(|(slot, item_id)| {
                    serde_json::json!({
                        "slot": slot,
                        "item_id": item_id,
                        "name": resolve_name(*item_id),
                    })
                })
                .collect::<Vec<_>>()
        });

        let stats = w
            .get::<euca_gameplay::StatModifiers>(entity)
            .map(|s| &s.values)
            .cloned();

        Some(serde_json::json!({
            "entity_id": entity_id,
            "inventory": inventory.unwrap_or_default(),
            "equipment": equipment.unwrap_or_default(),
            "stat_modifiers": stats.unwrap_or_default(),
        }))
    });

    Json(data.unwrap_or(serde_json::json!({"error": "Entity not found"})))
}
