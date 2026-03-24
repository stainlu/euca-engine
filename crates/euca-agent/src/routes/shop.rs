use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

#[derive(Deserialize)]
pub struct ShopTransactionRequest {
    pub entity_id: u32,
    pub item_id: u32,
}

/// POST /shop/buy — buy an item (or combine via recipe) for an entity.
pub async fn shop_buy(
    State(world): State<SharedWorld>,
    Json(req): Json<ShopTransactionRequest>,
) -> Json<MessageResponse> {
    let result = world.with(|w, _| {
        let entity = match find_entity(w, req.entity_id) {
            Some(e) => e,
            None => return Err(format!("Entity {} not found", req.entity_id)),
        };
        euca_gameplay::buy_item(w, entity, req.item_id).map_err(|e| e.to_string())
    });

    match result {
        Ok(()) => Json(MessageResponse {
            ok: true,
            message: Some(format!("Bought item {}", req.item_id)),
        }),
        Err(msg) => Json(MessageResponse {
            ok: false,
            message: Some(msg),
        }),
    }
}

/// POST /shop/sell — sell an item from inventory for 50% gold.
pub async fn shop_sell(
    State(world): State<SharedWorld>,
    Json(req): Json<ShopTransactionRequest>,
) -> Json<MessageResponse> {
    let result = world.with(|w, _| {
        let entity = match find_entity(w, req.entity_id) {
            Some(e) => e,
            None => return Err(format!("Entity {} not found", req.entity_id)),
        };
        euca_gameplay::sell_item(w, entity, req.item_id).map_err(|e| e.to_string())
    });

    match result {
        Ok(()) => Json(MessageResponse {
            ok: true,
            message: Some(format!("Sold item {}", req.item_id)),
        }),
        Err(msg) => Json(MessageResponse {
            ok: false,
            message: Some(msg),
        }),
    }
}

/// GET /shop/list — list all items from the ItemRegistry with their costs.
pub async fn shop_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let data = world.with_world(|w| {
        let registry = match w.resource::<euca_gameplay::ItemRegistry>() {
            Some(r) => r,
            None => return serde_json::json!({ "items": [] }),
        };

        let mut items: Vec<serde_json::Value> = registry
            .items
            .values()
            .map(|def| {
                serde_json::json!({
                    "id": def.id,
                    "name": def.name,
                    "cost": def.properties.get("cost").copied().unwrap_or(0.0) as i32,
                    "properties": def.properties,
                })
            })
            .collect();
        items.sort_by_key(|v| v["id"].as_u64().unwrap_or(0));

        serde_json::json!({ "items": items })
    });

    Json(data)
}
