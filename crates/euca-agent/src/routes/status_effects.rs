use axum::Json;
use axum::extract::{Path, State};

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

/// POST /effect/apply — apply a status effect to an entity
pub async fn effect_apply(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let tag = req
        .get("tag")
        .and_then(|v| v.as_str())
        .unwrap_or("effect")
        .to_string();
    let duration = req.get("duration").and_then(|v| v.as_f64()).unwrap_or(5.0) as f32;

    // Parse modifiers from array of "stat:op:value" strings.
    let modifiers: Vec<euca_gameplay::StatModifier> = req
        .get("modifiers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .filter_map(parse_modifier)
                .collect()
        })
        .unwrap_or_default();

    // Parse stack policy.
    let stack_policy = match req.get("stack_policy").and_then(|v| v.as_str()) {
        Some(s) if s.starts_with("stack") => {
            let max = s
                .strip_prefix("stack:")
                .and_then(|n| n.parse::<u32>().ok())
                .unwrap_or(5);
            euca_gameplay::StackPolicy::Stack { max }
        }
        _ => euca_gameplay::StackPolicy::Replace,
    };

    // Parse tick effect.
    let tick_effect = req
        .get("tick_effect")
        .and_then(|v| v.as_str())
        .and_then(parse_tick_effect);

    // Parse source entity.
    let source_id = req.get("source").and_then(|v| v.as_u64()).map(|v| v as u32);

    let result = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return Err(format!("Entity {entity_id} not found")),
        };

        let source = source_id.and_then(|id| find_entity(w, id));

        let effect = euca_gameplay::StatusEffect {
            tag: tag.clone(),
            modifiers,
            duration,
            remaining: duration,
            source,
            stack_policy,
            tick_effect,
        };

        euca_gameplay::apply_status_effect(w, entity, effect);
        Ok(())
    });

    match result {
        Ok(()) => Json(serde_json::json!({
            "ok": true,
            "message": format!("Applied effect '{tag}' to entity {entity_id} for {duration}s"),
        })),
        Err(msg) => Json(serde_json::json!({ "ok": false, "error": msg })),
    }
}

/// GET /effect/list/:id — list active status effects on an entity
pub async fn effect_list(
    State(world): State<SharedWorld>,
    Path(id): Path<u32>,
) -> Json<serde_json::Value> {
    let data = world.with_world(|w| {
        let entity = find_entity(w, id)?;
        let effects = w.get::<euca_gameplay::StatusEffects>(entity)?;

        let list: Vec<serde_json::Value> = effects
            .effects
            .iter()
            .map(|e| {
                let mods: Vec<String> = e
                    .modifiers
                    .iter()
                    .map(|m| {
                        let op = match m.op {
                            euca_gameplay::ModifierOp::Set => "set",
                            euca_gameplay::ModifierOp::Add => "add",
                            euca_gameplay::ModifierOp::Multiply => "multiply",
                        };
                        format!("{}:{}:{}", m.stat, op, m.value)
                    })
                    .collect();

                let tick = e.tick_effect.as_ref().map(|t| match t {
                    euca_gameplay::TickEffect::DamagePerSecond(v) => format!("dps:{v}"),
                    euca_gameplay::TickEffect::HealPerSecond(v) => format!("hps:{v}"),
                    euca_gameplay::TickEffect::Custom(s) => format!("custom:{s}"),
                });

                serde_json::json!({
                    "tag": e.tag,
                    "remaining": e.remaining,
                    "duration": e.duration,
                    "modifiers": mods,
                    "tick_effect": tick,
                    "source": e.source.map(|s| s.index()),
                })
            })
            .collect();

        Some(serde_json::json!({
            "entity_id": id,
            "effects": list,
            "count": list.len(),
        }))
    });

    Json(data.unwrap_or(serde_json::json!({"error": "Entity not found or has no status effects"})))
}

/// POST /effect/cleanse — remove effects matching a tag filter
pub async fn effect_cleanse(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let filter = req
        .get("filter")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let result = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return (false, format!("Entity {entity_id} not found")),
        };
        let removed = euca_gameplay::cleanse(w, entity, &filter);
        (
            true,
            format!("Removed {removed} effects matching '{filter}' from entity {entity_id}"),
        )
    });

    Json(MessageResponse {
        ok: result.0,
        message: Some(result.1),
    })
}

// ── Helpers ──

/// Parse "stat:op:value" into a StatModifier.
fn parse_modifier(s: &str) -> Option<euca_gameplay::StatModifier> {
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    if parts.len() != 3 {
        return None;
    }
    let stat = parts[0].to_string();
    let op = match parts[1] {
        "set" => euca_gameplay::ModifierOp::Set,
        "add" => euca_gameplay::ModifierOp::Add,
        "multiply" | "mul" => euca_gameplay::ModifierOp::Multiply,
        _ => return None,
    };
    let value: f64 = parts[2].parse().ok()?;
    Some(euca_gameplay::StatModifier { stat, op, value })
}

/// Parse tick effect string: "dps:N", "hps:N", or "custom:name".
fn parse_tick_effect(s: &str) -> Option<euca_gameplay::TickEffect> {
    if let Some(rest) = s.strip_prefix("dps:") {
        let v: f32 = rest.parse().ok()?;
        Some(euca_gameplay::TickEffect::DamagePerSecond(v))
    } else if let Some(rest) = s.strip_prefix("hps:") {
        let v: f32 = rest.parse().ok()?;
        Some(euca_gameplay::TickEffect::HealPerSecond(v))
    } else {
        s.strip_prefix("custom:")
            .map(|rest| euca_gameplay::TickEffect::Custom(rest.to_string()))
    }
}
