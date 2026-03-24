use axum::Json;
use axum::extract::State;

use euca_ecs::Entity;
use euca_math::Vec3;
use euca_physics::Velocity;
use euca_scene::{GlobalTransform, LocalTransform};

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

/// POST /game/create — create a match with config
pub async fn game_create(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let mode = req
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("deathmatch")
        .to_string();
    let score_limit = req
        .get("score_limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(10) as i32;
    let time_limit = req
        .get("time_limit")
        .and_then(|v| v.as_f64())
        .unwrap_or(300.0) as f32;
    let respawn_delay = req
        .get("respawn_delay")
        .and_then(|v| v.as_f64())
        .unwrap_or(3.0) as f32;

    world.with(|w, _| {
        let config = euca_gameplay::MatchConfig {
            mode: mode.clone(),
            score_limit,
            time_limit,
            respawn_delay,
        };
        let mut state = euca_gameplay::GameState::new(config);
        state.start();
        w.insert_resource(state);
    });

    Json(MessageResponse {
        ok: true,
        message: Some(format!("Match created: {mode}, score limit {score_limit}")),
    })
}

/// GET /game/state — get match state and scores
pub async fn game_state(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let data = world.with_world(|w| {
        w.resource::<euca_gameplay::GameState>().map(|state| {
            let phase = match &state.phase {
                euca_gameplay::GamePhase::Lobby => "lobby",
                euca_gameplay::GamePhase::Countdown { .. } => "countdown",
                euca_gameplay::GamePhase::Playing => "playing",
                euca_gameplay::GamePhase::PostMatch { .. } => "post_match",
            };
            serde_json::json!({
                "phase": phase,
                "mode": state.config.mode,
                "elapsed": state.elapsed,
                "scores": state.scoreboard().iter()
                    .map(|(idx, score)| serde_json::json!({"entity": idx, "score": score}))
                    .collect::<Vec<_>>(),
            })
        })
    });

    Json(
        data.unwrap_or(serde_json::json!({"error": "No game state. Use POST /game/create first."})),
    )
}

/// POST /trigger/create — create a trigger zone entity
pub async fn trigger_create(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let pos = req
        .get("position")
        .and_then(|v| v.as_array())
        .map(|a| {
            Vec3::new(
                a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            )
        })
        .unwrap_or(Vec3::ZERO);

    let half = req
        .get("zone")
        .and_then(|v| v.as_array())
        .map(|a| {
            Vec3::new(
                a.first().and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            )
        })
        .unwrap_or(Vec3::new(1.0, 1.0, 1.0));

    let action_str = req
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("damage:10");

    let action = if let Some(rest) = action_str.strip_prefix("damage:") {
        let amount = rest.parse::<f32>().unwrap_or(10.0);
        euca_gameplay::TriggerAction::Damage { amount }
    } else if let Some(rest) = action_str.strip_prefix("heal:") {
        let amount = rest.parse::<f32>().unwrap_or(10.0);
        euca_gameplay::TriggerAction::Heal { amount }
    } else {
        euca_gameplay::TriggerAction::Damage { amount: 10.0 }
    };

    let entity_id = world.with(|w, _| {
        let transform = euca_math::Transform::from_translation(pos);
        let entity = w.spawn(LocalTransform(transform));
        w.insert(entity, GlobalTransform::default());
        w.insert(entity, euca_gameplay::TriggerZone::new(half, action));
        entity.index()
    });

    Json(serde_json::json!({
        "ok": true,
        "entity_id": entity_id,
        "message": format!("Trigger zone created at ({}, {}, {})", pos.x, pos.y, pos.z),
    }))
}

/// POST /projectile/spawn — spawn a projectile
pub async fn projectile_spawn(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let from = req
        .get("from")
        .and_then(|v| v.as_array())
        .map(|a| {
            Vec3::new(
                a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            )
        })
        .unwrap_or(Vec3::ZERO);

    let direction = req
        .get("direction")
        .and_then(|v| v.as_array())
        .map(|a| {
            Vec3::new(
                a.first().and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            )
        })
        .unwrap_or(Vec3::new(1.0, 0.0, 0.0));

    let speed = req.get("speed").and_then(|v| v.as_f64()).unwrap_or(20.0) as f32;
    let damage = req.get("damage").and_then(|v| v.as_f64()).unwrap_or(25.0) as f32;
    let lifetime = req.get("lifetime").and_then(|v| v.as_f64()).unwrap_or(3.0) as f32;

    let entity_id = world.with(|w, _| {
        let owner = Entity::from_raw(0, 0);
        let transform = euca_math::Transform::from_translation(from);
        let entity = w.spawn(LocalTransform(transform));
        w.insert(entity, GlobalTransform::default());
        w.insert(
            entity,
            euca_gameplay::Projectile::new(direction, speed, damage, lifetime, owner),
        );
        entity.index()
    });

    Json(serde_json::json!({
        "ok": true,
        "entity_id": entity_id,
    }))
}

/// POST /ai/set — set AI behavior on an entity
pub async fn ai_set(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let behavior = req
        .get("behavior")
        .and_then(|v| v.as_str())
        .unwrap_or("idle");
    let target_id = req.get("target").and_then(|v| v.as_u64()).map(|v| v as u32);
    let speed = req.get("speed").and_then(|v| v.as_f64()).unwrap_or(3.0) as f32;

    let ok = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return false,
        };

        let goal = match behavior {
            "idle" => {
                let pos = w
                    .get::<LocalTransform>(entity)
                    .map(|lt| lt.0.translation)
                    .unwrap_or(Vec3::ZERO);
                euca_gameplay::AiGoal::idle(pos)
            }
            "chase" => {
                let target = target_id
                    .and_then(|id| find_entity(w, id))
                    .unwrap_or(Entity::from_raw(0, 0));
                euca_gameplay::AiGoal::chase(target, speed)
            }
            "patrol" => {
                let waypoints = req
                    .get("waypoints")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|wp| {
                                wp.as_array().map(|a| {
                                    Vec3::new(
                                        a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                                        a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                                        a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                                    )
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                euca_gameplay::AiGoal::patrol(waypoints, speed)
            }
            _ => {
                let pos = w
                    .get::<LocalTransform>(entity)
                    .map(|lt| lt.0.translation)
                    .unwrap_or(Vec3::ZERO);
                euca_gameplay::AiGoal::idle(pos)
            }
        };

        if w.get::<Velocity>(entity).is_none() {
            w.insert(entity, Velocity::default());
        }
        w.insert(entity, goal);
        true
    });

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("Set entity {entity_id} AI to {behavior}")
        } else {
            format!("Entity {entity_id} not found")
        }),
    })
}

/// POST /rule/create — create a game rule
pub async fn rule_create(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let when_str = req.get("when").and_then(|v| v.as_str()).unwrap_or("");
    let filter_str = req.get("filter").and_then(|v| v.as_str()).unwrap_or("any");
    let action_strs: Vec<String> = req
        .get("actions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let condition = match euca_gameplay::parse_when(when_str) {
        Some(c) => c,
        None => {
            return Json(serde_json::json!({
                "ok": false,
                "error": format!("Unknown condition: '{when_str}'. Use: death, timer:N, health-below:N"),
            }));
        }
    };

    let filter = euca_gameplay::parse_filter(filter_str).unwrap_or(euca_gameplay::RuleFilter::Any);

    let actions: Vec<euca_gameplay::GameAction> = action_strs
        .iter()
        .filter_map(|s| euca_gameplay::parse_action(s))
        .collect();

    if actions.is_empty() {
        return Json(serde_json::json!({
            "ok": false,
            "error": "No valid actions. Use: spawn, damage, heal, score, despawn, teleport, color, text",
        }));
    }

    // Wrap once in Arc -- cloning inside the ECS systems is then a cheap refcount bump.
    let actions = std::sync::Arc::new(actions);

    let rule_id = world.with(|w, _| match condition {
        euca_gameplay::RuleCondition::Death => {
            let entity = w.spawn(euca_gameplay::OnDeathRule { filter, actions });
            entity.index()
        }
        euca_gameplay::RuleCondition::Timer(interval) => {
            let entity = w.spawn(euca_gameplay::TimerRule {
                interval,
                elapsed: 0.0,
                repeat: true,
                actions,
            });
            entity.index()
        }
        euca_gameplay::RuleCondition::HealthBelow(threshold) => {
            let entity = w.spawn(euca_gameplay::HealthBelowRule {
                filter,
                threshold,
                triggered_entities: std::collections::HashSet::new(),
                actions,
            });
            entity.index()
        }
        euca_gameplay::RuleCondition::Score(threshold) => {
            let entity = w.spawn(euca_gameplay::OnScoreRule {
                score_threshold: threshold,
                triggered: false,
                actions,
            });
            entity.index()
        }
        euca_gameplay::RuleCondition::Phase(phase) => {
            let entity = w.spawn(euca_gameplay::OnPhaseRule {
                phase,
                triggered: false,
                actions,
            });
            entity.index()
        }
    });

    Json(serde_json::json!({
        "ok": true,
        "rule_id": rule_id,
        "when": when_str,
    }))
}

/// GET /rule/list — list all rules
pub async fn rule_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let rules = world.with_world(|w| {
        let mut rules = Vec::new();

        let death_rules = euca_ecs::Query::<(euca_ecs::Entity, &euca_gameplay::OnDeathRule)>::new(w);
        for (e, _r) in death_rules.iter() {
            rules.push(serde_json::json!({"id": e.index(), "type": "on_death"}));
        }

        let timer_rules = euca_ecs::Query::<(euca_ecs::Entity, &euca_gameplay::TimerRule)>::new(w);
        for (e, t) in timer_rules.iter() {
            rules.push(serde_json::json!({"id": e.index(), "type": "timer", "interval": t.interval}));
        }

        let health_rules =
            euca_ecs::Query::<(euca_ecs::Entity, &euca_gameplay::HealthBelowRule)>::new(w);
        for (e, h) in health_rules.iter() {
            rules.push(serde_json::json!({"id": e.index(), "type": "health_below", "threshold": h.threshold}));
        }

        rules
    });

    Json(serde_json::json!({"rules": rules, "count": rules.len()}))
}

/// POST /ability/use — activate an ability on an entity
pub async fn ability_use(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<super::MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let slot_str = req.get("slot").and_then(|v| v.as_str()).unwrap_or("Q");

    let slot = match slot_str {
        "W" | "w" => euca_gameplay::AbilitySlot::W,
        "E" | "e" => euca_gameplay::AbilitySlot::E,
        "R" | "r" => euca_gameplay::AbilitySlot::R,
        _ => euca_gameplay::AbilitySlot::Q,
    };

    let ok = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return false,
        };
        if let Some(events) = w.resource_mut::<euca_ecs::Events>() {
            events.send(euca_gameplay::UseAbilityEvent { entity, slot });
        }
        true
    });

    Json(super::MessageResponse {
        ok,
        message: Some(if ok {
            format!("Used ability {slot_str} on entity {entity_id}")
        } else {
            format!("Entity {entity_id} not found")
        }),
    })
}

/// GET /ability/list/:id — list entity's abilities, cooldowns, mana
pub async fn ability_list(
    State(world): State<SharedWorld>,
    axum::extract::Path(id): axum::extract::Path<u32>,
) -> Json<serde_json::Value> {
    let data = world.with_world(|w| {
        let entity = find_entity(w, id)?;
        let abilities = w.get::<euca_gameplay::AbilitySet>(entity).map(|set| {
            set.abilities
                .iter()
                .map(|(slot, ability)| {
                    serde_json::json!({
                        "slot": format!("{:?}", slot),
                        "name": ability.name,
                        "cooldown": ability.cooldown,
                        "cooldown_remaining": ability.cooldown_remaining,
                        "mana_cost": ability.mana_cost,
                        "ready": ability.is_ready(),
                    })
                })
                .collect::<Vec<_>>()
        });
        let mana = w
            .get::<euca_gameplay::Mana>(entity)
            .map(|m| serde_json::json!({"current": m.current, "max": m.max, "regen": m.regen}));
        let gold = w.get::<euca_gameplay::Gold>(entity).map(|g| g.0);
        let level = w
            .get::<euca_gameplay::Level>(entity)
            .map(|l| serde_json::json!({"level": l.level, "xp": l.xp, "xp_to_next": l.xp_to_next}));

        Some(serde_json::json!({
            "entity_id": id,
            "abilities": abilities.unwrap_or_default(),
            "mana": mana,
            "gold": gold,
            "level": level,
        }))
    });

    Json(data.unwrap_or(serde_json::json!({"error": "Entity not found"})))
}

// ── Status Effects ──

/// POST /effect/apply — apply a status effect to an entity
pub async fn effect_apply(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<super::MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let tag = req
        .get("tag")
        .and_then(|v| v.as_str())
        .unwrap_or("effect")
        .to_string();
    let duration = req.get("duration").and_then(|v| v.as_f64()).unwrap_or(5.0) as f32;

    let modifiers: Vec<euca_gameplay::StatModifier> = req
        .get("modifiers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let s = v.as_str()?;
                    parse_modifier(s)
                })
                .collect()
        })
        .unwrap_or_default();

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

    let tick_effect = req
        .get("tick_effect")
        .and_then(|v| v.as_str())
        .and_then(parse_tick_effect);

    let source_id = req.get("source").and_then(|v| v.as_u64()).map(|v| v as u32);

    let ok = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return false,
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

        euca_gameplay::apply_effect(w, entity, effect);
        true
    });

    Json(super::MessageResponse {
        ok,
        message: Some(if ok {
            format!("Applied effect '{tag}' to entity {entity_id} for {duration}s")
        } else {
            format!("Entity {entity_id} not found")
        }),
    })
}

/// GET /effect/list/:id — list active status effects on an entity
pub async fn effect_list(
    State(world): State<SharedWorld>,
    axum::extract::Path(id): axum::extract::Path<u32>,
) -> Json<serde_json::Value> {
    let data = world.with_world(|w| {
        let entity = find_entity(w, id)?;
        let effects = w.get::<euca_gameplay::StatusEffects>(entity).map(|status| {
            status
                .effects
                .iter()
                .map(|e| {
                    let modifiers: Vec<serde_json::Value> = e
                        .modifiers
                        .iter()
                        .map(|m| {
                            let op = match m.op {
                                euca_gameplay::ModifierOp::Set => "set",
                                euca_gameplay::ModifierOp::Add => "add",
                                euca_gameplay::ModifierOp::Multiply => "multiply",
                            };
                            serde_json::json!({
                                "stat": m.stat,
                                "op": op,
                                "value": m.value,
                            })
                        })
                        .collect();

                    let tick = e.tick_effect.as_ref().map(|t| match t {
                        euca_gameplay::TickEffect::DamagePerSecond(dps) => {
                            format!("dps:{dps}")
                        }
                        euca_gameplay::TickEffect::HealPerSecond(hps) => {
                            format!("hps:{hps}")
                        }
                        euca_gameplay::TickEffect::Custom(tag) => {
                            format!("custom:{tag}")
                        }
                    });

                    serde_json::json!({
                        "tag": e.tag,
                        "duration": e.duration,
                        "remaining": e.remaining,
                        "modifiers": modifiers,
                        "source": e.source.map(|s| s.index()),
                        "tick_effect": tick,
                    })
                })
                .collect::<Vec<_>>()
        });

        Some(serde_json::json!({
            "entity_id": id,
            "effects": effects.unwrap_or_default(),
        }))
    });

    Json(data.unwrap_or(serde_json::json!({"error": "Entity not found"})))
}

/// POST /effect/cleanse — remove effects matching a tag filter
pub async fn effect_cleanse(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<super::MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let filter = req
        .get("filter")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let result = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return None,
        };
        Some(euca_gameplay::cleanse(w, entity, &filter))
    });

    match result {
        Some(removed) => Json(super::MessageResponse {
            ok: true,
            message: Some(format!(
                "Cleansed {removed} effect(s) matching '{filter}' from entity {entity_id}"
            )),
        }),
        None => Json(super::MessageResponse {
            ok: false,
            message: Some(format!("Entity {entity_id} not found")),
        }),
    }
}

/// Parse a modifier string "stat:op:value" into a StatModifier.
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

/// Parse a tick effect string like "dps:10" or "hps:5" or "custom:bleed".
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
