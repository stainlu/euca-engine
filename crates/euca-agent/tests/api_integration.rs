//! Integration tests: verify route handlers work end-to-end via SharedWorld.
//!
//! Instead of spinning up an HTTP server, these tests call the axum handlers
//! directly through the SharedWorld, testing the complete path from request
//! to world state change.

use euca_agent::EngineControl;
use euca_agent::routes::TemplateRegistry;
use euca_ecs::{Events, Schedule, SharedWorld, World};

/// Create a shared world with all resources initialized.
fn test_world() -> SharedWorld {
    let world = World::new();
    let schedule = Schedule::new();
    let shared = SharedWorld::new(world, schedule);

    shared.with(|w, _| {
        w.insert_resource(Events::default());
        w.insert_resource(EngineControl::new());
        w.insert_resource(euca_agent::hud::HudCanvas::default());
        w.insert_resource(TemplateRegistry::new());
    });

    shared
}

#[test]
fn test_spawn_and_query_entity() {
    let shared = test_world();

    // Spawn an entity
    let (id, generation) = shared.with(|w, _| {
        let pos = [1.0f32, 2.0, 3.0];
        let transform =
            euca_math::Transform::from_translation(euca_math::Vec3::new(pos[0], pos[1], pos[2]));
        let entity = w.spawn(euca_scene::LocalTransform(transform));
        w.insert(entity, euca_scene::GlobalTransform::default());
        w.insert(entity, euca_gameplay::Health::new(100.0));
        w.insert(entity, euca_gameplay::Team(1));
        (entity.index(), entity.generation())
    });

    // Query it back
    let health = shared.with_world(|w| {
        let entity = euca_ecs::Entity::from_raw(id, generation);
        w.get::<euca_gameplay::Health>(entity)
            .map(|h| [h.current, h.max])
    });
    assert_eq!(health, Some([100.0, 100.0]));

    let team = shared.with_world(|w| {
        let entity = euca_ecs::Entity::from_raw(id, generation);
        w.get::<euca_gameplay::Team>(entity).map(|t| t.0)
    });
    assert_eq!(team, Some(1));
}

#[test]
fn test_despawn_entity() {
    let shared = test_world();

    let (id, generation) = shared.with(|w, _| {
        let entity = w.spawn(euca_scene::LocalTransform(euca_math::Transform::IDENTITY));
        (entity.index(), entity.generation())
    });

    // Despawn
    let ok = shared.with(|w, _| {
        let entity = euca_ecs::Entity::from_raw(id, generation);
        w.despawn(entity)
    });
    assert!(ok);

    // Verify gone
    let alive = shared.with_world(|w| {
        let entity = euca_ecs::Entity::from_raw(id, generation);
        w.is_alive(entity)
    });
    assert!(!alive);
}

#[test]
fn test_simulation_step() {
    let shared = test_world();

    let tick_before = shared.with_world(|w| w.current_tick());

    shared.with(|w, schedule| {
        for _ in 0..5 {
            schedule.run(w);
        }
    });

    let tick_after = shared.with_world(|w| w.current_tick());
    assert!(
        tick_after > tick_before,
        "Tick should advance after schedule.run()"
    );
}

#[test]
fn test_game_state() {
    let shared = test_world();

    // Create game
    shared.with(|w, _| {
        let config = euca_gameplay::MatchConfig {
            mode: "deathmatch".into(),
            score_limit: 10,
            time_limit: 300.0,
            respawn_delay: 3.0,
        };
        let mut state = euca_gameplay::GameState::new(config);
        state.start();
        w.insert_resource(state);
    });

    // Check state
    let phase = shared.with_world(|w| {
        w.resource::<euca_gameplay::GameState>()
            .map(|s| matches!(s.phase, euca_gameplay::GamePhase::Playing))
    });
    assert_eq!(phase, Some(true));
}

#[test]
fn test_template_create_and_spawn() {
    let shared = test_world();

    // Create template
    shared.with(|w, _| {
        let registry = w.resource_mut::<TemplateRegistry>().unwrap();
        registry.templates.insert(
            "soldier".into(),
            euca_agent::routes::SpawnRequest {
                agent_id: None,
                mesh: None,
                color: None,
                position: None,
                scale: None,
                velocity: None,
                collider: None,
                physics_body: None,
                health: Some(100.0),
                team: Some(1),
                combat: Some(true),
                combat_damage: None,
                combat_range: None,
                combat_speed: None,
                combat_cooldown: None,
                combat_style: None,
                ai_patrol: None,
                gold: None,
                gold_bounty: None,
                xp_bounty: None,
                role: None,
                spawn_point: None,
                player: None,
                building_type: None,
                lane: None,
            },
        );
    });

    // Verify template exists
    let has_template = shared.with_world(|w| {
        w.resource::<TemplateRegistry>()
            .map(|r| r.templates.contains_key("soldier"))
    });
    assert_eq!(has_template, Some(true));
}

#[test]
fn test_hud_add_and_clear() {
    let shared = test_world();

    // Add HUD element
    shared.with(|w, _| {
        let canvas = w.resource_mut::<euca_agent::hud::HudCanvas>().unwrap();
        canvas.add(euca_agent::hud::HudElement::Text {
            text: "Test".into(),
            x: 0.5,
            y: 0.1,
            size: 24.0,
            color: "white".into(),
        });
    });

    let count = shared.with_world(|w| {
        w.resource::<euca_agent::hud::HudCanvas>()
            .map(|c| c.elements.len())
    });
    assert_eq!(count, Some(1));

    // Clear
    shared.with(|w, _| {
        let canvas = w.resource_mut::<euca_agent::hud::HudCanvas>().unwrap();
        canvas.clear();
    });

    let count = shared.with_world(|w| {
        w.resource::<euca_agent::hud::HudCanvas>()
            .map(|c| c.elements.len())
    });
    assert_eq!(count, Some(0));
}

#[test]
fn test_damage_event_flow() {
    let shared = test_world();

    // Spawn entity with health
    let entity_id = shared.with(|w, _| {
        let entity = w.spawn(euca_gameplay::Health::new(100.0));
        entity.index()
    });

    // Send damage event
    shared.with(|w, _| {
        let entity = euca_ecs::Entity::from_raw(entity_id, 0);
        if let Some(events) = w.resource_mut::<Events>() {
            events.send(euca_gameplay::DamageEvent::new(entity, 30.0, None));
        }
    });

    // Apply damage system
    shared.with(|w, _| {
        euca_gameplay::apply_damage_system(w);
    });

    // Check health decreased
    let health = shared.with_world(|w| {
        let entity = euca_ecs::Entity::from_raw(entity_id, 0);
        w.get::<euca_gameplay::Health>(entity).map(|h| h.current)
    });
    assert_eq!(health, Some(70.0));
}

#[test]
fn test_rule_creation() {
    let shared = test_world();

    // Create a timer rule
    shared.with(|w, _| {
        let entity = w.spawn(euca_gameplay::TimerRule {
            interval: 5.0,
            elapsed: 0.0,
            repeat: true,
            actions: std::sync::Arc::new(vec![]),
        });
        assert!(w.is_alive(entity));
    });

    // Verify rule exists
    let count = shared.with_world(|w| {
        let query = euca_ecs::Query::<&euca_gameplay::TimerRule>::new(w);
        query.iter().count()
    });
    assert_eq!(count, 1);
}

// ── Fork tests: counterfactual simulation ──────────────────────────────────

#[test]
fn test_fork_main_state_independence() {
    let shared = test_world();

    // Set up main world state: one entity with 100 health.
    let entity_id = shared.with(|w, _| {
        let e = w.spawn(euca_gameplay::Health::new(100.0));
        w.insert(e, euca_gameplay::Team(1));
        e.index()
    });

    // Fork main into "scenario-a".
    shared.fork("scenario-a").unwrap();

    // Mutate the fork: lower health to 1.
    shared
        .with_fork("scenario-a", |w, _| {
            let e = euca_ecs::Entity::from_raw(entity_id, 0);
            w.get_mut::<euca_gameplay::Health>(e).unwrap().current = 1.0;
        })
        .unwrap();

    // Main is unchanged.
    let main_health = shared.with_world(|w| {
        let e = euca_ecs::Entity::from_raw(entity_id, 0);
        w.get::<euca_gameplay::Health>(e).map(|h| h.current)
    });
    assert_eq!(main_health, Some(100.0));

    // Fork has its own state.
    let fork_health = shared
        .with_fork("scenario-a", |w, _| {
            let e = euca_ecs::Entity::from_raw(entity_id, 0);
            w.get::<euca_gameplay::Health>(e).map(|h| h.current)
        })
        .unwrap();
    assert_eq!(fork_health, Some(1.0));
}

#[test]
fn test_fork_step_advances_only_fork_tick() {
    let shared = test_world();

    // Both main and fork start at tick 0.
    shared.fork("scenario-b").unwrap();
    let main_tick_before = shared.with_world(|w| w.current_tick());
    let fork_tick_before = shared
        .with_fork_ref("scenario-b", |w| w.current_tick())
        .unwrap();
    assert_eq!(main_tick_before, fork_tick_before);

    // Step the fork by 5 ticks. `Schedule::run` advances `world.tick()`
    // internally, so calling it 5 times advances the tick by 5.
    shared
        .with_fork("scenario-b", |w, schedule| {
            for _ in 0..5 {
                schedule.run(w);
            }
        })
        .unwrap();

    // Main tick is unchanged.
    assert_eq!(shared.with_world(|w| w.current_tick()), main_tick_before);

    // Fork tick advanced by 5.
    let fork_tick_after = shared
        .with_fork_ref("scenario-b", |w| w.current_tick())
        .unwrap();
    assert_eq!(fork_tick_after, fork_tick_before + 5);
}

#[test]
fn test_fork_spawn_does_not_affect_main() {
    let shared = test_world();

    // Main world is empty.
    let main_count_before = shared.with_world(|w| w.entity_count());

    // Create a fork and spawn 3 entities in it.
    shared.fork("spawn-scenario").unwrap();
    shared
        .with_fork("spawn-scenario", |w, _| {
            for _ in 0..3 {
                w.spawn(euca_gameplay::Health::new(50.0));
            }
        })
        .unwrap();

    // Main has no new entities.
    assert_eq!(shared.with_world(|w| w.entity_count()), main_count_before);

    // Fork has 3.
    let fork_count = shared
        .with_fork_ref("spawn-scenario", |w| w.entity_count())
        .unwrap();
    assert_eq!(fork_count, main_count_before + 3);
}

#[test]
fn test_fork_list_and_delete() {
    let shared = test_world();

    shared.fork("a").unwrap();
    shared.fork("b").unwrap();
    shared.fork("c").unwrap();

    let mut all = shared.list_forks();
    all.sort();
    assert_eq!(all, vec!["a".to_string(), "b".to_string(), "c".to_string()]);

    assert!(shared.fork_exists("b"));
    assert!(shared.delete_fork("b"));
    assert!(!shared.fork_exists("b"));

    let mut remaining = shared.list_forks();
    remaining.sort();
    assert_eq!(remaining, vec!["a".to_string(), "c".to_string()]);
}

#[test]
fn test_fork_duplicate_id_rejected() {
    let shared = test_world();
    shared.fork("only-one").unwrap();
    let err = shared.fork("only-one").unwrap_err();
    assert!(matches!(err, euca_ecs::ForkError::AlreadyExists(_)));
}

#[test]
fn test_fork_with_main_carries_resources() {
    let shared = test_world();

    // Set up a game state on main.
    shared.with(|w, _| {
        let config = euca_gameplay::MatchConfig {
            mode: "deathmatch".into(),
            score_limit: 5,
            time_limit: 60.0,
            respawn_delay: 2.0,
        };
        let mut state = euca_gameplay::GameState::new(config);
        state.start();
        w.insert_resource(state);
    });

    // Fork should carry the GameState resource.
    shared.fork("resource-test").unwrap();

    let fork_phase = shared
        .with_fork("resource-test", |w, _| {
            w.resource::<euca_gameplay::GameState>()
                .map(|s| matches!(s.phase, euca_gameplay::GamePhase::Playing))
        })
        .unwrap();
    assert_eq!(fork_phase, Some(true));

    // Mutating the fork's game state does not affect main.
    shared
        .with_fork("resource-test", |w, _| {
            w.resource_mut::<euca_gameplay::GameState>().unwrap().phase =
                euca_gameplay::GamePhase::PostMatch { winner: None };
        })
        .unwrap();

    let main_phase = shared.with_world(|w| {
        w.resource::<euca_gameplay::GameState>()
            .map(|s| matches!(s.phase, euca_gameplay::GamePhase::Playing))
    });
    assert_eq!(main_phase, Some(true));
}

#[test]
fn test_fork_damage_event_stays_in_fork() {
    let shared = test_world();

    // Spawn an entity with health on main.
    let entity_id = shared.with(|w, _| {
        let e = w.spawn(euca_gameplay::Health::new(100.0));
        e.index()
    });

    shared.fork("dmg-scenario").unwrap();

    // Send a damage event on the fork only, then run the damage system.
    shared
        .with_fork("dmg-scenario", |w, _| {
            let e = euca_ecs::Entity::from_raw(entity_id, 0);
            if let Some(events) = w.resource_mut::<Events>() {
                events.send(euca_gameplay::DamageEvent::new(e, 30.0, None));
            }
            euca_gameplay::apply_damage_system(w);
        })
        .unwrap();

    // Main health is unchanged.
    let main_hp = shared.with_world(|w| {
        let e = euca_ecs::Entity::from_raw(entity_id, 0);
        w.get::<euca_gameplay::Health>(e).map(|h| h.current)
    });
    assert_eq!(main_hp, Some(100.0));

    // Fork health is 70.
    let fork_hp = shared
        .with_fork("dmg-scenario", |w, _| {
            let e = euca_ecs::Entity::from_raw(entity_id, 0);
            w.get::<euca_gameplay::Health>(e).map(|h| h.current)
        })
        .unwrap();
    assert_eq!(fork_hp, Some(70.0));
}
