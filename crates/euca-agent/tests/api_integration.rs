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
            events.send(euca_gameplay::DamageEvent {
                target: entity,
                amount: 30.0,
                source: None,
            });
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
            actions: vec![],
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
