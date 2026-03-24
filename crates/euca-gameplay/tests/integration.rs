//! Integration tests — end-to-end gameplay scenarios run headlessly (no GPU).
//!
//! These tests verify that multiple gameplay systems compose correctly:
//! damage → death → scoring → respawn → game-over.

use euca_ecs::{Entity, Events, Query, World};
use euca_gameplay::{
    // v0.9.3/v0.9.4 systems
    BaseStats,
    DamageEvent,
    DamageResistance,
    Dead,
    DeathEvent,
    Equipment,
    GamePhase,
    GameState,
    Health,
    ItemDef,
    ItemRegistry,
    MatchConfig,
    ModifierOp,
    ResolvedStats,
    SpawnPoint,
    StackPolicy,
    StatModifier,
    StatModifiers,
    StatusEffect,
    StatusEffects,
    Team,
    ViewFilter,
    VisibilityRule,
    VisibleTo,
    Zone,
    ZoneEffect,
    ZoneShape,
    apply_damage_system,
    death_check_system,
    equipment_stat_system,
    game_state_system,
    respawn_system,
    start_respawn_on_death,
    stat_resolution_system,
    status_effect_tick_system,
    visibility_system,
    zone_system,
};
use euca_math::{Transform, Vec3};
use euca_scene::{GlobalTransform, LocalTransform};

/// Helper: create a world with Events resource.
fn test_world() -> World {
    let mut world = World::new();
    world.insert_resource(Events::default());
    world
}

/// Helper: spawn a fighter entity with Health, Team, and position.
fn spawn_fighter(world: &mut World, hp: f32, team: u8, pos: Vec3) -> Entity {
    let entity = world.spawn(Health::new(hp));
    world.insert(entity, Team(team));
    world.insert(entity, LocalTransform(Transform::from_translation(pos)));
    world.insert(entity, GlobalTransform::default());
    entity
}

/// Helper: send a damage event.
fn deal_damage(world: &mut World, target: Entity, amount: f32, source: Option<Entity>) {
    world
        .resource_mut::<Events>()
        .unwrap()
        .send(DamageEvent::new(target, amount, source));
}

/// Helper: advance one gameplay tick (damage → death → scoring → respawn).
fn step_gameplay(world: &mut World, dt: f32) {
    let respawn_delay = world
        .resource::<GameState>()
        .map(|gs| gs.config.respawn_delay)
        .unwrap_or(3.0);
    apply_damage_system(world);
    death_check_system(world);
    start_respawn_on_death(world, respawn_delay);
    game_state_system(world, dt);
    respawn_system(world, dt);

    if let Some(events) = world.resource_mut::<Events>() {
        events.update();
    }
    world.tick();
}

// ─── End-to-end scenarios ────────────────────────────────────────────

#[test]
fn full_kill_scores_point() {
    let mut world = test_world();
    world.insert_resource(GameState::new(MatchConfig {
        score_limit: 3,
        ..Default::default()
    }));

    let attacker = spawn_fighter(&mut world, 100.0, 1, Vec3::ZERO);
    let victim = spawn_fighter(&mut world, 50.0, 2, Vec3::new(5.0, 0.0, 0.0));

    // Deal lethal damage
    deal_damage(&mut world, victim, 50.0, Some(attacker));
    step_gameplay(&mut world, 0.016);

    // Victim should be dead
    assert!(world.get::<Dead>(victim).is_some());
    assert_eq!(world.get::<Health>(victim).unwrap().current, 0.0);

    // Attacker should have scored
    let state = world.resource::<GameState>().unwrap();
    assert_eq!(*state.scores.get(&attacker.index()).unwrap_or(&0), 1);
}

#[test]
fn game_ends_at_score_limit() {
    let mut world = test_world();
    let config = MatchConfig {
        score_limit: 2,
        time_limit: 0.0, // no time limit
        ..Default::default()
    };
    let mut state = GameState::new(config);
    state.start(); // skip to Playing phase
    world.insert_resource(state);

    let attacker = spawn_fighter(&mut world, 100.0, 1, Vec3::ZERO);

    // Kill two victims
    for i in 0..2 {
        let victim = spawn_fighter(&mut world, 10.0, 2, Vec3::new(10.0 + i as f32, 0.0, 0.0));
        deal_damage(&mut world, victim, 10.0, Some(attacker));
        step_gameplay(&mut world, 0.016);
    }

    let state = world.resource::<GameState>().unwrap();
    match &state.phase {
        GamePhase::PostMatch { winner } => {
            assert!(winner.is_some());
        }
        other => panic!("Expected PostMatch, got {:?}", other),
    }
}

#[test]
fn time_limit_ends_game() {
    let mut world = test_world();
    let config = MatchConfig {
        score_limit: 999, // unreachable
        time_limit: 10.0,
        ..Default::default()
    };
    let mut state = GameState::new(config);
    state.start();
    world.insert_resource(state);

    // Step past time limit
    for _ in 0..700 {
        step_gameplay(&mut world, 0.016);
    }

    let state = world.resource::<GameState>().unwrap();
    assert!(
        matches!(state.phase, GamePhase::PostMatch { .. }),
        "Expected PostMatch after time limit, got {:?}",
        state.phase
    );
}

#[test]
fn respawn_revives_dead_entity() {
    let mut world = test_world();
    world.insert_resource(GameState::new(MatchConfig {
        respawn_delay: 0.5,
        ..Default::default()
    }));

    // Add spawn point (position comes from LocalTransform)
    let spawn = world.spawn(SpawnPoint { team: 1 });
    world.insert(
        spawn,
        LocalTransform(Transform::from_translation(Vec3::new(0.0, 0.0, 0.0))),
    );

    let fighter = spawn_fighter(&mut world, 50.0, 1, Vec3::new(10.0, 0.0, 0.0));

    // Kill the fighter
    deal_damage(&mut world, fighter, 50.0, None);
    step_gameplay(&mut world, 0.016);
    assert!(world.get::<Dead>(fighter).is_some());

    // Step through respawn delay (0.5 seconds at 60fps = ~31 frames)
    for _ in 0..40 {
        step_gameplay(&mut world, 0.016);
    }

    // Should be alive again
    assert!(
        world.get::<Dead>(fighter).is_none(),
        "Fighter should have respawned"
    );
    let health = world.get::<Health>(fighter).unwrap();
    assert_eq!(
        health.current, health.max,
        "Health should be full after respawn"
    );
}

#[test]
fn multiple_damage_events_stack() {
    let mut world = test_world();
    world.insert_resource(GameState::new(MatchConfig::default()));

    let target = spawn_fighter(&mut world, 100.0, 1, Vec3::ZERO);

    // Multiple damage sources in one tick
    deal_damage(&mut world, target, 20.0, None);
    deal_damage(&mut world, target, 30.0, None);
    deal_damage(&mut world, target, 15.0, None);

    apply_damage_system(&mut world);

    let health = world.get::<Health>(target).unwrap();
    assert_eq!(health.current, 35.0, "100 - 20 - 30 - 15 = 35");
}

#[test]
fn overkill_damage_does_not_go_negative() {
    let mut world = test_world();
    world.insert_resource(GameState::new(MatchConfig::default()));

    let target = spawn_fighter(&mut world, 50.0, 1, Vec3::ZERO);

    deal_damage(&mut world, target, 100.0, None);
    deal_damage(&mut world, target, 200.0, None);

    apply_damage_system(&mut world);

    let health = world.get::<Health>(target).unwrap();
    assert_eq!(health.current, 0.0, "Health should clamp at 0");
}

#[test]
fn death_event_emitted_on_kill() {
    let mut world = test_world();

    let attacker = spawn_fighter(&mut world, 100.0, 1, Vec3::ZERO);
    let victim = spawn_fighter(&mut world, 10.0, 2, Vec3::new(5.0, 0.0, 0.0));

    deal_damage(&mut world, victim, 10.0, Some(attacker));
    apply_damage_system(&mut world);
    death_check_system(&mut world);

    let death_events: Vec<DeathEvent> = world
        .resource::<Events>()
        .unwrap()
        .read::<DeathEvent>()
        .cloned()
        .collect();

    assert_eq!(death_events.len(), 1);
    assert_eq!(death_events[0].entity, victim);
    assert_eq!(death_events[0].killer, Some(attacker));
}

#[test]
fn dead_entity_not_re_marked_on_subsequent_ticks() {
    let mut world = test_world();
    world.insert_resource(GameState::new(MatchConfig::default()));

    let target = spawn_fighter(&mut world, 10.0, 1, Vec3::ZERO);

    deal_damage(&mut world, target, 10.0, None);
    step_gameplay(&mut world, 0.016);
    assert!(world.get::<Dead>(target).is_some());

    // Step again — should not panic or emit duplicate death events
    step_gameplay(&mut world, 0.016);
    step_gameplay(&mut world, 0.016);

    // Still dead, no crash
    assert!(world.get::<Dead>(target).is_some());
}

#[test]
fn countdown_transitions_through_phases() {
    let mut world = test_world();

    let mut state = GameState::new(MatchConfig::default());
    state.phase = GamePhase::Countdown { remaining: 0.5 };
    world.insert_resource(state);

    // Not yet playing
    let state = world.resource::<GameState>().unwrap();
    assert!(matches!(state.phase, GamePhase::Countdown { .. }));

    // Step past countdown
    for _ in 0..40 {
        game_state_system(&mut world, 0.016);
    }

    let state = world.resource::<GameState>().unwrap();
    assert_eq!(state.phase, GamePhase::Playing);
}

#[test]
fn ten_entity_battle_runs_stable() {
    let mut world = test_world();
    let mut state = GameState::new(MatchConfig {
        score_limit: 100,
        time_limit: 0.0,
        ..Default::default()
    });
    state.start();
    world.insert_resource(state);

    // Spawn 5 entities per team
    let mut team1 = Vec::new();
    let mut team2 = Vec::new();
    for i in 0..5 {
        team1.push(spawn_fighter(
            &mut world,
            100.0,
            1,
            Vec3::new(i as f32, 0.0, 0.0),
        ));
        team2.push(spawn_fighter(
            &mut world,
            100.0,
            2,
            Vec3::new(i as f32 + 10.0, 0.0, 0.0),
        ));
    }

    // Run 100 ticks with cross-team damage each tick
    for tick in 0..100 {
        // Each team1 member damages a team2 member and vice versa
        let t1 = team1[tick % 5];
        let t2 = team2[tick % 5];

        if world.get::<Dead>(t2).is_none() {
            deal_damage(&mut world, t2, 5.0, Some(t1));
        }
        if world.get::<Dead>(t1).is_none() {
            deal_damage(&mut world, t1, 3.0, Some(t2));
        }

        step_gameplay(&mut world, 0.016);
    }

    // Should not have panicked — stability is the test
    let alive_count = {
        let query = Query::<Entity>::new(&world);
        query
            .iter()
            .filter(|e| world.get::<Health>(*e).is_some() && world.get::<Dead>(*e).is_none())
            .count()
    };

    // Some entities should have died (5 damage per tick for 100 ticks on 100hp entities)
    assert!(alive_count < 10, "Some entities should have died");
}

const DT: f32 = 1.0 / 60.0;

#[test]
fn headless_moba_respawn_flow() {
    let mut world = test_world();
    let config = MatchConfig {
        respawn_delay: 3.0,
        score_limit: 100,
        ..Default::default()
    };
    let mut state = GameState::new(config);
    state.start();
    world.insert_resource(state);

    // Spawn points (like moba.sh)
    let sp1 = world.spawn(SpawnPoint { team: 1 });
    world.insert(
        sp1,
        LocalTransform(Transform::from_translation(Vec3::new(-7.0, 0.5, 0.0))),
    );
    let sp2 = world.spawn(SpawnPoint { team: 2 });
    world.insert(
        sp2,
        LocalTransform(Transform::from_translation(Vec3::new(7.0, 0.5, 0.0))),
    );

    // Hero team 2
    let hero = spawn_fighter(&mut world, 500.0, 2, Vec3::new(3.0, 0.5, 0.0));

    // Deal lethal damage
    deal_damage(&mut world, hero, 500.0, None);
    step_gameplay(&mut world, DT);

    assert!(world.get::<Dead>(hero).is_some(), "Hero should be dead");

    // Step through respawn delay (3.0s = 180 ticks at 60fps) + margin
    for _ in 0..250 {
        step_gameplay(&mut world, DT);
    }

    assert!(
        world.get::<Dead>(hero).is_none(),
        "Hero should have respawned after 250 ticks (~4.2s > 3.0s delay)"
    );
    assert_eq!(
        world.get::<Health>(hero).unwrap().current,
        500.0,
        "Health should be full after respawn"
    );
}

// ─── v0.9.3/v0.9.4 pipeline integration tests ─────────────────────────

#[test]
fn stat_pipeline_base_equipment_status_effects() {
    let mut world = test_world();

    // Register an item that adds +20 attack_damage.
    let mut registry = ItemRegistry::new();
    registry.register(ItemDef {
        id: 1,
        name: "Long Sword".into(),
        properties: [("attack_damage".into(), 20.0)].into_iter().collect(),
    });
    world.insert_resource(registry);

    // Entity with base stats.
    let entity = world.spawn(BaseStats(
        [("attack_damage".into(), 50.0), ("speed".into(), 100.0)]
            .into_iter()
            .collect(),
    ));

    // Equip the sword via Equipment component.
    world.insert(
        entity,
        Equipment {
            equipped: [("weapon".into(), 1)].into_iter().collect(),
        },
    );
    world.insert(entity, StatModifiers::default());

    // Apply a status effect: speed × 0.5 (slow).
    world.insert(
        entity,
        StatusEffects {
            effects: vec![StatusEffect {
                tag: "slow".into(),
                modifiers: vec![StatModifier {
                    stat: "speed".into(),
                    op: ModifierOp::Multiply,
                    value: 0.5,
                }],
                duration: 10.0,
                remaining: 10.0,
                source: None,
                stack_policy: StackPolicy::Replace,
                tick_effect: None,
            }],
        },
    );

    // Run the full stat pipeline.
    equipment_stat_system(&mut world);
    stat_resolution_system(&mut world);

    let resolved = world.get::<ResolvedStats>(entity).unwrap();
    assert_eq!(
        resolved.0.get("attack_damage"),
        Some(&70.0),
        "base 50 + equipment 20 = 70"
    );
    assert_eq!(
        resolved.0.get("speed"),
        Some(&50.0),
        "base 100 * 0.5 slow = 50"
    );
}

#[test]
fn visibility_observer_with_radius_filter() {
    let mut world = World::new();

    // Observer at origin with WithinRadius(10) filter.
    let observer = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
    world.insert(
        observer,
        ViewFilter::new(vec![VisibilityRule::WithinRadius { radius: 10.0 }]),
    );

    // Near entity at distance 5 — should be visible.
    let near = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
        5.0, 0.0, 0.0,
    ))));

    // Far entity at distance 20 — should NOT be visible.
    let far = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
        20.0, 0.0, 0.0,
    ))));

    visibility_system(&mut world);

    // Near entity should have VisibleTo containing the observer.
    let near_vt = world
        .get::<VisibleTo>(near)
        .expect("near should have VisibleTo");
    assert!(
        near_vt.0.contains(&observer),
        "near entity should be visible to observer"
    );

    // Far entity should NOT be visible to the observer.
    let far_visible = world
        .get::<VisibleTo>(far)
        .map(|vt| vt.0.contains(&observer))
        .unwrap_or(false);
    assert!(!far_visible, "far entity should not be visible to observer");
}

#[test]
fn zone_status_effect_stat_compose() {
    let mut world = test_world();

    // Create a zone that applies a "slow" status effect (speed × 0.5).
    let zone_entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
    world.insert(
        zone_entity,
        Zone::new(
            ZoneShape::Circle { radius: 10.0 },
            vec![ZoneEffect::ApplyStatusEffect {
                tag: "slow".into(),
                modifiers: vec![("speed".into(), ModifierOp::Multiply, 0.5)],
                duration: 5.0,
            }],
        ),
    );

    // Place entity inside the zone with base stats.
    let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
        3.0, 0.0, 0.0,
    ))));
    world.insert(
        entity,
        BaseStats([("speed".into(), 100.0)].into_iter().collect()),
    );

    let dt = 1.0 / 60.0;

    // Pipeline: zone → status effect tick → stat resolution.
    zone_system(&mut world, dt);
    status_effect_tick_system(&mut world, dt);
    stat_resolution_system(&mut world);

    // The entity should have received the "slow" status effect from the zone.
    let effects = world
        .get::<StatusEffects>(entity)
        .expect("entity should have StatusEffects after entering zone");
    assert!(
        effects.effects.iter().any(|e| e.tag == "slow"),
        "entity should have the 'slow' effect"
    );

    // ResolvedStats should show speed halved.
    let resolved = world
        .get::<ResolvedStats>(entity)
        .expect("entity should have ResolvedStats after resolution");
    assert_eq!(
        resolved.0.get("speed"),
        Some(&50.0),
        "base 100 * 0.5 slow = 50"
    );
}

#[test]
fn damage_resistance_with_category() {
    let mut world = test_world();

    let target = world.spawn(Health::new(100.0));
    // 50 physical resistance.
    world.insert(
        target,
        DamageResistance([("physical".into(), 50.0_f64)].into_iter().collect()),
    );

    // Deal 100 physical damage. Effective = 100 * (100 / 150) ≈ 66.67.
    // Health: 100 - 66.67 ≈ 33.33.
    world
        .resource_mut::<Events>()
        .unwrap()
        .send(DamageEvent::new(target, 100.0, None));
    apply_damage_system(&mut world);

    let health_after_physical = world.get::<Health>(target).unwrap().current;
    let expected_remaining = 100.0 - 100.0 * (100.0_f32 / 150.0);
    assert!(
        (health_after_physical - expected_remaining).abs() < 0.01,
        "After physical damage with 50 resistance: expected ~{expected_remaining}, got {health_after_physical}"
    );

    // Clear events for next tick, then deal 50 "true" damage (bypasses resistance).
    world.resource_mut::<Events>().unwrap().update();
    world.tick();
    world
        .resource_mut::<Events>()
        .unwrap()
        .send(DamageEvent::with_category(target, 50.0, None, "true"));
    apply_damage_system(&mut world);

    let health_after_true = world.get::<Health>(target).unwrap().current;
    // ~33.33 - 50.0, clamped to 0.
    assert_eq!(
        health_after_true, 0.0,
        "True damage should bypass resistance and clamp health to 0"
    );
}
