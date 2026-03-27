//! Full-engine headless benchmarks for the Euca Engine.
//!
//! Measures per-tick performance of physics + transform propagation at
//! realistic entity counts (1K, 10K) and with gameplay systems (Health, damage).
//! All data is deterministic (grid-based positions, fixed seeds) so results
//! are reproducible across runs.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use euca_core::Time;
use euca_ecs::{Events, World};
use euca_gameplay::{DamageEvent, Health, apply_damage_system, death_check_system};
use euca_math::{Transform, Vec3};
use euca_physics::{Collider, PhysicsBody, PhysicsConfig, Velocity, physics_step_system};
use euca_scene::{GlobalTransform, LocalTransform, transform_propagation_system};

// ---------------------------------------------------------------------------
// World construction helpers
// ---------------------------------------------------------------------------

/// Spawn `count` dynamic physics entities on a deterministic grid.
///
/// Entities are placed on a square grid in the XZ plane at y = 0.5.
/// Each entity gets: LocalTransform, GlobalTransform, PhysicsBody::dynamic(),
/// Velocity (small deterministic linear velocity), and Collider::aabb().
fn spawn_physics_grid(world: &mut World, count: usize) {
    let side = (count as f32).sqrt().ceil() as usize;
    let spacing = 2.0_f32;

    for i in 0..count {
        let row = i / side;
        let col = i % side;
        let x = col as f32 * spacing;
        let z = row as f32 * spacing;

        let pos = Vec3::new(x, 0.5, z);
        let entity = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(entity, GlobalTransform::default());
        world.insert(entity, PhysicsBody::dynamic());
        // Small deterministic velocity so physics actually integrates motion.
        world.insert(
            entity,
            Velocity {
                linear: Vec3::new(0.1, 0.0, -0.05),
                angular: Vec3::ZERO,
            },
        );
        world.insert(entity, Collider::aabb(0.5, 0.5, 0.5));
    }
}

/// Build a minimal headless world with the resources needed by physics and
/// transform propagation systems.
fn build_headless_world() -> World {
    let mut world = World::new();
    world.insert_resource(Time::new());
    world.insert_resource(PhysicsConfig::new());
    world.insert_resource(Events::default());
    world
}

/// Run one "tick": physics step, transform propagation, world tick.
fn tick_physics(world: &mut World) {
    physics_step_system(world);
    transform_propagation_system(world);
    world.tick();
}

/// Run one "tick" with gameplay: physics, damage, death, transform propagation.
fn tick_gameplay(world: &mut World) {
    physics_step_system(world);
    apply_damage_system(world);
    death_check_system(world);
    transform_propagation_system(world);

    if let Some(events) = world.resource_mut::<Events>() {
        events.update();
    }
    world.tick();
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// Benchmark A: 1 000 entities with physics + transform propagation.
fn headless_tick_1k(c: &mut Criterion) {
    let mut world = build_headless_world();
    spawn_physics_grid(&mut world, 1_000);
    // Warm up: run one tick so caches and lazy init are settled.
    tick_physics(&mut world);

    c.bench_function("headless_tick_1k", |b| {
        b.iter(|| {
            tick_physics(black_box(&mut world));
        });
    });
}

/// Benchmark B: 10 000 entities with physics + transform propagation.
fn headless_tick_10k(c: &mut Criterion) {
    let mut world = build_headless_world();
    spawn_physics_grid(&mut world, 10_000);
    tick_physics(&mut world);

    c.bench_function("headless_tick_10k", |b| {
        b.iter(|| {
            tick_physics(black_box(&mut world));
        });
    });
}

/// Benchmark C: 1 000 entities with physics + gameplay (Health, damage).
///
/// Each tick sends a small damage event to the first 100 entities, exercises
/// the damage pipeline and death checking in addition to physics.
fn headless_tick_1k_gameplay(c: &mut Criterion) {
    let mut world = build_headless_world();
    spawn_physics_grid(&mut world, 1_000);

    // Attach Health to every entity so gameplay systems have work to do.
    // Collect entity IDs first — entities with PhysicsBody are our targets.
    let entities: Vec<euca_ecs::Entity> = {
        let q = euca_ecs::Query::<(euca_ecs::Entity, &PhysicsBody)>::new(&world);
        q.iter().map(|(e, _)| e).collect()
    };
    for &entity in &entities {
        world.insert(entity, Health::new(1000.0));
    }

    // Keep the first 100 entity IDs for per-tick damage injection.
    let damage_targets: Vec<euca_ecs::Entity> = entities.iter().copied().take(100).collect();

    // Warm up
    tick_gameplay(&mut world);

    c.bench_function("headless_tick_1k_gameplay", |b| {
        b.iter(|| {
            // Inject deterministic damage events for the first 100 entities.
            if let Some(events) = world.resource_mut::<Events>() {
                for &target in &damage_targets {
                    events.send(DamageEvent::new(target, 1.0, None));
                }
            }
            tick_gameplay(black_box(&mut world));
        });
    });
}

/// Benchmark D: 50 000 entities with physics + transform propagation.
///
/// This is the primary scaling target for the engine. Uses reduced sample
/// size since each iteration takes ~200ms+.
fn headless_tick_50k(c: &mut Criterion) {
    let mut group = c.benchmark_group("headless_tick_50k");
    group.sample_size(10);

    let mut world = build_headless_world();
    spawn_physics_grid(&mut world, 50_000);
    tick_physics(&mut world);

    group.bench_function("50000", |b| {
        b.iter(|| {
            tick_physics(black_box(&mut world));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    headless_tick_1k,
    headless_tick_10k,
    headless_tick_1k_gameplay,
    headless_tick_50k,
);
criterion_main!(benches);
