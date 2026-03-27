#![allow(dead_code)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use euca_ecs::{Query, Schedule, World};

// ── Components ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy)]
struct Velocity {
    dx: f32,
    dy: f32,
    dz: f32,
}

#[derive(Clone, Copy)]
struct Health {
    current: f32,
    max: f32,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Spawn `n` entities, each with Position + Velocity + Health.
fn spawn_three_component_entities(world: &mut World, n: u32) {
    for i in 0..n {
        let e = world.spawn(Position {
            x: i as f32,
            y: 0.0,
            z: 0.0,
        });
        world.insert(
            e,
            Velocity {
                dx: 1.0,
                dy: 0.0,
                dz: 0.0,
            },
        );
        world.insert(
            e,
            Health {
                current: 100.0,
                max: 100.0,
            },
        );
    }
}

// ── Spawn benchmarks ────────────────────────────────────────────────────────

fn bench_spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn");
    for count in [1_000u32, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{count}")),
            &count,
            |b, &n| {
                b.iter(|| {
                    let mut world = World::new();
                    spawn_three_component_entities(&mut world, n);
                    black_box(&world);
                });
            },
        );
    }
    group.finish();
}

// ── Query iteration benchmarks ──────────────────────────────────────────────

fn bench_query_iterate(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_iterate");
    for count in [1_000u32, 10_000] {
        let mut world = World::new();
        spawn_three_component_entities(&mut world, count);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{count}")),
            &count,
            |b, &n| {
                b.iter(|| {
                    let query = Query::<(&Position, &Velocity, &Health)>::new(&world);
                    let mut total = 0u32;
                    for (pos, vel, hp) in query.iter() {
                        black_box((pos, vel, hp));
                        total += 1;
                    }
                    assert_eq!(total, n);
                });
            },
        );
    }
    group.finish();
}

// ── Cached vs uncached query ────────────────────────────────────────────────

fn bench_query_cached_vs_uncached(c: &mut Criterion) {
    let mut world = World::new();
    spawn_three_component_entities(&mut world, 10_000);

    let mut group = c.benchmark_group("query_cached_vs_uncached");

    group.bench_function("uncached", |b| {
        b.iter(|| {
            let query = Query::<(&Position, &Velocity)>::new(&world);
            let mut total = 0u32;
            for (pos, vel) in query.iter() {
                black_box((pos, vel));
                total += 1;
            }
            assert_eq!(total, 10_000);
        });
    });

    group.bench_function("cached", |b| {
        b.iter(|| {
            let query = Query::<(&Position, &Velocity)>::new_cached(&world);
            let mut total = 0u32;
            for (pos, vel) in query.iter() {
                black_box((pos, vel));
                total += 1;
            }
            assert_eq!(total, 10_000);
        });
    });

    group.finish();
}

// ── Change detection overhead ───────────────────────────────────────────────
//
// Compares iterating with `&T` (read-only, no change tick writes) vs `&mut T`
// (writes a change tick per entity on each access).

fn bench_change_detection_overhead(c: &mut Criterion) {
    let mut world = World::new();
    spawn_three_component_entities(&mut world, 10_000);

    let mut group = c.benchmark_group("change_detection_overhead");

    group.bench_function("without_change_detection", |b| {
        b.iter(|| {
            let query = Query::<(&Position, &Velocity)>::new(&world);
            for (pos, vel) in query.iter() {
                black_box((pos, vel));
            }
        });
    });

    group.bench_function("with_change_detection", |b| {
        b.iter(|| {
            let query = Query::<(&mut Position, &Velocity)>::new(&world);
            for (pos, vel) in query.iter() {
                black_box((&*pos, vel));
            }
        });
    });

    group.finish();
}

// ── Archetype fragmentation ─────────────────────────────────────────────────
//
// All benchmarks iterate the same total number of entities (10 000) via a
// Query<&Position>. The difference is how many distinct archetypes those
// entities are spread across: 1, 10, or 100.
//
// We create archetype diversity by attaching a unique "tag" component type to
// each group. Because every distinct set of component types maps to its own
// archetype, N tag types produces N archetypes that all contain Position.

// Macro to generate unique zero-sized tag types.
macro_rules! define_tags {
    ($($name:ident),+ $(,)?) => { $( #[derive(Clone, Copy)] struct $name; )+ };
}

// 100 unique tags (T00 .. T99).
define_tags!(
    T00, T01, T02, T03, T04, T05, T06, T07, T08, T09, T10, T11, T12, T13, T14, T15, T16, T17, T18,
    T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37,
    T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56,
    T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75,
    T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94,
    T95, T96, T97, T98, T99,
);

/// Insert a tag component on entity `e` based on the archetype bucket index.
/// Each tag type produces a distinct archetype.
fn insert_tag_by_index(world: &mut World, e: euca_ecs::Entity, idx: usize) {
    match idx {
        0 => {
            world.insert(e, T00);
        }
        1 => {
            world.insert(e, T01);
        }
        2 => {
            world.insert(e, T02);
        }
        3 => {
            world.insert(e, T03);
        }
        4 => {
            world.insert(e, T04);
        }
        5 => {
            world.insert(e, T05);
        }
        6 => {
            world.insert(e, T06);
        }
        7 => {
            world.insert(e, T07);
        }
        8 => {
            world.insert(e, T08);
        }
        9 => {
            world.insert(e, T09);
        }
        10 => {
            world.insert(e, T10);
        }
        11 => {
            world.insert(e, T11);
        }
        12 => {
            world.insert(e, T12);
        }
        13 => {
            world.insert(e, T13);
        }
        14 => {
            world.insert(e, T14);
        }
        15 => {
            world.insert(e, T15);
        }
        16 => {
            world.insert(e, T16);
        }
        17 => {
            world.insert(e, T17);
        }
        18 => {
            world.insert(e, T18);
        }
        19 => {
            world.insert(e, T19);
        }
        20 => {
            world.insert(e, T20);
        }
        21 => {
            world.insert(e, T21);
        }
        22 => {
            world.insert(e, T22);
        }
        23 => {
            world.insert(e, T23);
        }
        24 => {
            world.insert(e, T24);
        }
        25 => {
            world.insert(e, T25);
        }
        26 => {
            world.insert(e, T26);
        }
        27 => {
            world.insert(e, T27);
        }
        28 => {
            world.insert(e, T28);
        }
        29 => {
            world.insert(e, T29);
        }
        30 => {
            world.insert(e, T30);
        }
        31 => {
            world.insert(e, T31);
        }
        32 => {
            world.insert(e, T32);
        }
        33 => {
            world.insert(e, T33);
        }
        34 => {
            world.insert(e, T34);
        }
        35 => {
            world.insert(e, T35);
        }
        36 => {
            world.insert(e, T36);
        }
        37 => {
            world.insert(e, T37);
        }
        38 => {
            world.insert(e, T38);
        }
        39 => {
            world.insert(e, T39);
        }
        40 => {
            world.insert(e, T40);
        }
        41 => {
            world.insert(e, T41);
        }
        42 => {
            world.insert(e, T42);
        }
        43 => {
            world.insert(e, T43);
        }
        44 => {
            world.insert(e, T44);
        }
        45 => {
            world.insert(e, T45);
        }
        46 => {
            world.insert(e, T46);
        }
        47 => {
            world.insert(e, T47);
        }
        48 => {
            world.insert(e, T48);
        }
        49 => {
            world.insert(e, T49);
        }
        50 => {
            world.insert(e, T50);
        }
        51 => {
            world.insert(e, T51);
        }
        52 => {
            world.insert(e, T52);
        }
        53 => {
            world.insert(e, T53);
        }
        54 => {
            world.insert(e, T54);
        }
        55 => {
            world.insert(e, T55);
        }
        56 => {
            world.insert(e, T56);
        }
        57 => {
            world.insert(e, T57);
        }
        58 => {
            world.insert(e, T58);
        }
        59 => {
            world.insert(e, T59);
        }
        60 => {
            world.insert(e, T60);
        }
        61 => {
            world.insert(e, T61);
        }
        62 => {
            world.insert(e, T62);
        }
        63 => {
            world.insert(e, T63);
        }
        64 => {
            world.insert(e, T64);
        }
        65 => {
            world.insert(e, T65);
        }
        66 => {
            world.insert(e, T66);
        }
        67 => {
            world.insert(e, T67);
        }
        68 => {
            world.insert(e, T68);
        }
        69 => {
            world.insert(e, T69);
        }
        70 => {
            world.insert(e, T70);
        }
        71 => {
            world.insert(e, T71);
        }
        72 => {
            world.insert(e, T72);
        }
        73 => {
            world.insert(e, T73);
        }
        74 => {
            world.insert(e, T74);
        }
        75 => {
            world.insert(e, T75);
        }
        76 => {
            world.insert(e, T76);
        }
        77 => {
            world.insert(e, T77);
        }
        78 => {
            world.insert(e, T78);
        }
        79 => {
            world.insert(e, T79);
        }
        80 => {
            world.insert(e, T80);
        }
        81 => {
            world.insert(e, T81);
        }
        82 => {
            world.insert(e, T82);
        }
        83 => {
            world.insert(e, T83);
        }
        84 => {
            world.insert(e, T84);
        }
        85 => {
            world.insert(e, T85);
        }
        86 => {
            world.insert(e, T86);
        }
        87 => {
            world.insert(e, T87);
        }
        88 => {
            world.insert(e, T88);
        }
        89 => {
            world.insert(e, T89);
        }
        90 => {
            world.insert(e, T90);
        }
        91 => {
            world.insert(e, T91);
        }
        92 => {
            world.insert(e, T92);
        }
        93 => {
            world.insert(e, T93);
        }
        94 => {
            world.insert(e, T94);
        }
        95 => {
            world.insert(e, T95);
        }
        96 => {
            world.insert(e, T96);
        }
        97 => {
            world.insert(e, T97);
        }
        98 => {
            world.insert(e, T98);
        }
        99 => {
            world.insert(e, T99);
        }
        _ => unreachable!(),
    }
}

/// Build a world with `total` entities spread across `num_archetypes` archetypes.
/// Every entity has a Position component; each archetype group also has a unique tag.
fn build_fragmented_world(total: u32, num_archetypes: usize) -> World {
    let mut world = World::new();
    for i in 0..total {
        let e = world.spawn(Position {
            x: i as f32,
            y: 0.0,
            z: 0.0,
        });
        insert_tag_by_index(&mut world, e, (i as usize) % num_archetypes);
    }
    world
}

fn bench_archetype_fragmentation(c: &mut Criterion) {
    const TOTAL: u32 = 10_000;
    let mut group = c.benchmark_group("archetype_fragmentation");

    for num_archetypes in [1usize, 10, 100] {
        let world = build_fragmented_world(TOTAL, num_archetypes);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_archetypes}_archetypes")),
            &num_archetypes,
            |b, _| {
                b.iter(|| {
                    let query = Query::<&Position>::new(&world);
                    let mut total = 0u32;
                    for pos in query.iter() {
                        black_box(pos);
                        total += 1;
                    }
                    assert_eq!(total, TOTAL);
                });
            },
        );
    }
    group.finish();
}

// ── Despawn benchmark ───────────────────────────────────────────────────────

fn bench_despawn(c: &mut Criterion) {
    c.bench_function("despawn_1k", |b| {
        b.iter(|| {
            let mut world = World::new();
            let mut entities = Vec::with_capacity(1_000);
            for i in 0..1_000u32 {
                let e = world.spawn(Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                });
                world.insert(
                    e,
                    Velocity {
                        dx: 1.0,
                        dy: 0.0,
                        dz: 0.0,
                    },
                );
                world.insert(
                    e,
                    Health {
                        current: 100.0,
                        max: 100.0,
                    },
                );
                entities.push(e);
            }
            for e in entities {
                world.despawn(e);
            }
            black_box(&world);
        });
    });
}

// ── Additional marker components ────────────────────────────────────────────

/// Zero-sized marker tags used to build entities with varying component counts.
#[derive(Clone, Copy)]
struct Tag1;
#[derive(Clone, Copy)]
struct Tag2;

// ── Archetype column lookup benchmark ──────────────────────────────────────
//
// Creates archetypes with varying numbers of components (2, 5, 10, 20) and
// benchmarks `World::get::<Position>(entity)`. This validates that binary
// search over archetype columns scales logarithmically with column count.

/// Attach `n` tag components to an entity to produce an archetype with
/// Position + n tags. We reuse T00..T19 from the fragmentation section.
fn spawn_entity_with_n_components(world: &mut World, n: usize) -> euca_ecs::Entity {
    let e = world.spawn(Position {
        x: 1.0,
        y: 2.0,
        z: 3.0,
    });
    // Each inserted tag creates a broader archetype: Position + T00 + T01 + ...
    // We insert up to n-1 tags (Position itself is the first component).
    let tags_needed = n.saturating_sub(1);
    if tags_needed > 0 {
        world.insert(e, T00);
    }
    if tags_needed > 1 {
        world.insert(e, T01);
    }
    if tags_needed > 2 {
        world.insert(e, T02);
    }
    if tags_needed > 3 {
        world.insert(e, T03);
    }
    if tags_needed > 4 {
        world.insert(e, T04);
    }
    if tags_needed > 5 {
        world.insert(e, T05);
    }
    if tags_needed > 6 {
        world.insert(e, T06);
    }
    if tags_needed > 7 {
        world.insert(e, T07);
    }
    if tags_needed > 8 {
        world.insert(e, T08);
    }
    if tags_needed > 9 {
        world.insert(e, T09);
    }
    if tags_needed > 10 {
        world.insert(e, T10);
    }
    if tags_needed > 11 {
        world.insert(e, T11);
    }
    if tags_needed > 12 {
        world.insert(e, T12);
    }
    if tags_needed > 13 {
        world.insert(e, T13);
    }
    if tags_needed > 14 {
        world.insert(e, T14);
    }
    if tags_needed > 15 {
        world.insert(e, T15);
    }
    if tags_needed > 16 {
        world.insert(e, T16);
    }
    if tags_needed > 17 {
        world.insert(e, T17);
    }
    if tags_needed > 18 {
        world.insert(e, T18);
    }
    e
}

fn bench_archetype_column_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("archetype_column_lookup");

    for num_components in [2usize, 5, 10, 20] {
        let mut world = World::new();
        let entity = spawn_entity_with_n_components(&mut world, num_components);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_components}_components")),
            &num_components,
            |b, _| {
                b.iter(|| {
                    let pos = world.get::<Position>(entity);
                    black_box(pos);
                });
            },
        );
    }
    group.finish();
}

// ── Parallel query iteration benchmark ─────────────────────────────────────
//
// Compares `World::par_for_each<Position>` (rayon parallel) vs sequential
// `Query::<&Position>::iter()` at different entity counts.

fn bench_parallel_query_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_query_iteration");

    for count in [1_000u32, 10_000, 100_000] {
        let mut world = World::new();
        spawn_three_component_entities(&mut world, count);

        group.bench_with_input(
            BenchmarkId::new("sequential", format!("{count}")),
            &count,
            |b, &n| {
                b.iter(|| {
                    let query = Query::<&Position>::new(&world);
                    let mut total = 0u32;
                    for pos in query.iter() {
                        black_box(pos);
                        total += 1;
                    }
                    assert_eq!(total, n);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("par_for_each", format!("{count}")),
            &count,
            |b, _| {
                b.iter(|| {
                    world.par_for_each::<Position>(|entity, pos| {
                        black_box((entity, pos));
                    });
                });
            },
        );
    }
    group.finish();
}

// ── World tick schedule benchmark ──────────────────────────────────────────
//
// Creates a Schedule with 5 systems and benchmarks `schedule.run()` at
// different entity counts. Measures total per-tick overhead including
// system dispatch, query matching, and iteration.

fn bench_world_tick_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("world_tick_schedule");

    for count in [1_000u32, 10_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{count}")),
            &count,
            |b, &n| {
                b.iter_custom(|iters| {
                    let mut world = World::new();
                    spawn_three_component_entities(&mut world, n);

                    let mut schedule = Schedule::new();
                    // System 1: read Position
                    schedule.add_system(|world: &mut World| {
                        let query = Query::<&Position>::new(world);
                        for pos in query.iter() {
                            black_box(pos);
                        }
                    });
                    // System 2: read Velocity
                    schedule.add_system(|world: &mut World| {
                        let query = Query::<&Velocity>::new(world);
                        for vel in query.iter() {
                            black_box(vel);
                        }
                    });
                    // System 3: read Health
                    schedule.add_system(|world: &mut World| {
                        let query = Query::<&Health>::new(world);
                        for hp in query.iter() {
                            black_box(hp);
                        }
                    });
                    // System 4: read Position + Velocity (movement-like)
                    schedule.add_system(|world: &mut World| {
                        let query = Query::<(&Position, &Velocity)>::new(world);
                        for (pos, vel) in query.iter() {
                            black_box((pos, vel));
                        }
                    });
                    // System 5: read Position + Health (damage-range-like)
                    schedule.add_system(|world: &mut World| {
                        let query = Query::<(&Position, &Health)>::new(world);
                        for (pos, hp) in query.iter() {
                            black_box((pos, hp));
                        }
                    });

                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        schedule.run(&mut world);
                    }
                    start.elapsed()
                });
            },
        );
    }
    group.finish();
}

// ── Entity spawn batch benchmark ───────────────────────────────────────────
//
// Measures the cost of spawning N entities, each with 5 components
// (Position, Velocity, Health, Tag1, Tag2). This benchmarks archetype
// lookup, column allocation, and entity-to-archetype bookkeeping at scale.

/// Spawn a single entity with 5 components: Position, Velocity, Health, Tag1, Tag2.
fn spawn_five_component_entity(world: &mut World, i: u32) {
    let e = world.spawn(Position {
        x: i as f32,
        y: 0.0,
        z: 0.0,
    });
    world.insert(
        e,
        Velocity {
            dx: 1.0,
            dy: 0.0,
            dz: 0.0,
        },
    );
    world.insert(
        e,
        Health {
            current: 100.0,
            max: 100.0,
        },
    );
    world.insert(e, Tag1);
    world.insert(e, Tag2);
}

fn bench_entity_spawn_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_spawn_batch");

    for count in [100u32, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{count}")),
            &count,
            |b, &n| {
                b.iter(|| {
                    let mut world = World::new();
                    for i in 0..n {
                        spawn_five_component_entity(&mut world, i);
                    }
                    black_box(&world);
                });
            },
        );
    }
    group.finish();
}

// ── Criterion harness ───────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_spawn,
    bench_query_iterate,
    bench_query_cached_vs_uncached,
    bench_change_detection_overhead,
    bench_archetype_fragmentation,
    bench_despawn,
    bench_archetype_column_lookup,
    bench_parallel_query_iteration,
    bench_world_tick_schedule,
    bench_entity_spawn_batch,
);
criterion_main!(benches);
