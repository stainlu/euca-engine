use criterion::{Criterion, black_box, criterion_group, criterion_main};
use euca_ecs::{Entity, Query, World};

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

fn bench_spawn(c: &mut Criterion) {
    c.bench_function("spawn 10k entities", |b| {
        b.iter(|| {
            let mut world = World::new();
            for i in 0..10_000 {
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
            }
            black_box(&world);
        });
    });
}

fn bench_query_iter(c: &mut Criterion) {
    let mut world = World::new();
    for i in 0..10_000 {
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
    }

    c.bench_function("query iter 10k (Position, Velocity)", |b| {
        b.iter(|| {
            let query = Query::<(&Position, &Velocity)>::new(&world);
            let mut count = 0u32;
            for (pos, vel) in query.iter() {
                black_box((pos, vel));
                count += 1;
            }
            assert_eq!(count, 10_000);
        });
    });
}

fn bench_get_component(c: &mut Criterion) {
    let mut world = World::new();
    let mut entities = Vec::new();
    for i in 0..10_000 {
        let e = world.spawn(Position {
            x: i as f32,
            y: 0.0,
            z: 0.0,
        });
        entities.push(e);
    }

    c.bench_function("get component 10k random", |b| {
        b.iter(|| {
            for &e in &entities {
                black_box(world.get::<Position>(e));
            }
        });
    });
}

fn bench_despawn(c: &mut Criterion) {
    c.bench_function("spawn+despawn 10k entities", |b| {
        b.iter(|| {
            let mut world = World::new();
            let mut entities = Vec::with_capacity(10_000);
            for i in 0..10_000 {
                entities.push(world.spawn(Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                }));
            }
            for e in entities {
                world.despawn(e);
            }
            black_box(&world);
        });
    });
}

fn bench_par_for_each(c: &mut Criterion) {
    let mut world = World::new();
    for i in 0..100_000 {
        world.spawn(Position {
            x: i as f32,
            y: 0.0,
            z: 0.0,
        });
    }

    c.bench_function("par_for_each 100k entities", |b| {
        b.iter(|| {
            let mut sum = std::sync::atomic::AtomicU64::new(0);
            world.par_for_each::<Position>(|_e, pos| {
                sum.fetch_add(pos.x as u64, std::sync::atomic::Ordering::Relaxed);
            });
            black_box(sum.load(std::sync::atomic::Ordering::Relaxed));
        });
    });
}

fn bench_headless_tick(c: &mut Criterion) {
    let mut world = World::new();
    for i in 0..1_000 {
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
    }

    let mut schedule = euca_ecs::Schedule::new();
    schedule.add_system(|world: &mut World| {
        let updates: Vec<(Entity, f32, f32, f32)> = {
            let query = Query::<(Entity, &Velocity)>::new(world);
            query.iter().map(|(e, v)| (e, v.dx, v.dy, v.dz)).collect()
        };
        for (entity, dx, dy, dz) in updates {
            if let Some(pos) = world.get_mut::<Position>(entity) {
                pos.x += dx;
                pos.y += dy;
                pos.z += dz;
            }
        }
    });

    c.bench_function("headless tick 1k entities (movement system)", |b| {
        b.iter(|| {
            schedule.run(&mut world);
        });
    });
}

criterion_group!(
    benches,
    bench_spawn,
    bench_query_iter,
    bench_get_component,
    bench_despawn,
    bench_par_for_each,
    bench_headless_tick,
);
criterion_main!(benches);
