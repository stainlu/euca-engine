use criterion::{Criterion, black_box, criterion_group, criterion_main};
use euca_ecs::World;
use euca_math::{Transform, Vec3};
use euca_physics::{
    Collider, PhysicsBody, PhysicsConfig, Ray, Velocity, intersect_aabb, overlap_sphere,
    physics_step_system, raycast_world,
};
use euca_scene::{GlobalTransform, LocalTransform};

// ---------------------------------------------------------------------------
// Deterministic pseudo-random number generator (avoids adding `rand` dep)
// ---------------------------------------------------------------------------

struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Returns a float in [min, max).
    fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        let t = (self.next_u64() & 0xFFFF_FFFF) as f32 / u32::MAX as f32;
        min + t * (max - min)
    }

    fn random_pos(&mut self, spread: f32) -> Vec3 {
        Vec3::new(
            self.range_f32(-spread, spread),
            self.range_f32(-spread, spread),
            self.range_f32(-spread, spread),
        )
    }
}

// ---------------------------------------------------------------------------
// World setup helpers
// ---------------------------------------------------------------------------

/// Build a world with `n` dynamic bodies (PhysicsBody, Collider, Velocity,
/// LocalTransform, GlobalTransform) scattered randomly.
fn build_dynamic_world(n: usize) -> World {
    let mut world = World::new();
    world.insert_resource(PhysicsConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        fixed_dt: 1.0 / 60.0,
        max_substeps: 1,
    });

    let spread = (n as f32).sqrt() * 2.0;
    let mut rng = SimpleRng::new(42);

    for _ in 0..n {
        let pos = rng.random_pos(spread);
        let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(e, GlobalTransform::default());
        world.insert(e, PhysicsBody::dynamic());
        world.insert(
            e,
            Velocity {
                linear: Vec3::new(
                    rng.range_f32(-1.0, 1.0),
                    rng.range_f32(-1.0, 1.0),
                    rng.range_f32(-1.0, 1.0),
                ),
                angular: Vec3::ZERO,
            },
        );
        world.insert(e, Collider::aabb(0.5, 0.5, 0.5));
    }

    world
}

/// Build a world with `n` static colliders (for raycast/overlap queries).
fn build_collider_world(n: usize) -> World {
    let mut world = World::new();

    let spread = (n as f32).sqrt() * 2.0;
    let mut rng = SimpleRng::new(123);

    for _ in 0..n {
        let pos = rng.random_pos(spread);
        let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(e, GlobalTransform::default());
        world.insert(e, PhysicsBody::fixed());
        world.insert(e, Collider::aabb(0.5, 0.5, 0.5));
    }

    world
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_physics_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("physics_step");

    for &n in &[100, 1000] {
        let mut world = build_dynamic_world(n);

        group.bench_function(format!("{n}"), |b| {
            b.iter(|| {
                physics_step_system(black_box(&mut world));
            });
        });
    }

    group.finish();
}

fn bench_raycast(c: &mut Criterion) {
    let mut group = c.benchmark_group("raycast");

    for &n in &[100, 1000] {
        let world = build_collider_world(n);
        let ray = Ray::new(Vec3::new(-100.0, 0.0, 0.0), Vec3::X);

        group.bench_function(format!("{n}"), |b| {
            b.iter(|| {
                black_box(raycast_world(
                    black_box(&world),
                    &ray,
                    f32::INFINITY,
                    u32::MAX,
                ));
            });
        });
    }

    group.finish();
}

fn bench_overlap_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("overlap_sphere");

    for &n in &[100, 1000] {
        let world = build_collider_world(n);

        group.bench_function(format!("{n}"), |b| {
            b.iter(|| {
                black_box(overlap_sphere(black_box(&world), Vec3::ZERO, 5.0, u32::MAX));
            });
        });
    }

    group.finish();
}

fn bench_collision_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("collision_detection");

    for &n in &[100, 1000] {
        // Pre-generate N AABB pairs (some overlapping, some not).
        let mut rng = SimpleRng::new(77);
        let pairs: Vec<(Vec3, Vec3)> = (0..n)
            .map(|_| {
                let a = rng.random_pos(10.0);
                // Place b near a so roughly half will overlap.
                let b = a + Vec3::new(
                    rng.range_f32(-1.5, 1.5),
                    rng.range_f32(-1.5, 1.5),
                    rng.range_f32(-1.5, 1.5),
                );
                (a, b)
            })
            .collect();

        group.bench_function(format!("{n}"), |b| {
            b.iter(|| {
                for &(pos_a, pos_b) in black_box(&pairs) {
                    black_box(intersect_aabb(pos_a, 0.5, 0.5, 0.5, pos_b, 0.5, 0.5, 0.5));
                }
            });
        });
    }

    group.finish();
}

fn bench_broad_phase(c: &mut Criterion) {
    let mut group = c.benchmark_group("broad_phase");

    for &n in &[100, 1000] {
        // Build a world with N dynamic bodies. physics_step_system runs
        // broadphase (spatial hash build + query) followed by narrowphase.
        // Using zero gravity and zero velocity isolates the
        // broadphase/narrowphase cost (no integration work).
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });

        let spread = (n as f32).sqrt() * 2.0;
        let mut rng = SimpleRng::new(99);

        for _ in 0..n {
            let pos = rng.random_pos(spread);
            let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
            world.insert(e, GlobalTransform::default());
            world.insert(e, PhysicsBody::dynamic());
            world.insert(e, Velocity::default());
            world.insert(e, Collider::aabb(0.5, 0.5, 0.5));
        }

        group.bench_function(format!("{n}"), |b| {
            b.iter(|| {
                physics_step_system(black_box(&mut world));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_physics_step,
    bench_raycast,
    bench_overlap_sphere,
    bench_collision_detection,
    bench_broad_phase,
);
criterion_main!(benches);
