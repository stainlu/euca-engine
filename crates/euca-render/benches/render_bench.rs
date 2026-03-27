//! Benchmarks for the rendering extraction and batching pipeline.
//!
//! These benchmarks measure CPU-side rendering costs (extraction, sorting,
//! instance data generation) without requiring a GPU. GPU-dependent benchmarks
//! would require a real device and are measured via profiling tools instead.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use euca_ecs::World;
use euca_math::{Transform, Vec3};
use euca_render::{
    DrawCommand, MaterialHandle, MeshHandle, MeshRenderer, MaterialRef, RenderExtractor,
};
use euca_scene::{GlobalTransform, LocalTransform};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a world with `n` renderable entities on a grid.
/// Each entity gets LocalTransform, GlobalTransform, MeshRenderer, MaterialRef.
/// Materials cycle through `n_materials` unique handles.
fn build_render_world(n: usize, n_materials: usize) -> World {
    let mut world = World::new();
    let side = (n as f32).sqrt().ceil() as usize;

    for i in 0..n {
        let row = i / side;
        let col = i % side;
        let pos = Vec3::new(col as f32 * 2.0, 0.0, row as f32 * 2.0);

        let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(e, GlobalTransform(Transform::from_translation(pos)));
        world.insert(e, MeshRenderer { mesh: MeshHandle(0) });
        world.insert(
            e,
            MaterialRef {
                handle: MaterialHandle((i % n_materials) as u32),
            },
        );
    }
    world
}

/// Build a Vec<DrawCommand> from a world (simulates the extraction step).
fn extract_draw_commands(world: &World) -> Vec<DrawCommand> {
    use euca_ecs::Query;
    let query = Query::<(&GlobalTransform, &MeshRenderer, &MaterialRef)>::new(world);
    query
        .iter()
        .map(|(gt, mr, mat)| DrawCommand {
            mesh: mr.mesh,
            material: mat.handle,
            model_matrix: gt.0.to_matrix(),
            aabb: None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// Benchmark full DrawCommand extraction from ECS (the baseline that
/// RenderExtractor aims to beat).
fn bench_extract_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("render_extract_full");

    for &n in &[1_000, 10_000, 50_000] {
        let world = build_render_world(n, 5);

        if n >= 50_000 {
            group.sample_size(10);
        }

        group.bench_function(format!("{n}"), |b| {
            b.iter(|| {
                black_box(extract_draw_commands(black_box(&world)));
            });
        });
    }

    group.finish();
}

/// Benchmark RenderExtractor::sync with 100% change rate (first sync = all new).
fn bench_extractor_sync_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("extractor_sync_full");

    for &n in &[1_000, 10_000, 50_000] {
        let world = build_render_world(n, 5);

        if n >= 50_000 {
            group.sample_size(10);
        }

        group.bench_function(format!("{n}"), |b| {
            b.iter(|| {
                let mut extractor = RenderExtractor::new();
                extractor.sync(black_box(&world));
                black_box(extractor.active_count());
            });
        });
    }

    group.finish();
}

/// Benchmark RenderExtractor::sync with ~1% change rate (steady state).
/// First call populates, second call measures incremental sync cost.
fn bench_extractor_sync_partial(c: &mut Criterion) {
    let mut group = c.benchmark_group("extractor_sync_partial");

    for &n in &[10_000, 50_000] {
        let mut world = build_render_world(n, 5);
        let mut extractor = RenderExtractor::new();
        extractor.sync(&world);

        // Advance world tick so change detection distinguishes old from new.
        world.tick();

        // Modify ~1% of entities to simulate typical frame changes.
        let change_count = (n / 100).max(1);
        let entities: Vec<euca_ecs::Entity> = {
            use euca_ecs::Query;
            let q = Query::<(euca_ecs::Entity, &GlobalTransform)>::new(&world);
            q.iter().map(|(e, _)| e).take(change_count).collect()
        };
        for &e in &entities {
            if let Some(gt) = world.get_mut::<GlobalTransform>(e) {
                gt.0.translation.x += 0.1;
            }
        }

        if n >= 50_000 {
            group.sample_size(10);
        }

        group.bench_function(format!("{n}_1pct"), |b| {
            b.iter(|| {
                extractor.sync(black_box(&world));
                black_box(extractor.active_count());
            });
        });
    }

    group.finish();
}

/// Benchmark batch building (sort by mesh/material + InstanceData generation).
/// This measures the CPU cost of the renderer's per-frame batching path.
fn bench_batch_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_build");

    for &n in &[1_000, 10_000, 50_000] {
        let world = build_render_world(n, 10);
        let commands = extract_draw_commands(&world);
        let cmd_refs: Vec<&DrawCommand> = commands.iter().collect();

        if n >= 50_000 {
            group.sample_size(10);
        }

        group.bench_function(format!("{n}"), |b| {
            b.iter(|| {
                // build_batches_from_refs is a private method, so we call the
                // public draw path components manually: sort + instance build.
                // For now, measure the full extraction + sort cost.
                let mut indices: Vec<usize> = (0..cmd_refs.len()).collect();
                indices.sort_by_key(|&i| (cmd_refs[i].mesh.0, cmd_refs[i].material.0));
                black_box(&indices);
                // Measure model_matrix inverse (normal matrix) computation
                for &i in &indices {
                    let model = cmd_refs[i].model_matrix;
                    black_box(model.inverse().transpose());
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_extract_full,
    bench_extractor_sync_full,
    bench_extractor_sync_partial,
    bench_batch_build,
);
criterion_main!(benches);
