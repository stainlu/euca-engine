//! Benchmarks for the euca-animation crate.
//!
//! Covers the three core hot paths:
//! - Per-joint pose blending (`AnimPose::blend`)
//! - Multi-layer blender evaluation (`AnimationBlender::evaluate`)
//! - State machine update tick (`AnimStateMachine::update`)

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use euca_animation::{
    AnimPose, AnimStateMachine, AnimationBlender, CompareOp, TransitionCondition,
};
use euca_math::{Quat, Transform, Vec3};

// ---------------------------------------------------------------------------
// Helpers — deterministic pose construction (no external rand)
// ---------------------------------------------------------------------------

/// Build a pose with `n` joints whose transforms vary deterministically by index.
fn make_pose(n: usize, seed: f32) -> AnimPose {
    AnimPose {
        joints: (0..n)
            .map(|i| {
                let f = i as f32 + seed;
                Transform {
                    translation: Vec3::new(f, f * 0.5, f * 0.25),
                    rotation: Quat::from_axis_angle(Vec3::Y, f * 0.1),
                    scale: Vec3::new(1.0 + f * 0.01, 1.0 + f * 0.02, 1.0 + f * 0.015),
                }
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// A. blend_two_poses — AnimPose::blend with varying joint counts
// ---------------------------------------------------------------------------

fn blend_two_poses(c: &mut Criterion) {
    let mut group = c.benchmark_group("blend_two_poses");

    for joint_count in [20, 50, 100] {
        let pose_a = make_pose(joint_count, 0.0);
        let pose_b = make_pose(joint_count, 100.0);

        group.bench_with_input(
            BenchmarkId::from_parameter(joint_count),
            &joint_count,
            |b, _| {
                b.iter(|| black_box(pose_a.blend(&pose_b, 0.5)));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// B. state_machine_evaluate — AnimStateMachine::update with 5 states
// ---------------------------------------------------------------------------

fn build_five_state_machine() -> (AnimStateMachine, Vec<f32>) {
    let mut sm = AnimStateMachine::new(0);

    // States: idle(0), walk(1), run(2), jump(3), fall(4)
    sm.add_state("idle", 0);
    sm.add_state("walk", 1);
    sm.add_state("run", 2);
    sm.add_state("jump", 3);
    sm.add_state("fall", 4);

    // idle -> walk: speed > 0.1
    sm.add_transition(
        0,
        1,
        vec![TransitionCondition::FloatCompare {
            param: "speed".into(),
            op: CompareOp::Greater,
            threshold: 0.1,
        }],
        0.2,
    );

    // walk -> run: speed > 0.6
    sm.add_transition(
        1,
        2,
        vec![TransitionCondition::FloatCompare {
            param: "speed".into(),
            op: CompareOp::Greater,
            threshold: 0.6,
        }],
        0.3,
    );

    // walk -> idle: speed <= 0.1
    sm.add_transition(
        1,
        0,
        vec![TransitionCondition::FloatCompare {
            param: "speed".into(),
            op: CompareOp::LessOrEqual,
            threshold: 0.1,
        }],
        0.2,
    );

    // run -> walk: speed <= 0.6
    sm.add_transition(
        2,
        1,
        vec![TransitionCondition::FloatCompare {
            param: "speed".into(),
            op: CompareOp::LessOrEqual,
            threshold: 0.6,
        }],
        0.25,
    );

    // Any state -> jump: is_jumping == true
    sm.add_any_state_transition(
        3,
        vec![TransitionCondition::BoolEquals {
            param: "is_jumping".into(),
            value: true,
        }],
        0.1,
    );

    // jump -> fall: airborne_time > 0.3
    sm.add_transition(
        3,
        4,
        vec![TransitionCondition::FloatCompare {
            param: "airborne_time".into(),
            op: CompareOp::Greater,
            threshold: 0.3,
        }],
        0.15,
    );

    // fall -> idle: is_grounded == true
    sm.add_transition(
        4,
        0,
        vec![TransitionCondition::BoolEquals {
            param: "is_grounded".into(),
            value: true,
        }],
        0.2,
    );

    // Set initial parameters (steady walk — no transitions fire at idle)
    sm.set_float("speed", 0.3);
    sm.set_bool("is_jumping", false);
    sm.set_float("airborne_time", 0.0);
    sm.set_bool("is_grounded", true);

    let clip_durations = vec![2.0, 1.0, 0.8, 0.6, 1.5];
    (sm, clip_durations)
}

fn state_machine_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_machine_evaluate");

    let (sm_template, clip_durations) = build_five_state_machine();

    // Benchmark a steady-state update (no transition fires)
    group.bench_function("steady_state", |b| {
        let mut sm = sm_template.clone();
        b.iter(|| {
            black_box(sm.update(black_box(0.016), &clip_durations));
        });
    });

    // Benchmark an update where a transition fires
    group.bench_function("with_transition", |b| {
        b.iter_batched(
            || {
                let mut sm = sm_template.clone();
                // Put machine into idle (state 0) so the speed > 0.1 condition
                // triggers idle -> walk on the next update.
                sm.set_float("speed", 0.0);
                sm.update(0.016, &clip_durations);
                // Now set speed high enough to trigger the transition
                sm.set_float("speed", 0.5);
                sm
            },
            |mut sm| {
                black_box(sm.update(black_box(0.016), &clip_durations));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// C. blender_evaluate — AnimationBlender with varying layer counts
// ---------------------------------------------------------------------------

fn blender_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("blender_evaluate");

    let joint_count = 50;

    for layer_count in [2, 4, 8] {
        // Pre-build the poses so construction cost is excluded from the benchmark
        let poses: Vec<AnimPose> = (0..layer_count)
            .map(|i| make_pose(joint_count, i as f32 * 10.0))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(layer_count),
            &layer_count,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut blender = AnimationBlender::new();
                        for (i, pose) in poses.iter().enumerate() {
                            // Weights decrease for later layers (primary = strongest)
                            let weight = 1.0 / (i as f32 + 1.0);
                            blender.add_layer(pose.clone(), weight);
                        }
                        blender
                    },
                    |blender| {
                        black_box(blender.evaluate(black_box(n)));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    blend_two_poses,
    state_machine_evaluate,
    blender_evaluate,
);
criterion_main!(benches);
