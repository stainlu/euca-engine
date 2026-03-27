use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use euca_math::{Mat4, Quat, Vec3, Vec4};

// ---------------------------------------------------------------------------
// Helpers: deterministic test data (no RNG dependency)
// ---------------------------------------------------------------------------

/// Generate N distinct Vec3 values by cycling through a simple hash-like sequence.
fn make_vec3s(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            let f = i as f32;
            Vec3::new(f * 0.123 + 1.0, f * 0.456 + 2.0, f * 0.789 + 3.0)
        })
        .collect()
}

fn make_vec4s(n: usize) -> Vec<Vec4> {
    (0..n)
        .map(|i| {
            let f = i as f32;
            Vec4::new(
                f * 0.123 + 1.0,
                f * 0.456 + 2.0,
                f * 0.789 + 3.0,
                f * 0.234 + 4.0,
            )
        })
        .collect()
}

fn make_mat4s(n: usize) -> Vec<Mat4> {
    (0..n)
        .map(|i| {
            let f = i as f32;
            let axis = Vec3::new(1.0 + f * 0.01, 2.0 + f * 0.02, 3.0 + f * 0.03).normalize();
            Mat4::from_scale_rotation_translation(
                Vec3::new(1.0 + f * 0.1, 1.5 + f * 0.1, 2.0 + f * 0.1),
                Quat::from_axis_angle(axis, f * 0.05),
                Vec3::new(f * 10.0, f * 20.0, f * 30.0),
            )
        })
        .collect()
}

fn make_quats(n: usize) -> Vec<Quat> {
    (0..n)
        .map(|i| {
            let f = i as f32;
            let axis = Vec3::new(1.0 + f * 0.01, 2.0 + f * 0.02, 3.0 + f * 0.03).normalize();
            Quat::from_axis_angle(axis, f * 0.1)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark functions
// ---------------------------------------------------------------------------

const N: usize = 10_000;

fn vec3_dot(c: &mut Criterion) {
    let a = make_vec3s(N);
    let b = make_vec3s(N);
    c.bench_function("vec3_dot", |bench| {
        bench.iter(|| {
            for i in 0..N {
                black_box(a[i].dot(b[i]));
            }
        });
    });
}

fn vec3_cross(c: &mut Criterion) {
    let a = make_vec3s(N);
    let b = make_vec3s(N);
    c.bench_function("vec3_cross", |bench| {
        bench.iter(|| {
            for i in 0..N {
                black_box(a[i].cross(b[i]));
            }
        });
    });
}

fn vec3_normalize(c: &mut Criterion) {
    let v = make_vec3s(N);
    c.bench_function("vec3_normalize", |bench| {
        bench.iter(|| {
            for vi in &v {
                black_box(vi.normalize());
            }
        });
    });
}

fn vec3_length(c: &mut Criterion) {
    let v = make_vec3s(N);
    c.bench_function("vec3_length", |bench| {
        bench.iter(|| {
            for vi in &v {
                black_box(vi.length());
            }
        });
    });
}

fn vec4_dot(c: &mut Criterion) {
    let a = make_vec4s(N);
    let b = make_vec4s(N);
    c.bench_function("vec4_dot", |bench| {
        bench.iter(|| {
            for i in 0..N {
                black_box(a[i].dot(b[i]));
            }
        });
    });
}

fn mat4_multiply(c: &mut Criterion) {
    let a = make_mat4s(N);
    let b = make_mat4s(N);
    c.bench_function("mat4_multiply", |bench| {
        bench.iter(|| {
            for i in 0..N {
                black_box(a[i] * b[i]);
            }
        });
    });
}

fn mat4_transform_point(c: &mut Criterion) {
    let m = make_mat4s(N);
    let p = make_vec3s(N);
    c.bench_function("mat4_transform_point", |bench| {
        bench.iter(|| {
            for i in 0..N {
                black_box(m[i].transform_point3(p[i]));
            }
        });
    });
}

fn quat_multiply(c: &mut Criterion) {
    let a = make_quats(N);
    let b = make_quats(N);
    c.bench_function("quat_multiply", |bench| {
        bench.iter(|| {
            for i in 0..N {
                black_box(a[i] * b[i]);
            }
        });
    });
}

fn quat_rotate_vec3(c: &mut Criterion) {
    let q = make_quats(N);
    let v = make_vec3s(N);
    c.bench_function("quat_rotate_vec3", |bench| {
        bench.iter(|| {
            for i in 0..N {
                black_box(q[i] * v[i]);
            }
        });
    });
}

fn mat4_inverse(c: &mut Criterion) {
    let m = make_mat4s(N);
    c.bench_function("mat4_inverse", |bench| {
        bench.iter(|| {
            for mi in &m {
                black_box(mi.inverse());
            }
        });
    });
}

// ---------------------------------------------------------------------------
// SIMD validation benchmarks
// ---------------------------------------------------------------------------

/// Multi-scale normalize throughput — validates rsqrt performance across data sizes.
fn vec3_normalize_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec3_normalize_throughput");
    for &size in &[100, 1_000, 10_000, 100_000] {
        let v = make_vec3s(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, &n| {
            bench.iter(|| {
                for i in 0..n {
                    black_box(v[i].normalize());
                }
            });
        });
    }
    group.finish();
}

/// Pure scalar normalize — baseline reference for comparing against SIMD rsqrt path.
fn vec3_normalize_reference_scalar(c: &mut Criterion) {
    let v = make_vec3s(N);
    c.bench_function("vec3_normalize_reference_scalar", |bench| {
        bench.iter(|| {
            for vi in &v {
                let (x, y, z) = (vi.x, vi.y, vi.z);
                let inv = 1.0_f32 / (x * x + y * y + z * z).sqrt();
                black_box(Vec3::new(x * inv, y * inv, z * inv));
            }
        });
    });
}

/// Chain of 10 Mat4 multiplies per iteration — validates FMA utilization under
/// sustained multiply pressure.
fn mat4_multiply_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat4_multiply_chain");
    for &size in &[1_000, 10_000] {
        let mats = make_mat4s(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, &n| {
            bench.iter(|| {
                for chunk in mats[..n].chunks(10) {
                    let mut acc = chunk[0];
                    for m in &chunk[1..] {
                        acc = acc * *m;
                    }
                    black_box(acc);
                }
            });
        });
    }
    group.finish();
}

/// Slerp throughput with diverse interpolation factors — exercises the full
/// slerp code path (sin/cos branch, not the near-parallel lerp fallback).
fn quat_slerp_diverse(c: &mut Criterion) {
    let a = make_quats(N);
    let b: Vec<Quat> = (0..N)
        .map(|i| {
            // Rotate around varied axes with meaningful angular separation.
            let f = i as f32;
            let axis = Vec3::new(
                (f * 0.73 + 1.0).sin(),
                (f * 0.37 + 2.0).cos(),
                (f * 0.51 + 3.0).sin(),
            )
            .normalize();
            Quat::from_axis_angle(axis, 0.5 + (f * 0.13) % 2.5)
        })
        .collect();
    c.bench_function("quat_slerp_diverse", |bench| {
        bench.iter(|| {
            for i in 0..N {
                // Varied t values across [0,1].
                let t = (i as f32 * 0.0001).fract();
                black_box(a[i].slerp(b[i], t));
            }
        });
    });
}

/// Side-by-side scalar vs SIMD dot product — measures the speedup from SIMD
/// horizontal-sum dot relative to an inline scalar fallback.
fn simd_dot_vs_scalar_reference(c: &mut Criterion) {
    let a = make_vec3s(N);
    let b = make_vec3s(N);
    let mut group = c.benchmark_group("simd_dot_vs_scalar_reference");

    group.bench_function("scalar_dot", |bench| {
        bench.iter(|| {
            for i in 0..N {
                let (ax, ay, az) = (a[i].x, a[i].y, a[i].z);
                let (bx, by, bz) = (b[i].x, b[i].y, b[i].z);
                black_box(ax * bx + ay * by + az * bz);
            }
        });
    });

    group.bench_function("simd_dot", |bench| {
        bench.iter(|| {
            for i in 0..N {
                black_box(a[i].dot(b[i]));
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    vec3_dot,
    vec3_cross,
    vec3_normalize,
    vec3_length,
    vec4_dot,
    mat4_multiply,
    mat4_transform_point,
    quat_multiply,
    quat_rotate_vec3,
    mat4_inverse,
    vec3_normalize_throughput,
    vec3_normalize_reference_scalar,
    mat4_multiply_chain,
    quat_slerp_diverse,
    simd_dot_vs_scalar_reference,
);
criterion_main!(benches);
