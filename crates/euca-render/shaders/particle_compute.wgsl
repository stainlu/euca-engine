// GPU particle compute shader — emit and update particles on the GPU.
//
// Two entry points: `emit` spawns new particles, `update` advances physics.

struct Particle {
    position: vec3<f32>,
    age: f32,
    velocity: vec3<f32>,
    lifetime: f32,
    size: f32,
    color: vec4<f32>,
    _pad: vec3<f32>,
};

struct EmitParams {
    emitter_position: vec3<f32>,
    emit_count: u32,
    speed_min: f32,
    speed_max: f32,
    size_min: f32,
    size_max: f32,
    lifetime_min: f32,
    lifetime_max: f32,
    gravity: vec3<f32>,
    dt: f32,
    time: f32,
    max_particles: u32,
    color_start: vec4<f32>,
    color_end: vec4<f32>,
    // Cone emission: direction + half-angle
    emit_direction: vec3<f32>,
    cone_half_angle: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> counters: array<atomic<u32>>;
// counters[0] = alive_count
@group(0) @binding(2) var<uniform> params: EmitParams;

// PCG hash for pseudo-random numbers on GPU
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967295.0;
}

fn rand_unit_sphere(seed: u32) -> vec3<f32> {
    let theta = rand_f32(seed) * 6.2831853;
    let phi = acos(2.0 * rand_f32(seed + 1u) - 1.0);
    let r = pow(rand_f32(seed + 2u), 1.0 / 3.0);
    return vec3<f32>(
        r * sin(phi) * cos(theta),
        r * sin(phi) * sin(theta),
        r * cos(phi),
    );
}

@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.emit_count {
        return;
    }

    // Allocate a particle slot
    let slot = atomicAdd(&counters[0], 1u);
    if slot >= params.max_particles {
        atomicSub(&counters[0], 1u);
        return;
    }

    let seed = idx * 7919u + u32(params.time * 1000.0);

    // Random direction within cone
    let dir = normalize(rand_unit_sphere(seed) + params.emit_direction * 3.0);
    let speed = mix(params.speed_min, params.speed_max, rand_f32(seed + 100u));
    let lifetime = mix(params.lifetime_min, params.lifetime_max, rand_f32(seed + 200u));
    let size = mix(params.size_min, params.size_max, rand_f32(seed + 300u));

    var p: Particle;
    p.position = params.emitter_position;
    p.velocity = dir * speed;
    p.age = 0.0;
    p.lifetime = lifetime;
    p.size = size;
    p.color = params.color_start;
    p._pad = vec3<f32>(0.0);

    particles[slot] = p;
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let alive = atomicLoad(&counters[0]);
    if idx >= alive {
        return;
    }

    var p = particles[idx];
    p.age += params.dt;

    // Kill expired particles by swapping with last alive
    if p.age >= p.lifetime {
        let last = atomicSub(&counters[0], 1u) - 1u;
        if idx < last {
            particles[idx] = particles[last];
        }
        return;
    }

    // Physics: gravity + velocity integration
    p.velocity += params.gravity * params.dt;
    p.position += p.velocity * params.dt;

    // Color interpolation over lifetime
    let t = p.age / p.lifetime;
    p.color = mix(params.color_start, params.color_end, vec4<f32>(t));

    // Size fade: shrink toward end of life
    p.size *= (1.0 - t * 0.3);

    particles[idx] = p;
}
