// Per-pixel motion blur compute shader.
//
// Two-pass approach:
//   Pass 0 (tile): Divide screen into tiles, find max velocity magnitude per tile.
//   Pass 1 (blur): For each pixel, sample along velocity vector with distance weighting.
//
// Bindings (group 0):
//   @binding(0) params       -- uniform MotionBlurParams
//   @binding(1) color_tex    -- input color texture (texture_2d<f32>)
//   @binding(2) velocity_tex -- velocity buffer (Rg16Float)
//   @binding(3) tile_max_tex -- tile max velocity (storage, written by tile pass / read by blur pass)
//   @binding(4) output_tex   -- output color (storage, write)

struct MotionBlurParams {
    resolution: vec2f,       // screen dimensions
    inv_resolution: vec2f,   // 1.0 / resolution
    intensity: f32,          // velocity multiplier
    max_velocity: f32,       // clamp velocity magnitude (pixels)
    sample_count: u32,       // number of taps along velocity
    tile_size: u32,          // tile dimensions (16)
    tile_count: vec2u,       // number of tiles (ceil(resolution / tile_size))
    _pad: vec2u,
};

@group(0) @binding(0) var<uniform> params: MotionBlurParams;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var velocity_tex: texture_2d<f32>;
@group(0) @binding(3) var tile_max_tex: texture_storage_2d<rg16float, read_write>;
@group(0) @binding(4) var output_tex: texture_storage_2d<rgba16float, write>;

// ── Pass 0: Tile max velocity ──────────────────────────────────────────────
// Each workgroup processes one 16x16 tile. We find the maximum velocity
// magnitude in the tile and store the velocity with that magnitude.
var<workgroup> shared_max_vel: array<vec2f, 256>;

@compute @workgroup_size(16, 16)
fn tile_max(@builtin(global_invocation_id) gid: vec3u,
            @builtin(local_invocation_index) lid: u32) {
    let pixel = vec2i(gid.xy);
    let dims = vec2i(params.resolution);

    var vel = vec2f(0.0);
    if pixel.x < dims.x && pixel.y < dims.y {
        let raw_vel = textureLoad(velocity_tex, pixel, 0).xy;
        // Convert from UV-space to pixel-space and apply intensity
        vel = raw_vel * params.resolution * params.intensity;
        // Clamp to max velocity
        let mag = length(vel);
        if mag > params.max_velocity {
            vel = vel * (params.max_velocity / mag);
        }
    }

    shared_max_vel[lid] = vel;
    workgroupBarrier();

    // Parallel reduction: find the velocity with maximum magnitude
    var stride = 128u;
    loop {
        if stride == 0u {
            break;
        }
        if lid < stride {
            let a = shared_max_vel[lid];
            let b = shared_max_vel[lid + stride];
            if dot(b, b) > dot(a, a) {
                shared_max_vel[lid] = b;
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if lid == 0u {
        let tile = vec2i(gid.xy) / vec2i(i32(params.tile_size));
        let tile_dims = vec2i(params.tile_count);
        if tile.x < tile_dims.x && tile.y < tile_dims.y {
            textureStore(tile_max_tex, tile, vec4f(shared_max_vel[0], 0.0, 0.0));
        }
    }
}

// ── Pass 1: Directional blur ───────────────────────────────────────────────
// For each pixel, check tile velocity. If negligible, pass through.
// Otherwise, sample along the pixel's velocity vector.
@compute @workgroup_size(8, 8)
fn blur(@builtin(global_invocation_id) gid: vec3u) {
    let pixel = vec2i(gid.xy);
    let dims = vec2i(params.resolution);
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    // Check tile velocity magnitude — early out if static
    let tile_coord = pixel / vec2i(i32(params.tile_size));
    let tile_vel = textureLoad(tile_max_tex, tile_coord, 0).xy;
    let tile_mag = length(tile_vel);

    let center_color = textureLoad(color_tex, pixel, 0);

    // Skip blur if tile velocity is negligible (< 0.5 pixels)
    if tile_mag < 0.5 {
        textureStore(output_tex, pixel, center_color);
        return;
    }

    // Per-pixel velocity (in pixels)
    let raw_vel = textureLoad(velocity_tex, pixel, 0).xy;
    var vel = raw_vel * params.resolution * params.intensity;
    let mag = length(vel);

    // If this pixel itself has negligible motion, pass through
    if mag < 0.5 {
        textureStore(output_tex, pixel, center_color);
        return;
    }

    // Clamp velocity
    if mag > params.max_velocity {
        vel = vel * (params.max_velocity / mag);
    }

    // Sample along the velocity vector
    let n = params.sample_count;
    var accum = vec4f(0.0);
    var total_weight = 0.0;

    for (var i = 0u; i < n; i++) {
        // t ranges from -0.5 to +0.5 (sample behind and ahead of the pixel)
        let t = (f32(i) + 0.5) / f32(n) - 0.5;
        let offset = vel * t;
        let sample_pos = vec2i(pixel) + vec2i(i32(round(offset.x)), i32(round(offset.y)));

        // Clamp to screen bounds
        let clamped = clamp(sample_pos, vec2i(0), dims - vec2i(1));
        let sample_color = textureLoad(color_tex, clamped, 0);

        // Weight by distance from center (Gaussian-like falloff)
        let w = 1.0 - 2.0 * abs(t);
        accum += sample_color * w;
        total_weight += w;
    }

    let result = accum / total_weight;
    textureStore(output_tex, pixel, result);
}
