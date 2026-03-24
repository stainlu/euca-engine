// Screen-Space Global Illumination (SSGI) compute shader.
//
// Ray-marches the depth buffer in screen space to approximate indirect diffuse
// lighting. For each half-res pixel, N rays are cast in a cosine-weighted
// hemisphere around the surface normal. On intersection, the previous frame's
// HDR color is sampled at the hit point. Results are temporally accumulated
// with the previous frame's GI via a blend factor.
//
// Runs at half resolution for performance; the output is upsampled during
// compositing.

struct SsgiParams {
    inv_view_proj: mat4x4f,
    prev_view_proj: mat4x4f,
    screen_size: vec2f,       // half-res dimensions
    ray_count: u32,
    max_steps: u32,
    max_distance: f32,
    intensity: f32,
    temporal_blend: f32,
    frame_index: u32,
}

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var prev_color_tex: texture_2d<f32>;
@group(0) @binding(3) var prev_depth_tex: texture_2d<f32>;
@group(0) @binding(4) var history_tex: texture_2d<f32>;
@group(0) @binding(5) var<uniform> params: SsgiParams;
@group(0) @binding(6) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var linear_sampler: sampler;

// -----------------------------------------------------------------------
// Pseudo-random hash (PCG-based)
// -----------------------------------------------------------------------

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_float(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967295.0;
}

// -----------------------------------------------------------------------
// Coordinate reconstruction
// -----------------------------------------------------------------------

// Reconstruct world position from UV (in full-res space) and depth.
fn reconstruct_world_pos(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(uv * 2.0 - 1.0, depth, 1.0);
    let ndc_flipped = vec4f(ndc.x, -ndc.y, ndc.z, 1.0);
    let world = params.inv_view_proj * ndc_flipped;
    return world.xyz / world.w;
}

// Project world position to previous frame UV.
fn project_to_prev_uv(world_pos: vec3f) -> vec2f {
    let clip = params.prev_view_proj * vec4f(world_pos, 1.0);
    let ndc = clip.xy / clip.w;
    return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

// Decode prepass normal: stored as N * 0.5 + 0.5 in view space.
fn decode_normal(encoded: vec3f) -> vec3f {
    return normalize(encoded * 2.0 - 1.0);
}

// -----------------------------------------------------------------------
// Cosine-weighted hemisphere sampling
// -----------------------------------------------------------------------

// Build an orthonormal basis (TBN) from a normal vector.
fn build_tbn(n: vec3f) -> mat3x3f {
    var t: vec3f;
    if abs(n.y) < 0.999 {
        t = normalize(cross(n, vec3f(0.0, 1.0, 0.0)));
    } else {
        t = normalize(cross(n, vec3f(1.0, 0.0, 0.0)));
    }
    let b = cross(n, t);
    return mat3x3f(t, b, n);
}

// Sample a cosine-weighted direction in the hemisphere around +Z,
// then transform by the TBN matrix to orient around the surface normal.
fn cosine_hemisphere_sample(tbn: mat3x3f, seed: ptr<function, u32>) -> vec3f {
    let u1 = rand_float(seed);
    let u2 = rand_float(seed);
    let r = sqrt(u1);
    let theta = 6.2831853 * u2; // 2 * PI
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(1.0 - u1, 0.0));
    return normalize(tbn * vec3f(x, y, z));
}

// -----------------------------------------------------------------------
// Main compute entry
// -----------------------------------------------------------------------

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let half_pixel = vec2i(id.xy);
    let half_dims = vec2i(params.screen_size);

    if half_pixel.x >= half_dims.x || half_pixel.y >= half_dims.y {
        return;
    }

    // Map half-res pixel to full-res coordinates (center of 2x2 block).
    let full_pixel = half_pixel * 2 + vec2i(1, 1);
    let full_dims = half_dims * 2;
    let full_uv = (vec2f(full_pixel) + 0.5) / vec2f(full_dims);

    // Sample depth and normal from full-res buffers.
    let depth = textureLoad(depth_tex, full_pixel, 0).r;

    // Skip sky pixels (depth at or beyond far plane).
    if depth >= 1.0 {
        textureStore(output, half_pixel, vec4f(0.0));
        return;
    }

    let normal_encoded = textureLoad(normal_tex, full_pixel, 0).rgb;
    let normal = decode_normal(normal_encoded);

    // Reconstruct world-space position of the surface.
    let world_pos = reconstruct_world_pos(full_uv, depth);

    // Build TBN basis for hemisphere sampling.
    let tbn = build_tbn(normal);

    // Per-pixel random seed derived from position and frame index.
    var seed = pcg_hash(u32(half_pixel.x) + u32(half_pixel.y) * 4096u + params.frame_index * 16777259u);

    // Ray-march N rays and accumulate indirect lighting.
    var gi_accum = vec3f(0.0);
    var valid_rays = 0u;

    for (var ray_i = 0u; ray_i < params.ray_count; ray_i++) {
        let ray_dir = cosine_hemisphere_sample(tbn, &seed);

        // cos(theta) is implicitly baked into cosine-weighted sampling,
        // so we weight uniformly (each sample already accounts for the
        // Lambert cosine term).

        // March in screen space.
        var hit_color = vec3f(0.0);
        var hit = false;

        // Step along the ray in world space, project to screen each step.
        let step_distance = params.max_distance / f32(params.max_steps);

        for (var step = 1u; step <= params.max_steps; step++) {
            let march_pos = world_pos + ray_dir * step_distance * f32(step);

            // Project to screen UV via the current frame's inverse VP
            // (we use the same inverse VP to reconstruct, so project using
            // the non-inverted VP — we reconstruct it from inv_view_proj).
            // Instead, project through previous frame VP for consistency
            // with the previous color buffer.
            let prev_uv = project_to_prev_uv(march_pos);

            // Out of screen bounds — stop this ray.
            if prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0 {
                break;
            }

            // Sample depth at the projected position (from previous frame
            // depth to match the previous color buffer).
            let sample_pixel = vec2i(prev_uv * vec2f(full_dims));
            let scene_depth = textureLoad(prev_depth_tex, sample_pixel, 0).r;

            // Reconstruct world-space position at the sampled depth.
            let scene_world_pos = reconstruct_world_pos(prev_uv, scene_depth);

            // Check intersection: ray position is behind the scene surface.
            let to_scene = scene_world_pos - march_pos;
            let dist_to_scene = length(to_scene);

            // Thickness threshold: proportional to step size for robustness.
            let thickness = step_distance * 2.0;

            if dist_to_scene < thickness {
                // We have a hit — sample the previous frame's color.
                hit_color = textureSampleLevel(prev_color_tex, linear_sampler, prev_uv, 0.0).rgb;
                hit = true;
                break;
            }
        }

        if hit {
            gi_accum += hit_color;
            valid_rays += 1u;
        }
    }

    // Average the accumulated radiance.
    var current_gi = vec3f(0.0);
    if valid_rays > 0u {
        current_gi = gi_accum / f32(valid_rays) * params.intensity;
    }

    // Temporal accumulation: blend with history.
    let half_uv = (vec2f(half_pixel) + 0.5) / params.screen_size;
    let history = textureSampleLevel(history_tex, linear_sampler, half_uv, 0.0).rgb;
    let blended = mix(current_gi, history, params.temporal_blend);

    textureStore(output, half_pixel, vec4f(blended, 1.0));
}
