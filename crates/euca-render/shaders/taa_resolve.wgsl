// Enhanced Temporal Anti-Aliasing resolve shader.
//
// Uses velocity-buffer reprojection, variance-based neighborhood clamping in
// YCoCg space, and disocclusion detection for robust temporal accumulation.

struct TaaParams {
    inv_vp: mat4x4f,       // current frame inverse view-projection
    prev_vp: mat4x4f,      // previous frame view-projection (for depth disocclusion)
    jitter: vec2f,          // current frame sub-pixel jitter (clip space)
    resolution: vec2f,      // screen dimensions (width, height)
    blend_factor: f32,      // how much of current frame to blend in (0.05-0.1)
    variance_gamma: f32,    // variance clamp tightness (1.0 = tight, 2.0 = loose)
    depth_threshold: f32,   // disocclusion depth threshold (view-space)
    _pad: f32,
};

@group(0) @binding(0) var current_frame: texture_2d<f32>;
@group(0) @binding(1) var history_frame: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var<uniform> params: TaaParams;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var linear_sampler: sampler;
@group(0) @binding(6) var velocity_tex: texture_2d<f32>;

// Convert linear RGB to YCoCg for perceptually-correct neighborhood clamping.
fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let y  = dot(rgb, vec3<f32>(0.25, 0.5, 0.25));
    let co = dot(rgb, vec3<f32>(0.5, 0.0, -0.5));
    let cg = dot(rgb, vec3<f32>(-0.25, 0.5, -0.25));
    return vec3<f32>(y, co, cg);
}

// Convert YCoCg back to linear RGB.
fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let y = ycocg.x; let co = ycocg.y; let cg = ycocg.z;
    return vec3<f32>(y + co - cg, y + cg, y - co - cg);
}

// Reconstruct world position from depth + inverse VP (used for disocclusion).
fn reconstruct_world_pos(uv: vec2f, depth: f32) -> vec4f {
    let ndc = vec4f(uv * 2.0 - 1.0, depth, 1.0);
    let ndc_flipped = vec4f(ndc.x, -ndc.y, ndc.z, 1.0);
    let world = params.inv_vp * ndc_flipped;
    return world / world.w;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pixel = vec2i(id.xy);
    let dims = vec2i(params.resolution);
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let uv = (vec2f(id.xy) + 0.5) / params.resolution;

    // 1. Sample current frame
    let current_color = textureLoad(current_frame, pixel, 0);

    // 2. Velocity-based reprojection: previous UV = current UV - velocity
    let velocity = textureLoad(velocity_tex, pixel, 0).xy;
    let prev_uv = uv - velocity;

    // 3. Disocclusion detection
    let current_depth = textureLoad(depth_tex, pixel, 0);
    var is_disoccluded = false;

    // Check if reprojected UV is off-screen
    if prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0 {
        is_disoccluded = true;
    }

    // Check depth continuity at reprojected position
    if !is_disoccluded {
        let prev_pixel = vec2i(prev_uv * params.resolution);
        let clamped_prev = clamp(prev_pixel, vec2i(0), dims - vec2i(1));
        let prev_depth = textureLoad(depth_tex, clamped_prev, 0);
        // Compare linearised depth difference against threshold.
        // For reverse-Z, closer objects have larger depth values.
        let depth_diff = abs(current_depth - prev_depth);
        if depth_diff > params.depth_threshold {
            is_disoccluded = true;
        }
    }

    // 4. Variance-based neighborhood clamping (3x3 in YCoCg space)
    //    Compute mean and variance, clamp history to [mean - gamma*sigma, mean + gamma*sigma].
    var moment1 = vec3<f32>(0.0);
    var moment2 = vec3<f32>(0.0);
    var neighbor_sum = vec4<f32>(0.0);
    var sample_count = 0.0;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor = vec2i(pixel.x + dx, pixel.y + dy);
            if neighbor.x >= 0 && neighbor.x < dims.x && neighbor.y >= 0 && neighbor.y < dims.y {
                let n_color = textureLoad(current_frame, neighbor, 0);
                let n_ycocg = rgb_to_ycocg(n_color.rgb);
                moment1 += n_ycocg;
                moment2 += n_ycocg * n_ycocg;
                neighbor_sum += n_color;
                sample_count += 1.0;
            }
        }
    }

    let mean = moment1 / sample_count;
    let variance = moment2 / sample_count - mean * mean;
    let sigma = sqrt(max(variance, vec3<f32>(0.0)));
    let gamma = params.variance_gamma;
    let clamp_min = mean - gamma * sigma;
    let clamp_max = mean + gamma * sigma;

    // 5. If disoccluded, use neighborhood average (no valid history)
    if is_disoccluded {
        let avg_color = neighbor_sum / sample_count;
        // Blend heavily toward current frame for disoccluded pixels
        let result = mix(avg_color, current_color, 0.8);
        textureStore(output, pixel, result);
        return;
    }

    // 6. Sample history at reprojected position (bilinear via sampler)
    let history_color = textureSampleLevel(history_frame, linear_sampler, prev_uv, 0.0);

    // 7. Clamp history to variance-based AABB in YCoCg space
    let history_ycocg = rgb_to_ycocg(history_color.rgb);
    let clamped_ycocg = clamp(history_ycocg, clamp_min, clamp_max);
    let clamped_history = vec4f(ycocg_to_rgb(clamped_ycocg), history_color.a);

    // 8. Adaptive blend factor: increase blend toward current when clamping is aggressive
    let clamp_distance = length(history_ycocg - clamped_ycocg);
    let adaptive_blend = clamp(params.blend_factor + clamp_distance * 0.5, params.blend_factor, 0.5);

    // 9. Blend: mostly history, adaptive fraction of current
    let result = mix(clamped_history, current_color, adaptive_blend);
    textureStore(output, pixel, result);
}
