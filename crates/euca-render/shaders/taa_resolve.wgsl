// Temporal Anti-Aliasing resolve shader.
//
// Blends the current jittered frame with the accumulated history buffer.
// Uses neighborhood clamping to prevent ghosting on moving objects.

struct TaaParams {
    inv_vp: mat4x4f,       // current frame inverse view-projection
    prev_vp: mat4x4f,      // previous frame view-projection (for reprojection)
    jitter: vec2f,          // current frame sub-pixel jitter (clip space)
    resolution: vec2f,      // screen dimensions (width, height)
    blend_factor: f32,      // how much of current frame to blend in (0.05–0.1)
    _pad: vec3f,
};

@group(0) @binding(0) var current_frame: texture_2d<f32>;
@group(0) @binding(1) var history_frame: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var<uniform> params: TaaParams;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var linear_sampler: sampler;

// Reconstruct world position from depth + inverse VP.
fn reconstruct_world_pos(uv: vec2f, depth: f32) -> vec4f {
    let ndc = vec4f(uv * 2.0 - 1.0, depth, 1.0);
    // Flip Y: NDC Y is up, UV Y is down
    let ndc_flipped = vec4f(ndc.x, -ndc.y, ndc.z, 1.0);
    let world = params.inv_vp * ndc_flipped;
    return world / world.w;
}

// Reproject world position to previous frame UV.
fn reproject_uv(world_pos: vec4f) -> vec2f {
    let prev_clip = params.prev_vp * world_pos;
    let prev_ndc = prev_clip.xy / prev_clip.w;
    // NDC → UV, flip Y back
    return vec2f(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pixel = vec2i(id.xy);
    let dims = vec2i(params.resolution);
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let uv = (vec2f(id.xy) + 0.5) / params.resolution;

    // 1. Sample current frame (unjittered UV for neighborhood clamping)
    let current_color = textureLoad(current_frame, pixel, 0);

    // 2. Reconstruct world position from depth
    let depth = textureLoad(depth_tex, pixel, 0);
    let world_pos = reconstruct_world_pos(uv, depth);

    // 3. Reproject to previous frame
    let prev_uv = reproject_uv(world_pos);

    // 4. Sample history at reprojected position (bilinear via sampler)
    var history_color: vec4f;
    if prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0 {
        history_color = textureSampleLevel(history_frame, linear_sampler, prev_uv, 0.0);
    } else {
        // Off-screen — use current frame only
        textureStore(output, pixel, current_color);
        return;
    }

    // 5. Neighborhood clamping (3×3 min/max AABB of current frame)
    //    Prevents ghosting by constraining history to plausible color range.
    var neighborhood_min = current_color.rgb;
    var neighborhood_max = current_color.rgb;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor = vec2i(pixel.x + dx, pixel.y + dy);
            if neighbor.x >= 0 && neighbor.x < dims.x && neighbor.y >= 0 && neighbor.y < dims.y {
                let n = textureLoad(current_frame, neighbor, 0).rgb;
                neighborhood_min = min(neighborhood_min, n);
                neighborhood_max = max(neighborhood_max, n);
            }
        }
    }
    let clamped_history = vec4f(
        clamp(history_color.rgb, neighborhood_min, neighborhood_max),
        history_color.a,
    );

    // 6. Blend: mostly history, small fraction of current
    let result = mix(clamped_history, current_color, params.blend_factor);
    textureStore(output, pixel, result);
}
