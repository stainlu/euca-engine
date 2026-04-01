// Depth of Field compute shader.
//
// Two-pass approach:
//   Pass 0 (coc): Compute circle-of-confusion per pixel from depth.
//   Pass 1 (gather): Variable-radius disk blur weighted by CoC.
//
// Bindings (group 0):
//   @binding(0) params    -- uniform DofParams
//   @binding(1) color_tex -- input color texture (texture_2d<f32>)
//   @binding(2) depth_tex -- depth buffer (texture_depth_2d)
//   @binding(3) coc_tex   -- circle-of-confusion (storage, r16float, read_write)
//   @binding(4) output_tex-- output color (storage, rgba16float, write)

struct DofParams {
    resolution: vec2f,        // screen dimensions
    inv_resolution: vec2f,    // 1.0 / resolution
    focus_distance: f32,      // distance to focal plane (world units)
    aperture: f32,            // aperture diameter (controls blur amount)
    focal_length: f32,        // lens focal length (e.g. 0.05 for 50mm)
    max_blur_radius: f32,     // maximum CoC radius in pixels
    near_far: vec2f,          // x = near plane, y = far plane
    _pad: vec2f,
};

@group(0) @binding(0) var<uniform> params: DofParams;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var coc_tex: texture_storage_2d<r16float, read_write>;
@group(0) @binding(4) var output_tex: texture_storage_2d<rgba16float, write>;

// Linearise a reverse-Z depth value to view-space distance.
fn linearize_depth(d: f32) -> f32 {
    let near = params.near_far.x;
    let far = params.near_far.y;
    // Reverse-Z: near maps to 1.0, far maps to 0.0.
    // Linear depth = near * far / (near + d * (far - near))
    return near * far / (near + d * (far - near));
}

// Thin-lens CoC formula: coc = |1/focus - 1/depth| * aperture * focal_length
// Positive CoC = background blur (behind focus), negative = foreground blur.
fn compute_coc(linear_depth: f32) -> f32 {
    let inv_focus = 1.0 / max(params.focus_distance, 0.001);
    let inv_depth = 1.0 / max(linear_depth, 0.001);
    let raw_coc = (inv_focus - inv_depth) * params.aperture * params.focal_length;
    // Convert from world-space CoC to pixel-space and clamp.
    // Scale by resolution height for aspect-independent behavior.
    let pixel_coc = raw_coc * params.resolution.y;
    return clamp(pixel_coc, -params.max_blur_radius, params.max_blur_radius);
}

// ── Pass 0: CoC computation ────────────────────────────────────────────────
@compute @workgroup_size(8, 8)
fn coc_pass(@builtin(global_invocation_id) gid: vec3u) {
    let pixel = vec2i(gid.xy);
    let dims = vec2i(params.resolution);
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let depth = textureLoad(depth_tex, pixel, 0);
    let linear_depth = linearize_depth(depth);
    let coc = compute_coc(linear_depth);
    textureStore(coc_tex, pixel, vec4f(coc, 0.0, 0.0, 0.0));
}

// ── Pass 1: Gather blur ───────────────────────────────────────────────────
// Variable-radius disk blur based on CoC magnitude.
// Uses a fixed Poisson-disk sampling pattern scaled by the CoC.
// Separates near and far field to prevent background bleeding onto foreground.

// 16-tap Poisson disk pattern (normalised to unit circle).
const POISSON_DISK: array<vec2f, 16> = array<vec2f, 16>(
    vec2f(-0.94201624, -0.39906216),
    vec2f( 0.94558609, -0.76890725),
    vec2f(-0.09418410, -0.92938870),
    vec2f( 0.34495938,  0.29387760),
    vec2f(-0.91588581,  0.45771432),
    vec2f(-0.81544232, -0.87912464),
    vec2f(-0.38277543,  0.27676845),
    vec2f( 0.97484398,  0.75648379),
    vec2f( 0.44323325, -0.97511554),
    vec2f( 0.53742981, -0.47373420),
    vec2f(-0.26496911, -0.41893023),
    vec2f( 0.79197514,  0.19090188),
    vec2f(-0.24188840,  0.99706507),
    vec2f(-0.81409955,  0.91437590),
    vec2f( 0.19984126,  0.78641367),
    vec2f( 0.14383161, -0.14100790),
);

@compute @workgroup_size(8, 8)
fn gather_pass(@builtin(global_invocation_id) gid: vec3u) {
    let pixel = vec2i(gid.xy);
    let dims = vec2i(params.resolution);
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let center_coc = textureLoad(coc_tex, pixel).x;
    let center_color = textureLoad(color_tex, pixel, 0);
    let abs_coc = abs(center_coc);

    // If CoC is negligible, pass through (in-focus pixel).
    if abs_coc < 0.5 {
        textureStore(output_tex, pixel, center_color);
        return;
    }

    // Gather samples in a disk scaled by the CoC radius.
    var far_accum = vec4f(0.0);
    var far_weight = 0.0;
    var near_accum = vec4f(0.0);
    var near_weight = 0.0;

    for (var i = 0u; i < 16u; i++) {
        let offset = POISSON_DISK[i] * abs_coc;
        let sample_pos = vec2i(
            pixel.x + i32(round(offset.x)),
            pixel.y + i32(round(offset.y)),
        );
        let clamped = clamp(sample_pos, vec2i(0), dims - vec2i(1));

        let sample_color = textureLoad(color_tex, clamped, 0);
        let sample_coc = textureLoad(coc_tex, clamped).x;

        // Separate near (negative CoC) and far (positive CoC) fields.
        // Near-field samples bleed outward (use their own CoC as weight).
        // Far-field samples only contribute if they're actually behind focus.
        if sample_coc < 0.0 {
            // Near-field (foreground): weight by sample's blur radius.
            let w = smoothstep(0.0, 2.0, abs(sample_coc));
            near_accum += sample_color * w;
            near_weight += w;
        } else {
            // Far-field (background): weight by center's blur radius
            // to prevent sharp background bleeding onto near.
            let w = smoothstep(0.0, 2.0, min(abs_coc, abs(sample_coc)));
            far_accum += sample_color * w;
            far_weight += w;
        }
    }

    // Combine near and far fields.
    var result = center_color;

    if center_coc >= 0.0 {
        // Center pixel is background — use far-field blur.
        if far_weight > 0.0 {
            result = far_accum / far_weight;
        }
        // Blend in near-field on top (foreground bleeds over background).
        if near_weight > 0.0 {
            let near_color = near_accum / near_weight;
            let near_alpha = smoothstep(0.0, 2.0, near_weight / 16.0);
            result = mix(result, near_color, near_alpha);
        }
    } else {
        // Center pixel is foreground — use near-field blur.
        if near_weight > 0.0 {
            result = near_accum / near_weight;
        }
    }

    textureStore(output_tex, pixel, result);
}
