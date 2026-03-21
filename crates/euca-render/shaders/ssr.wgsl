// Screen-Space Reflections (SSR) shader.
//
// Reads depth, normal+material (G-buffer RT1), and the color buffer.
// For each pixel: if the surface is metallic and roughness is below the
// threshold, reflect the view ray using the surface normal and ray-march
// in screen space to find an intersection with the depth buffer.
//
// Output: reflection color blended with a fade at screen edges and distance.

struct SsrUniforms {
    inv_projection: mat4x4<f32>,
    projection: mat4x4<f32>,
    // x = max_steps, y = step_size, z = max_distance, w = thickness
    params0: vec4<f32>,
    // x = roughness_threshold, y = screen_width, z = screen_height, w = unused
    params1: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_material_tex: texture_2d<f32>;
@group(0) @binding(2) var color_tex: texture_2d<f32>;
@group(0) @binding(3) var tex_sampler: sampler;
@group(0) @binding(4) var<uniform> ssr: SsrUniforms;

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    let x = f32(i32(id) / 2) * 4.0 - 1.0;
    let y = f32(i32(id) % 2) * 4.0 - 1.0;
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}

// Reconstruct view-space position from depth and UV.
fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let ndc_fixed = vec4<f32>(ndc.x, -ndc.y, ndc.z, 1.0);
    let view_h = ssr.inv_projection * ndc_fixed;
    return view_h.xyz / view_h.w;
}

// Project a view-space position back to UV coordinates.
fn project_to_uv(view_pos: vec3<f32>) -> vec3<f32> {
    let clip = ssr.projection * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    return vec3<f32>(uv, ndc.z);
}

// Decode octahedral normal from two channels (matching G-buffer RT1 encoding).
fn decode_octahedral_normal(enc: vec2<f32>) -> vec3<f32> {
    let f = enc * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.0);
    if n.x >= 0.0 {
        n.x -= t;
    } else {
        n.x += t;
    }
    if n.y >= 0.0 {
        n.y -= t;
    } else {
        n.y += t;
    }
    return normalize(n);
}

// Fade reflections near screen edges to avoid hard cutoffs.
fn screen_edge_fade(uv: vec2<f32>) -> f32 {
    let edge = smoothstep(0.0, 0.05, uv.x)
             * smoothstep(0.0, 0.05, uv.y)
             * (1.0 - smoothstep(0.95, 1.0, uv.x))
             * (1.0 - smoothstep(0.95, 1.0, uv.y));
    return edge;
}

// Sky/ambient color used when ray misses.
const SKY_COLOR: vec3<f32> = vec3<f32>(0.15, 0.3, 0.65);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let max_steps = u32(ssr.params0.x);
    let step_size = ssr.params0.y;
    let max_distance = ssr.params0.z;
    let thickness = ssr.params0.w;
    let roughness_threshold = ssr.params1.x;

    // Sample G-buffer: normal (xy octahedral), metallic (z), roughness (w).
    let gbuffer = textureSample(normal_material_tex, tex_sampler, in.uv);
    let metallic = gbuffer.z;
    let roughness = gbuffer.w;

    // Skip non-reflective surfaces: not metallic or too rough.
    if metallic < 0.01 || roughness >= roughness_threshold {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let depth = textureSample(depth_tex, tex_sampler, in.uv).r;
    // Skip sky pixels (depth at or beyond far plane).
    if depth >= 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Reconstruct view-space position and normal.
    let view_pos = view_pos_from_depth(in.uv, depth);
    let normal_view = decode_octahedral_normal(gbuffer.xy);

    // Compute reflected ray direction in view space.
    let view_dir = normalize(view_pos);
    let reflect_dir = reflect(view_dir, normal_view);

    // Ray-march in view space, checking screen-space depth at each step.
    var ray_pos = view_pos;
    var hit = false;
    var hit_uv = vec2<f32>(0.0);
    var march_distance = 0.0;

    for (var i = 0u; i < max_steps; i++) {
        ray_pos += reflect_dir * step_size;
        march_distance += step_size;

        if march_distance > max_distance {
            break;
        }

        // Project current ray position to screen space.
        let projected = project_to_uv(ray_pos);
        let sample_uv = projected.xy;
        let ray_depth = projected.z;

        // Out of screen bounds -- stop marching.
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            break;
        }

        // Sample the depth buffer at the projected position.
        let scene_depth = textureSample(depth_tex, tex_sampler, sample_uv).r;
        let scene_view_pos = view_pos_from_depth(sample_uv, scene_depth);
        let depth_diff = ray_pos.z - scene_view_pos.z;

        // Intersection: ray is behind scene geometry (within thickness tolerance).
        if depth_diff > 0.0 && depth_diff < thickness {
            hit = true;
            hit_uv = sample_uv;
            break;
        }
    }

    if hit {
        let reflection_color = textureSample(color_tex, tex_sampler, hit_uv).rgb;
        let edge_fade = screen_edge_fade(hit_uv);
        let distance_fade = 1.0 - saturate(march_distance / max_distance);
        let roughness_fade = 1.0 - saturate(roughness / roughness_threshold);
        let alpha = metallic * edge_fade * distance_fade * roughness_fade;
        return vec4<f32>(reflection_color, alpha);
    } else {
        // Miss: subtle sky-colored ambient reflection for highly metallic surfaces.
        let ambient_strength = metallic * 0.1 * (1.0 - saturate(roughness / roughness_threshold));
        return vec4<f32>(SKY_COLOR * ambient_strength, ambient_strength);
    }
}
