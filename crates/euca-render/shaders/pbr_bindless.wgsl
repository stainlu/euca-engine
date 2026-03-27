// Bindless PBR shader variant.
// Identical Cook-Torrance BRDF to pbr.wgsl, but material uniforms and textures
// are accessed via storage buffer array + texture binding array, eliminating
// per-batch bind group switches.
//
// Requires: TEXTURE_BINDING_ARRAY + NON_UNIFORM_INDEXING GPU features.

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

struct InstanceData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    material_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct PointLightData {
    position: vec4<f32>,
    color: vec4<f32>,
}

struct SpotLightData {
    position: vec4<f32>,
    direction: vec4<f32>,
    color: vec4<f32>,
    cone: vec4<f32>,
}

struct SceneUniforms {
    camera_pos: vec4<f32>,
    light_direction: vec4<f32>,
    light_color: vec4<f32>,
    ambient_color: vec4<f32>,
    camera_vp: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    inv_vp: mat4x4<f32>,
    cascade_vps: array<mat4x4<f32>, 3>,
    cascade_splits: vec4<f32>,
    point_lights: array<PointLightData, 4>,
    spot_lights: array<SpotLightData, 2>,
    num_point_lights: vec4<f32>,
    num_spot_lights: vec4<f32>,
    probe_sh: array<vec4<f32>, 9>,
    probe_enabled: vec4<f32>,
    shadow_params: vec4<f32>,
    ibl_params: vec4<f32>,
}

// Bindless material: uniform data + texture indices into the binding array.
// 96 bytes, 16-byte aligned for storage buffer array stride.
struct BindlessMaterial {
    albedo: vec4<f32>,
    metallic: f32,
    roughness: f32,
    has_normal_map: f32,
    has_metallic_roughness_tex: f32,
    emissive: vec3<f32>,
    has_emissive_tex: f32,
    has_ao_tex: f32,
    alpha_mode: f32,
    alpha_cutoff: f32,
    _pad0: f32,
    albedo_tex_idx: u32,
    normal_tex_idx: u32,
    metallic_roughness_tex_idx: u32,
    ao_tex_idx: u32,
    emissive_tex_idx: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

// Group 0: Instance data (per-entity model matrix + material_id)
@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;

// Group 1: Scene uniforms + shadow maps + IBL (unchanged from pbr.wgsl)
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var shadow_map: texture_depth_2d_array;
@group(1) @binding(2) var shadow_sampler: sampler_comparison;
@group(1) @binding(3) var shadow_depth_sampler: sampler;
@group(1) @binding(4) var ibl_irradiance_map: texture_cube<f32>;
@group(1) @binding(5) var ibl_specular_map: texture_cube<f32>;
@group(1) @binding(6) var ibl_brdf_lut: texture_2d<f32>;
@group(1) @binding(7) var ibl_sampler: sampler;

// Group 2: Bindless materials (single bind group for ALL materials + textures)
@group(2) @binding(0) var<storage, read> materials: array<BindlessMaterial>;
@group(2) @binding(1) var material_sampler: sampler;
@group(2) @binding(2) var textures: binding_array<texture_2d<f32>>;

// Sentinel value indicating "no texture" (material uses uniform color only).
const NO_TEXTURE: u32 = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Vertex stage
// ---------------------------------------------------------------------------

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) material_id: u32,
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) iid: u32) -> VertexOutput {
    let inst = instances[iid];
    var out: VertexOutput;
    let world_pos = (inst.model * vec4<f32>(in.position, 1.0)).xyz;
    out.clip_position = scene.camera_vp * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = normalize((inst.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);
    out.world_tangent = normalize((inst.model * vec4<f32>(in.tangent, 0.0)).xyz);
    out.uv = in.uv;
    out.material_id = inst.material_id;
    return out;
}

// ---------------------------------------------------------------------------
// PBR helper functions (identical to pbr.wgsl)
// ---------------------------------------------------------------------------

const PI: f32 = 3.14159265359;

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn evaluate_sh(normal: vec3<f32>, sh: array<vec4<f32>, 9>) -> vec3<f32> {
    var result = sh[0].xyz * 0.282095;
    result += sh[1].xyz * 0.488603 * normal.y;
    result += sh[2].xyz * 0.488603 * normal.z;
    result += sh[3].xyz * 0.488603 * normal.x;
    result += sh[4].xyz * 1.092548 * normal.x * normal.y;
    result += sh[5].xyz * 1.092548 * normal.y * normal.z;
    result += sh[6].xyz * 0.315392 * (3.0 * normal.z * normal.z - 1.0);
    result += sh[7].xyz * 1.092548 * normal.x * normal.z;
    result += sh[8].xyz * 0.546274 * (normal.x * normal.x - normal.y * normal.y);
    return max(result, vec3<f32>(0.0));
}

// ---------------------------------------------------------------------------
// Shadow mapping (identical to pbr.wgsl)
// ---------------------------------------------------------------------------

const SHADOW_MAP_SIZE: f32 = 2048.0;

const POISSON_16: array<vec2<f32>, 16> = array(
    vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2(0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845), vec2(0.97484398,  0.75648379),
    vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2(0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
    vec2(0.19984126,  0.78641367), vec2(0.14383161, -0.14100790),
);

fn interleaved_gradient_noise(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

fn rotate_poisson(sample: vec2<f32>, angle: f32) -> vec2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec2(sample.x * c - sample.y * s, sample.x * s + sample.y * c);
}

fn find_blocker(shadow_uv: vec2<f32>, receiver_depth: f32, search_radius: f32,
                cascade_index: i32, rotation: f32) -> vec2<f32> {
    var blocker_sum = 0.0;
    var blocker_count = 0.0;
    for (var i = 0; i < 16; i++) {
        let offset = rotate_poisson(POISSON_16[i], rotation) * search_radius;
        let sample_uv = shadow_uv + offset;
        let shadow_depth = textureSampleLevel(
            shadow_map, shadow_depth_sampler, sample_uv, cascade_index, 0
        );
        if shadow_depth < receiver_depth {
            blocker_sum += shadow_depth;
            blocker_count += 1.0;
        }
    }
    return vec2(blocker_sum / max(blocker_count, 1.0), blocker_count);
}

fn pcss_shadow(shadow_uv: vec2<f32>, receiver_depth: f32, cascade_index: i32,
               pixel_pos: vec2<f32>, light_size: f32) -> f32 {
    let rotation = interleaved_gradient_noise(pixel_pos) * 6.28318;
    let search_radius = light_size / SHADOW_MAP_SIZE;
    let blocker = find_blocker(shadow_uv, receiver_depth, search_radius, cascade_index, rotation);
    if blocker.y < 1.0 { return 1.0; }
    let penumbra = (receiver_depth - blocker.x) / blocker.x * light_size;
    let filter_radius = penumbra / SHADOW_MAP_SIZE;
    let clamped_radius = clamp(filter_radius, 1.0 / SHADOW_MAP_SIZE, search_radius * 2.0);
    var shadow = 0.0;
    for (var i = 0; i < 16; i++) {
        let offset = rotate_poisson(POISSON_16[i], rotation) * clamped_radius;
        shadow += textureSampleCompare(
            shadow_map, shadow_sampler,
            shadow_uv + offset, cascade_index, receiver_depth
        );
    }
    return shadow / 16.0;
}

fn shadow_bias_position(world_pos: vec3<f32>, world_normal: vec3<f32>,
                        light_dir: vec3<f32>, cascade_idx: i32) -> vec3<f32> {
    let normal_bias_scale = scene.shadow_params.y;
    let slope_bias_scale = scene.shadow_params.z;
    let cascade_bias_scale = scene.shadow_params.w;
    let NdotL = max(dot(world_normal, light_dir), 0.001);
    let slope_factor = sqrt(1.0 - NdotL * NdotL) / NdotL;
    let base_bias = normal_bias_scale + slope_bias_scale * slope_factor;
    let effective_bias = base_bias * (1.0 + f32(cascade_idx) * cascade_bias_scale);
    return world_pos + world_normal * effective_bias;
}

fn cascade_project(world_pos: vec3<f32>, cascade_idx: i32) -> vec3<f32> {
    let vp = scene.cascade_vps[cascade_idx];
    let clip = vp * vec4<f32>(world_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    return vec3<f32>(uv.x, uv.y, ndc.z);
}

const CASCADE_COUNT: i32 = 3;
const CASCADE_TRANSITION_WIDTH: f32 = 0.1;

fn shadow_factor_biased(world_pos: vec3<f32>, world_normal: vec3<f32>,
                        pixel_pos: vec2<f32>) -> f32 {
    let light_dir = normalize(-scene.light_direction.xyz);
    let light_size = scene.shadow_params.x;
    var cascade_idx = 0i;
    for (var ci = 0i; ci < CASCADE_COUNT; ci++) {
        let biased = shadow_bias_position(world_pos, world_normal, light_dir, ci);
        let proj = cascade_project(biased, ci);
        if proj.x >= 0.0 && proj.x <= 1.0 && proj.y >= 0.0 && proj.y <= 1.0 {
            cascade_idx = ci;
            break;
        }
    }
    let biased_primary = shadow_bias_position(world_pos, world_normal, light_dir, cascade_idx);
    let proj_primary = cascade_project(biased_primary, cascade_idx);
    let uv_primary = vec2<f32>(proj_primary.x, proj_primary.y);
    if uv_primary.x < 0.0 || uv_primary.x > 1.0 || uv_primary.y < 0.0 || uv_primary.y > 1.0 {
        return 1.0;
    }
    let shadow_primary = pcss_shadow(uv_primary, proj_primary.z, cascade_idx, pixel_pos, light_size);
    let edge_dist = min(
        min(uv_primary.x, 1.0 - uv_primary.x),
        min(uv_primary.y, 1.0 - uv_primary.y)
    );
    let next_idx = cascade_idx + 1;
    if edge_dist < CASCADE_TRANSITION_WIDTH && next_idx < CASCADE_COUNT {
        let biased_next = shadow_bias_position(world_pos, world_normal, light_dir, next_idx);
        let proj_next = cascade_project(biased_next, next_idx);
        let uv_next = vec2<f32>(proj_next.x, proj_next.y);
        if uv_next.x >= 0.0 && uv_next.x <= 1.0 && uv_next.y >= 0.0 && uv_next.y <= 1.0 {
            let shadow_next = pcss_shadow(uv_next, proj_next.z, next_idx, pixel_pos, light_size);
            let blend = smoothstep(0.0, CASCADE_TRANSITION_WIDTH, edge_dist);
            return mix(shadow_next, shadow_primary, blend);
        }
    }
    return shadow_primary;
}

// ---------------------------------------------------------------------------
// Bindless texture sampling helpers
// ---------------------------------------------------------------------------

fn sample_texture_or_white(tex_idx: u32, uv: vec2<f32>) -> vec4<f32> {
    if tex_idx == NO_TEXTURE {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    return textureSample(textures[tex_idx], material_sampler, uv);
}

fn sample_texture_or_flat_normal(tex_idx: u32, uv: vec2<f32>) -> vec3<f32> {
    if tex_idx == NO_TEXTURE {
        return vec3<f32>(0.5, 0.5, 1.0);
    }
    return textureSample(textures[tex_idx], material_sampler, uv).rgb;
}

// ---------------------------------------------------------------------------
// Fragment stage (bindless variant)
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let mat = materials[in.material_id];

    // --- Albedo & alpha ---
    let tex_color = sample_texture_or_white(mat.albedo_tex_idx, in.uv);
    let albedo = mat.albedo.rgb * tex_color.rgb;
    let alpha = mat.albedo.a * tex_color.a;

    if mat.alpha_mode > 0.5 && mat.alpha_mode < 1.5 {
        if alpha < mat.alpha_cutoff {
            discard;
        }
    }

    // --- Metallic / roughness ---
    var metallic = mat.metallic;
    var roughness = mat.roughness;
    if mat.has_metallic_roughness_tex > 0.5 {
        let mr_sample = sample_texture_or_white(mat.metallic_roughness_tex_idx, in.uv);
        roughness = mat.roughness * mr_sample.g;
        metallic = mat.metallic * mr_sample.b;
    }
    roughness = max(roughness, 0.04);

    // --- Normal mapping ---
    var N: vec3<f32>;
    if mat.has_normal_map > 0.5 {
        let sampled = sample_texture_or_flat_normal(mat.normal_tex_idx, in.uv);
        let tangent_normal = sampled * 2.0 - 1.0;
        let T = normalize(in.world_tangent);
        let N_vert = normalize(in.world_normal);
        let B = cross(N_vert, T);
        N = normalize(T * tangent_normal.x + B * tangent_normal.y + N_vert * tangent_normal.z);
    } else {
        N = normalize(in.world_normal);
    }

    let V = normalize(scene.camera_pos.xyz - in.world_pos);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    // --- Directional light (Cook-Torrance) ---
    let L = normalize(-scene.light_direction.xyz);
    let H = normalize(V + L);
    let NdotL = max(dot(N, L), 0.0);
    let light_intensity = scene.light_color.w;
    let radiance = scene.light_color.rgb * light_intensity;

    let D = distribution_ggx(N, H, roughness);
    let G = geometry_smith(N, V, L, roughness);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let numerator = D * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    let specular = numerator / denominator;

    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);
    let shadow = shadow_factor_biased(in.world_pos, N, in.clip_position.xy);
    var Lo = (kD * albedo / PI + specular) * radiance * NdotL * shadow;

    // --- Point lights ---
    let n_point = i32(scene.num_point_lights.x);
    for (var pi = 0; pi < n_point; pi++) {
        let pl = scene.point_lights[pi];
        let pl_pos = pl.position.xyz;
        let pl_range = pl.position.w;
        let pl_color = pl.color.rgb;
        let pl_intensity = pl.color.a;
        let pl_dir = pl_pos - in.world_pos;
        let pl_dist = length(pl_dir);
        if pl_dist > pl_range { continue; }
        let pl_L = pl_dir / pl_dist;
        let pl_NdotL = max(dot(N, pl_L), 0.0);
        let pl_attenuation = 1.0 / (pl_dist * pl_dist + 0.01);
        let pl_falloff = saturate(1.0 - pl_dist / pl_range);
        let pl_radiance = pl_color * pl_intensity * pl_attenuation * pl_falloff;
        Lo += (kD * albedo / PI) * pl_radiance * pl_NdotL;
    }

    // --- Spot lights ---
    let n_spot = i32(scene.num_spot_lights.x);
    for (var si = 0; si < n_spot; si++) {
        let sl = scene.spot_lights[si];
        let sl_pos = sl.position.xyz;
        let sl_range = sl.position.w;
        let sl_dir_norm = normalize(sl.direction.xyz);
        let sl_color = sl.color.rgb;
        let sl_intensity = sl.color.a;
        let sl_inner_cos = sl.cone.x;
        let sl_outer_cos = sl.cone.y;
        let sl_to_frag = in.world_pos - sl_pos;
        let sl_dist = length(sl_to_frag);
        if sl_dist > sl_range { continue; }
        let sl_L = -normalize(sl_to_frag);
        let sl_NdotL = max(dot(N, sl_L), 0.0);
        let sl_cos_theta = dot(normalize(sl_to_frag), sl_dir_norm);
        let sl_cone_atten = saturate((sl_cos_theta - sl_outer_cos) / (sl_inner_cos - sl_outer_cos));
        let sl_dist_atten = 1.0 / (sl_dist * sl_dist + 0.01);
        let sl_falloff = saturate(1.0 - sl_dist / sl_range);
        let sl_radiance = sl_color * sl_intensity * sl_dist_atten * sl_falloff * sl_cone_atten;
        Lo += (kD * albedo / PI) * sl_radiance * sl_NdotL;
    }

    // --- Ambient (SH probes or flat) ---
    var ambient: vec3<f32>;
    if scene.probe_enabled.x > 0.5 {
        ambient = evaluate_sh(N, scene.probe_sh) * albedo;
    } else {
        let ambient_intensity = scene.ambient_color.w;
        ambient = scene.ambient_color.rgb * ambient_intensity * albedo;
    }
    if mat.has_ao_tex > 0.5 {
        let ao = sample_texture_or_white(mat.ao_tex_idx, in.uv).r;
        ambient = ambient * ao;
    }
    var color = ambient + Lo;

    // --- IBL ---
    let n_dot_v = max(dot(N, V), 0.0);
    if (scene.ibl_params.x > 0.5) {
        let ibl_intensity = scene.ibl_params.y;
        let F_ibl = fresnel_schlick_roughness(n_dot_v, F0, roughness);
        let kS_ibl = F_ibl;
        let kD_ibl = (1.0 - kS_ibl) * (1.0 - metallic);
        let irradiance = textureSample(ibl_irradiance_map, ibl_sampler, N).rgb;
        let diffuse_ibl = irradiance * albedo * kD_ibl;
        let R = reflect(-V, N);
        let max_mip = 4.0;
        let specular_color = textureSampleLevel(ibl_specular_map, ibl_sampler, R, roughness * max_mip).rgb;
        let brdf = textureSample(ibl_brdf_lut, ibl_sampler, vec2(n_dot_v, roughness)).rg;
        let specular_ibl = specular_color * (F_ibl * brdf.x + brdf.y);
        color += (diffuse_ibl + specular_ibl) * ibl_intensity;
    }

    // --- Emissive ---
    var emissive_color = mat.emissive;
    if mat.has_emissive_tex > 0.5 {
        let emissive_sample = sample_texture_or_white(mat.emissive_tex_idx, in.uv).rgb;
        emissive_color = emissive_color * emissive_sample;
    }
    color = color + emissive_color;

    // --- Output ---
    var out_alpha = 1.0;
    if mat.alpha_mode > 1.5 {
        out_alpha = alpha;
    }
    return vec4<f32>(color, out_alpha);
}
