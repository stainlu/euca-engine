// Physically-Based Rendering (PBR) shader.
// Cook-Torrance BRDF with GGX distribution, Smith geometry, Schlick Fresnel.
// Supports: directional light with cascaded shadow maps, point lights, spot lights,
//           normal mapping, metallic-roughness workflow, AO, emissive, alpha modes.

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

struct InstanceData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
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
}

struct MaterialUniforms {
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
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;

@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var shadow_map: texture_depth_2d_array;
@group(1) @binding(2) var shadow_sampler: sampler_comparison;

@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var albedo_tex: texture_2d<f32>;
@group(2) @binding(2) var albedo_sampler: sampler;
@group(2) @binding(3) var normal_tex: texture_2d<f32>;
@group(2) @binding(4) var metallic_roughness_tex: texture_2d<f32>;
@group(2) @binding(5) var ao_tex: texture_2d<f32>;
@group(2) @binding(6) var emissive_tex: texture_2d<f32>;

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
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) iid: u32) -> VertexOutput {
    let model = instances[iid].model;
    let normal_mat = instances[iid].normal_matrix;
    var out: VertexOutput;
    let world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    out.clip_position = scene.camera_vp * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = normalize((normal_mat * vec4<f32>(in.normal, 0.0)).xyz);
    out.world_tangent = normalize((model * vec4<f32>(in.tangent, 0.0)).xyz);
    out.uv = in.uv;
    return out;
}

// ---------------------------------------------------------------------------
// PBR helper functions
// ---------------------------------------------------------------------------

const PI: f32 = 3.14159265359;

// GGX/Trowbridge-Reitz normal distribution function.
fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry function (single direction).
fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith's method combining two Schlick-GGX terms for view and light directions.
fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

// Schlick approximation for Fresnel reflectance.
fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ---------------------------------------------------------------------------
// L2 Spherical Harmonics evaluation (9 coefficients)
// ---------------------------------------------------------------------------

fn evaluate_sh(normal: vec3<f32>, sh: array<vec4<f32>, 9>) -> vec3<f32> {
    // L0
    var result = sh[0].xyz * 0.282095;
    // L1
    result += sh[1].xyz * 0.488603 * normal.y;
    result += sh[2].xyz * 0.488603 * normal.z;
    result += sh[3].xyz * 0.488603 * normal.x;
    // L2
    result += sh[4].xyz * 1.092548 * normal.x * normal.y;
    result += sh[5].xyz * 1.092548 * normal.y * normal.z;
    result += sh[6].xyz * 0.315392 * (3.0 * normal.z * normal.z - 1.0);
    result += sh[7].xyz * 1.092548 * normal.x * normal.z;
    result += sh[8].xyz * 0.546274 * (normal.x * normal.x - normal.y * normal.y);
    return max(result, vec3<f32>(0.0));
}

// ---------------------------------------------------------------------------
// Cascaded shadow mapping
// ---------------------------------------------------------------------------

fn shadow_factor_biased(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    let normal_bias = 0.05;
    let biased_pos = world_pos + world_normal * normal_bias;

    // Select the tightest cascade that contains this fragment.
    var cascade_idx = 0i;
    for (var ci = 0i; ci < 3; ci++) {
        let vp = scene.cascade_vps[ci];
        let clip = vp * vec4<f32>(biased_pos, 1.0);
        let ndc = clip.xyz / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
        if uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 {
            cascade_idx = ci;
            break;
        }
    }

    let vp = scene.cascade_vps[cascade_idx];
    let light_clip = vp * vec4<f32>(biased_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let current_depth = ndc.z;

    if shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 {
        return 1.0;
    }

    // 3x3 PCF (percentage-closer filtering).
    let texel_size = 1.0 / 2048.0;
    var shadow = 0.0;
    for (var x = -1i; x <= 1i; x++) {
        for (var y = -1i; y <= 1i; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow += textureSampleCompare(
                shadow_map, shadow_sampler,
                shadow_uv + offset, cascade_idx, current_depth
            );
        }
    }
    return shadow / 9.0;
}

// ---------------------------------------------------------------------------
// Fragment stage
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // --- Albedo & alpha ---
    let tex_color = textureSample(albedo_tex, albedo_sampler, in.uv);
    let albedo = material.albedo.rgb * tex_color.rgb;
    let alpha = material.albedo.a * tex_color.a;

    // Alpha cutoff (mask mode).
    if material.alpha_mode > 0.5 && material.alpha_mode < 1.5 {
        if alpha < material.alpha_cutoff {
            discard;
        }
    }

    // --- Metallic / roughness ---
    var metallic = material.metallic;
    var roughness = material.roughness;
    if material.has_metallic_roughness_tex > 0.5 {
        let mr_sample = textureSample(metallic_roughness_tex, albedo_sampler, in.uv);
        roughness = material.roughness * mr_sample.g;
        metallic = material.metallic * mr_sample.b;
    }
    roughness = max(roughness, 0.04);

    // --- Normal mapping ---
    var N: vec3<f32>;
    if material.has_normal_map > 0.5 {
        let sampled = textureSample(normal_tex, albedo_sampler, in.uv).rgb;
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
    let shadow = shadow_factor_biased(in.world_pos, N);
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
        if pl_dist > pl_range {
            continue;
        }

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
        if sl_dist > sl_range {
            continue;
        }

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
    if material.has_ao_tex > 0.5 {
        let ao = textureSample(ao_tex, albedo_sampler, in.uv).r;
        ambient = ambient * ao;
    }
    var color = ambient + Lo;

    // --- Emissive ---
    var emissive_color = material.emissive;
    if material.has_emissive_tex > 0.5 {
        let emissive_sample = textureSample(emissive_tex, albedo_sampler, in.uv).rgb;
        emissive_color = emissive_color * emissive_sample;
    }
    color = color + emissive_color;

    // --- Output ---
    var out_alpha = 1.0;
    if material.alpha_mode > 1.5 {
        out_alpha = alpha;
    }
    return vec4<f32>(color, out_alpha);
}
