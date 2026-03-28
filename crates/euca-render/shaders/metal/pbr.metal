// Physically-Based Rendering (PBR) shader — Metal Shading Language.
// Cook-Torrance BRDF with GGX distribution, Smith geometry, Schlick Fresnel.
// Supports: directional light with cascaded shadow maps, point lights, spot lights,
//           normal mapping, metallic-roughness workflow, AO, emissive, alpha modes.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Structures — layouts must match the Rust #[repr(C)] and WGSL definitions.
// ---------------------------------------------------------------------------

struct InstanceData {
    float4x4 model;
    float4x4 normal_matrix;
    uint      material_id;
    uint      _pad0;
    uint      _pad1;
    uint      _pad2;
};

struct PointLightData {
    float4 position;  // xyz = position, w = range
    float4 color;     // xyz = color, w = intensity
};

struct SpotLightData {
    float4 position;   // xyz = position, w = range
    float4 direction;  // xyz = direction (unnormalized)
    float4 color;      // xyz = color, w = intensity
    float4 cone;       // x = inner_cos, y = outer_cos
};

struct SceneUniforms {
    float4   camera_pos;
    float4   light_direction;
    float4   light_color;
    float4   ambient_color;
    float4x4 camera_vp;
    float4x4 light_vp;
    float4x4 inv_vp;
    float4x4 cascade_vps[3];
    float4   cascade_splits;
    PointLightData point_lights[4];
    SpotLightData  spot_lights[2];
    float4   num_point_lights;
    float4   num_spot_lights;
    float4   probe_sh[9];
    float4   probe_enabled;
    float4   shadow_params;
    float4   ibl_params;
};

struct MaterialUniforms {
    float4         albedo;
    float          metallic;
    float          roughness;
    float          has_normal_map;
    float          has_metallic_roughness_tex;
    packed_float3  emissive;              // packed to match Rust [f32; 3] (12 bytes, align 4)
    float          has_emissive_tex;
    float          has_ao_tex;
    float          alpha_mode;
    float          alpha_cutoff;
    float          _pad;
};

// ---------------------------------------------------------------------------
// Vertex I/O
// ---------------------------------------------------------------------------

struct VertexInput {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float3 tangent  [[attribute(2)]];
    float2 uv       [[attribute(3)]];
};

struct VertexOutput {
    float4 clip_position [[position]];
    float3 world_pos;
    float3 world_normal;
    float3 world_tangent;
    float2 uv;
};

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

vertex VertexOutput vertex_main(
    VertexInput                       in       [[stage_in]],
    uint                              iid      [[instance_id]],
    const device InstanceData*        instances [[buffer(0)]],
    constant SceneUniforms&           scene    [[buffer(1)]]
) {
    float4x4 model      = instances[iid].model;
    float4x4 normal_mat = instances[iid].normal_matrix;

    VertexOutput out;
    float3 world_pos    = (model * float4(in.position, 1.0)).xyz;
    out.clip_position   = scene.camera_vp * float4(world_pos, 1.0);
    out.world_pos       = world_pos;
    out.world_normal    = normalize((normal_mat * float4(in.normal, 0.0)).xyz);
    out.world_tangent   = normalize((model * float4(in.tangent, 0.0)).xyz);
    out.uv              = in.uv;
    return out;
}

// ---------------------------------------------------------------------------
// PBR helper functions
// ---------------------------------------------------------------------------

constant float PI = 3.14159265359;

// GGX/Trowbridge-Reitz normal distribution function.
static float distribution_ggx(float3 N, float3 H, float roughness) {
    float a      = roughness * roughness;
    float a2     = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom  = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry function (single direction).
static float geometry_schlick_ggx(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith's method combining two Schlick-GGX terms.
static float geometry_smith(float3 N, float3 V, float3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

// Schlick approximation for Fresnel reflectance.
static float3 fresnel_schlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Schlick Fresnel with roughness correction for IBL.
static float3 fresnel_schlick_roughness(float cos_theta, float3 f0, float roughness) {
    return f0 + (max(float3(1.0 - roughness), f0) - f0) *
           pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ---------------------------------------------------------------------------
// L2 Spherical Harmonics evaluation (9 coefficients)
// ---------------------------------------------------------------------------

static float3 evaluate_sh(float3 normal, const constant float4 sh[9]) {
    // L0
    float3 result = sh[0].xyz * 0.282095;
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
    return max(result, float3(0.0));
}

// ---------------------------------------------------------------------------
// Cascaded shadow mapping with PCSS
// ---------------------------------------------------------------------------

constant float SHADOW_MAP_SIZE = 2048.0;
constant int   CASCADE_COUNT   = 3;
constant float CASCADE_TRANSITION_WIDTH = 0.1;

constant float2 POISSON_16[16] = {
    float2(-0.94201624, -0.39906216), float2( 0.94558609, -0.76890725),
    float2(-0.09418410, -0.92938870), float2( 0.34495938,  0.29387760),
    float2(-0.91588581,  0.45771432), float2(-0.81544232, -0.87912464),
    float2(-0.38277543,  0.27676845), float2( 0.97484398,  0.75648379),
    float2( 0.44323325, -0.97511554), float2( 0.53742981, -0.47373420),
    float2(-0.26496911, -0.41893023), float2( 0.79197514,  0.19090188),
    float2(-0.24188840,  0.99706507), float2(-0.81409955,  0.91437590),
    float2( 0.19984126,  0.78641367), float2( 0.14383161, -0.14100790),
};

// Interleaved gradient noise for per-pixel Poisson disk rotation.
static float interleaved_gradient_noise(float2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, float2(0.06711056, 0.00583715))));
}

// Rotate a 2D sample by the given angle (radians).
static float2 rotate_poisson(float2 sample, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float2(sample.x * c - sample.y * s, sample.x * s + sample.y * c);
}

// Stage 1 of PCSS: search for occluders near the receiver.
// Returns float2(average_blocker_depth, blocker_count).
static float2 find_blocker(
    float2 shadow_uv, float receiver_depth, float search_radius,
    int cascade_index, float rotation,
    depth2d_array<float> shadow_map, sampler shadow_depth_sampler
) {
    float blocker_sum   = 0.0;
    float blocker_count = 0.0;
    for (int i = 0; i < 16; i++) {
        float2 offset    = rotate_poisson(POISSON_16[i], rotation) * search_radius;
        float2 sample_uv = shadow_uv + offset;
        float shadow_depth = shadow_map.sample(shadow_depth_sampler, sample_uv,
                                               cascade_index, level(0));
        if (shadow_depth < receiver_depth) {
            blocker_sum   += shadow_depth;
            blocker_count += 1.0;
        }
    }
    return float2(blocker_sum / max(blocker_count, 1.0), blocker_count);
}

// Full PCSS shadow: blocker search -> penumbra estimation -> filtered PCF.
static float pcss_shadow(
    float2 shadow_uv, float receiver_depth, int cascade_index,
    float2 pixel_pos, float light_size,
    depth2d_array<float> shadow_map, sampler shadow_sampler,
    sampler shadow_depth_sampler
) {
    float rotation     = interleaved_gradient_noise(pixel_pos) * 6.28318;
    float search_radius = light_size / SHADOW_MAP_SIZE;

    // Stage 1: blocker search.
    float2 blocker = find_blocker(shadow_uv, receiver_depth, search_radius,
                                  cascade_index, rotation,
                                  shadow_map, shadow_depth_sampler);
    if (blocker.y < 1.0) {
        return 1.0;  // No blockers — fully lit.
    }

    // Stage 2: penumbra estimation.
    float penumbra       = (receiver_depth - blocker.x) / blocker.x * light_size;
    float filter_radius  = penumbra / SHADOW_MAP_SIZE;
    float clamped_radius = clamp(filter_radius, 1.0 / SHADOW_MAP_SIZE, search_radius * 2.0);

    // Stage 3: filtered PCF with Poisson disk.
    float shadow = 0.0;
    for (int i = 0; i < 16; i++) {
        float2 offset = rotate_poisson(POISSON_16[i], rotation) * clamped_radius;
        shadow += shadow_map.sample_compare(shadow_sampler,
                                            shadow_uv + offset,
                                            cascade_index,
                                            receiver_depth);
    }
    return shadow / 16.0;
}

// Compute a normal-offset bias position for shadow sampling.
static float3 shadow_bias_position(
    float3 world_pos, float3 world_normal, float3 light_dir,
    int cascade_idx, constant SceneUniforms& scene
) {
    float normal_bias_scale = scene.shadow_params.y;
    float slope_bias_scale  = scene.shadow_params.z;
    float cascade_bias_scale = scene.shadow_params.w;

    float NdotL       = max(dot(world_normal, light_dir), 0.001);
    float slope_factor = sqrt(1.0 - NdotL * NdotL) / NdotL;
    float base_bias    = normal_bias_scale + slope_bias_scale * slope_factor;
    float effective_bias = base_bias * (1.0 + float(cascade_idx) * cascade_bias_scale);

    return world_pos + world_normal * effective_bias;
}

// Project a world-space position into a cascade's shadow UV and depth.
static float3 cascade_project(float3 world_pos, int cascade_idx,
                               constant SceneUniforms& scene) {
    float4x4 vp  = scene.cascade_vps[cascade_idx];
    float4   clip = vp * float4(world_pos, 1.0);
    float3   ndc  = clip.xyz / clip.w;
    float2   uv   = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    return float3(uv.x, uv.y, ndc.z);
}

static float shadow_factor_biased(
    float3 world_pos, float3 world_normal, float2 pixel_pos,
    constant SceneUniforms& scene,
    depth2d_array<float> shadow_map, sampler shadow_sampler,
    sampler shadow_depth_sampler
) {
    float3 light_dir = normalize(-scene.light_direction.xyz);
    float  light_size = scene.shadow_params.x;

    // Select the tightest cascade that contains this fragment.
    int cascade_idx = 0;
    for (int ci = 0; ci < CASCADE_COUNT; ci++) {
        float3 biased = shadow_bias_position(world_pos, world_normal, light_dir, ci, scene);
        float3 proj   = cascade_project(biased, ci, scene);
        if (proj.x >= 0.0 && proj.x <= 1.0 && proj.y >= 0.0 && proj.y <= 1.0) {
            cascade_idx = ci;
            break;
        }
    }

    // Compute primary cascade shadow.
    float3 biased_primary = shadow_bias_position(world_pos, world_normal, light_dir,
                                                  cascade_idx, scene);
    float3 proj_primary   = cascade_project(biased_primary, cascade_idx, scene);
    float2 uv_primary     = proj_primary.xy;

    if (uv_primary.x < 0.0 || uv_primary.x > 1.0 ||
        uv_primary.y < 0.0 || uv_primary.y > 1.0) {
        return 1.0;
    }

    float shadow_primary = pcss_shadow(uv_primary, proj_primary.z, cascade_idx,
                                        pixel_pos, light_size,
                                        shadow_map, shadow_sampler,
                                        shadow_depth_sampler);

    // Cascade blending: smooth transition near the boundary.
    float edge_dist = min(
        min(uv_primary.x, 1.0 - uv_primary.x),
        min(uv_primary.y, 1.0 - uv_primary.y)
    );

    int next_idx = cascade_idx + 1;
    if (edge_dist < CASCADE_TRANSITION_WIDTH && next_idx < CASCADE_COUNT) {
        float3 biased_next = shadow_bias_position(world_pos, world_normal, light_dir,
                                                   next_idx, scene);
        float3 proj_next   = cascade_project(biased_next, next_idx, scene);
        float2 uv_next     = proj_next.xy;

        if (uv_next.x >= 0.0 && uv_next.x <= 1.0 &&
            uv_next.y >= 0.0 && uv_next.y <= 1.0) {
            float shadow_next = pcss_shadow(uv_next, proj_next.z, next_idx,
                                             pixel_pos, light_size,
                                             shadow_map, shadow_sampler,
                                             shadow_depth_sampler);
            float blend = smoothstep(0.0, CASCADE_TRANSITION_WIDTH, edge_dist);
            return mix(shadow_next, shadow_primary, blend);
        }
    }

    return shadow_primary;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

fragment float4 fragment_main(
    VertexOutput                in                       [[stage_in]],
    constant SceneUniforms&     scene                    [[buffer(0)]],
    constant MaterialUniforms&  material                 [[buffer(1)]],
    depth2d_array<float>        shadow_map               [[texture(0)]],
    sampler                     shadow_sampler            [[sampler(0)]],
    sampler                     shadow_depth_sampler      [[sampler(1)]],
    texturecube<float>          ibl_irradiance_map       [[texture(1)]],
    texturecube<float>          ibl_specular_map         [[texture(2)]],
    texture2d<float>            ibl_brdf_lut             [[texture(3)]],
    sampler                     ibl_sampler              [[sampler(2)]],
    texture2d<float>            albedo_tex               [[texture(4)]],
    sampler                     albedo_sampler           [[sampler(3)]],
    texture2d<float>            normal_tex               [[texture(5)]],
    texture2d<float>            metallic_roughness_tex   [[texture(6)]],
    texture2d<float>            ao_tex                   [[texture(7)]],
    texture2d<float>            emissive_tex             [[texture(8)]]
) {
    // --- Albedo & alpha ---
    float4 tex_color = albedo_tex.sample(albedo_sampler, in.uv);
    float3 albedo    = material.albedo.rgb * tex_color.rgb;
    float  alpha     = material.albedo.a * tex_color.a;

    // Alpha cutoff (mask mode).
    if (material.alpha_mode > 0.5 && material.alpha_mode < 1.5) {
        if (alpha < material.alpha_cutoff) {
            discard_fragment();
        }
    }

    // --- Metallic / roughness ---
    float metallic  = material.metallic;
    float roughness = material.roughness;
    if (material.has_metallic_roughness_tex > 0.5) {
        float4 mr_sample = metallic_roughness_tex.sample(albedo_sampler, in.uv);
        roughness = material.roughness * mr_sample.g;
        metallic  = material.metallic  * mr_sample.b;
    }
    roughness = max(roughness, 0.04f);

    // --- Normal mapping ---
    float3 N;
    if (material.has_normal_map > 0.5) {
        float3 sampled       = normal_tex.sample(albedo_sampler, in.uv).rgb;
        float3 tangent_normal = sampled * 2.0 - 1.0;
        float3 T     = normalize(in.world_tangent);
        float3 N_vert = normalize(in.world_normal);
        float3 B     = cross(N_vert, T);
        N = normalize(T * tangent_normal.x + B * tangent_normal.y + N_vert * tangent_normal.z);
    } else {
        N = normalize(in.world_normal);
    }

    float3 V  = normalize(scene.camera_pos.xyz - in.world_pos);
    float3 F0 = mix(float3(0.04), albedo, metallic);

    // --- Directional light (Cook-Torrance) ---
    float3 L = normalize(-scene.light_direction.xyz);
    float3 H = normalize(V + L);
    float  NdotL          = max(dot(N, L), 0.0);
    float  light_intensity = scene.light_color.w;
    float3 radiance       = scene.light_color.rgb * light_intensity;

    float  D        = distribution_ggx(N, H, roughness);
    float  G        = geometry_smith(N, V, L, roughness);
    float3 F        = fresnel_schlick(max(dot(H, V), 0.0), F0);
    float3 numerator   = D * G * F;
    float  denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    float3 specular    = numerator / denominator;

    float3 kS = F;
    float3 kD = (float3(1.0) - kS) * (1.0 - metallic);
    float  shadow = shadow_factor_biased(in.world_pos, N, in.clip_position.xy,
                                          scene, shadow_map, shadow_sampler,
                                          shadow_depth_sampler);
    float3 Lo = (kD * albedo / PI + specular) * radiance * NdotL * shadow;

    // --- Point lights ---
    int n_point = int(scene.num_point_lights.x);
    for (int pi = 0; pi < n_point; pi++) {
        float3 pl_pos       = scene.point_lights[pi].position.xyz;
        float  pl_range     = scene.point_lights[pi].position.w;
        float3 pl_color     = scene.point_lights[pi].color.rgb;
        float  pl_intensity = scene.point_lights[pi].color.a;

        float3 pl_dir  = pl_pos - in.world_pos;
        float  pl_dist = length(pl_dir);
        if (pl_dist > pl_range) { continue; }

        float3 pl_L         = pl_dir / pl_dist;
        float  pl_NdotL     = max(dot(N, pl_L), 0.0);
        float  pl_attenuation = 1.0 / (pl_dist * pl_dist + 0.01);
        float  pl_falloff   = saturate(1.0 - pl_dist / pl_range);
        float3 pl_radiance  = pl_color * pl_intensity * pl_attenuation * pl_falloff;
        Lo += (kD * albedo / PI) * pl_radiance * pl_NdotL;
    }

    // --- Spot lights ---
    int n_spot = int(scene.num_spot_lights.x);
    for (int si = 0; si < n_spot; si++) {
        float3 sl_pos       = scene.spot_lights[si].position.xyz;
        float  sl_range     = scene.spot_lights[si].position.w;
        float3 sl_dir_norm  = normalize(scene.spot_lights[si].direction.xyz);
        float3 sl_color     = scene.spot_lights[si].color.rgb;
        float  sl_intensity = scene.spot_lights[si].color.a;
        float  sl_inner_cos = scene.spot_lights[si].cone.x;
        float  sl_outer_cos = scene.spot_lights[si].cone.y;

        float3 sl_to_frag = in.world_pos - sl_pos;
        float  sl_dist    = length(sl_to_frag);
        if (sl_dist > sl_range) { continue; }

        float3 sl_L         = -normalize(sl_to_frag);
        float  sl_NdotL     = max(dot(N, sl_L), 0.0);
        float  sl_cos_theta = dot(normalize(sl_to_frag), sl_dir_norm);
        float  sl_cone_atten = saturate((sl_cos_theta - sl_outer_cos) /
                                         (sl_inner_cos - sl_outer_cos));
        float  sl_dist_atten = 1.0 / (sl_dist * sl_dist + 0.01);
        float  sl_falloff    = saturate(1.0 - sl_dist / sl_range);
        float3 sl_radiance   = sl_color * sl_intensity * sl_dist_atten *
                                sl_falloff * sl_cone_atten;
        Lo += (kD * albedo / PI) * sl_radiance * sl_NdotL;
    }

    // --- Ambient (SH probes or flat) ---
    float3 ambient;
    if (scene.probe_enabled.x > 0.5) {
        ambient = evaluate_sh(N, scene.probe_sh) * albedo;
    } else {
        float ambient_intensity = scene.ambient_color.w;
        ambient = scene.ambient_color.rgb * ambient_intensity * albedo;
    }
    if (material.has_ao_tex > 0.5) {
        float ao = ao_tex.sample(albedo_sampler, in.uv).r;
        ambient *= ao;
    }
    float3 color = ambient + Lo;

    // --- IBL (Image-Based Lighting) ---
    float n_dot_v = max(dot(N, V), 0.0);
    if (scene.ibl_params.x > 0.5) {
        float ibl_intensity = scene.ibl_params.y;

        float3 F_ibl  = fresnel_schlick_roughness(n_dot_v, F0, roughness);
        float3 kS_ibl = F_ibl;
        float3 kD_ibl = (1.0 - kS_ibl) * (1.0 - metallic);

        // Diffuse IBL: sample irradiance cubemap with the surface normal.
        float3 irradiance  = ibl_irradiance_map.sample(ibl_sampler, N).rgb;
        float3 diffuse_ibl = irradiance * albedo * kD_ibl;

        // Specular IBL: sample pre-filtered environment map at the reflection direction.
        float3 R       = reflect(-V, N);
        float  max_mip = 4.0;
        float3 specular_color = ibl_specular_map.sample(ibl_sampler, R,
                                                         level(roughness * max_mip)).rgb;

        // BRDF LUT: lookup Fresnel scale and bias.
        float2 brdf = ibl_brdf_lut.sample(ibl_sampler, float2(n_dot_v, roughness)).rg;
        float3 specular_ibl = specular_color * (F_ibl * brdf.x + brdf.y);

        color += (diffuse_ibl + specular_ibl) * ibl_intensity;
    }

    // --- Emissive ---
    float3 emissive_color = material.emissive;
    if (material.has_emissive_tex > 0.5) {
        float3 emissive_sample = emissive_tex.sample(albedo_sampler, in.uv).rgb;
        emissive_color *= emissive_sample;
    }
    color += emissive_color;

    // --- Output ---
    float out_alpha = 1.0;
    if (material.alpha_mode > 1.5) {
        out_alpha = alpha;
    }
    return float4(color, out_alpha);
}
