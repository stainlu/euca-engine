// Animated water surface shader.
// Multi-octave sine wave vertex displacement, fresnel transparency, specular highlights.
//
// Uses the same instance (group 0) and scene (group 1) bind groups as the PBR
// shader — no per-material group needed since water properties are baked in.

diagnostic(off, derivative_uniformity);

// ---------------------------------------------------------------------------
// Shared structures (must match pbr.wgsl layout exactly)
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
    elapsed_time: vec4<f32>,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(1) @binding(0) var<uniform> scene: SceneUniforms;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WATER_COLOR_DEEP: vec3<f32> = vec3<f32>(0.02, 0.12, 0.22);
const WATER_COLOR_SHALLOW: vec3<f32> = vec3<f32>(0.10, 0.35, 0.45);
const SPECULAR_COLOR: vec3<f32> = vec3<f32>(1.0, 0.95, 0.85);
const WAVE_HEIGHT: f32 = 0.15;
const BASE_ALPHA: f32 = 0.55;

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
    @location(2) uv: vec2<f32>,
}

// Multi-octave wave displacement.  Each octave has a different direction,
// frequency, and phase speed so the surface looks organic rather than tiled.
fn wave_height(xz: vec2<f32>, t: f32) -> f32 {
    var h: f32 = 0.0;
    // Octave 1: broad swell
    h += sin(xz.x * 1.2 + xz.y * 0.8 + t * 1.4) * 0.45;
    // Octave 2: cross chop
    h += sin(xz.x * 2.5 - xz.y * 1.7 + t * 2.1) * 0.25;
    // Octave 3: fine ripple
    h += sin(xz.x * 5.0 + xz.y * 4.3 + t * 3.6) * 0.12;
    // Octave 4: micro detail
    h += sin(xz.x * 8.7 - xz.y * 7.1 + t * 5.0) * 0.06;
    return h * WAVE_HEIGHT;
}

// Analytical normal from the partial derivatives of the wave function.
fn wave_normal(xz: vec2<f32>, t: f32) -> vec3<f32> {
    let eps = 0.05;
    let hc = wave_height(xz, t);
    let hx = wave_height(xz + vec2<f32>(eps, 0.0), t);
    let hz = wave_height(xz + vec2<f32>(0.0, eps), t);
    let dx = (hx - hc) / eps;
    let dz = (hz - hc) / eps;
    return normalize(vec3<f32>(-dx, 1.0, -dz));
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) iid: u32) -> VertexOutput {
    let model = instances[iid].model;
    let t = scene.elapsed_time.x;

    var world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;

    // Displace Y by the wave function
    world_pos.y += wave_height(world_pos.xz, t);

    var out: VertexOutput;
    out.clip_position = scene.camera_vp * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = wave_normal(world_pos.xz, t);
    out.uv = in.uv;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment stage
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let V = normalize(scene.camera_pos.xyz - in.world_pos);
    let L = normalize(-scene.light_direction.xyz);

    // Fresnel: more reflective at glancing angles
    let NdotV = max(dot(N, V), 0.0);
    let fresnel = pow(1.0 - NdotV, 4.0) * 0.8 + 0.2;

    // Blend between deep and shallow water based on view angle
    let water_color = mix(WATER_COLOR_DEEP, WATER_COLOR_SHALLOW, NdotV);

    // Ambient contribution
    let ambient = water_color * scene.ambient_color.xyz * scene.ambient_color.w;

    // Diffuse (wrap lighting for softer underwater look)
    let NdotL = max(dot(N, L), 0.0);
    let diffuse = water_color * scene.light_color.xyz * scene.light_color.w * NdotL * 0.6;

    // Specular (Blinn-Phong, tight highlight for water)
    let H = normalize(L + V);
    let NdotH = max(dot(N, H), 0.0);
    let spec = pow(NdotH, 128.0) * fresnel;
    let specular = SPECULAR_COLOR * scene.light_color.xyz * scene.light_color.w * spec;

    let color = ambient + diffuse + specular;

    // Alpha: base transparency boosted at glancing angles (fresnel)
    let alpha = mix(BASE_ALPHA, 0.85, fresnel);

    return vec4<f32>(color, alpha);
}
