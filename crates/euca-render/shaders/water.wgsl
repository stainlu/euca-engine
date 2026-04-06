// Water surface shader with animated wave displacement and fresnel-based transparency.
// Uses the same bind group layout as PBR (group 0 = instance, group 1 = scene)
// but omits the material bind group — all water properties are hardcoded in the shader.

diagnostic(off, derivative_uniformity);

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

const PI: f32 = 3.14159265359;

// Water color palette
const WATER_SHALLOW: vec3<f32> = vec3(0.10, 0.35, 0.55);
const WATER_DEEP: vec3<f32> = vec3(0.04, 0.12, 0.25);
const WATER_SPECULAR_COLOR: vec3<f32> = vec3(1.0, 0.95, 0.85);

// Wave parameters: each octave is (direction_x, direction_z, frequency, amplitude)
const WAVE_OCTAVE_COUNT: i32 = 4;

// ---------------------------------------------------------------------------
// Wave functions
// ---------------------------------------------------------------------------

/// Multi-octave Gerstner-inspired sine wave displacement.
/// Returns vec3(dx, dy, dz) world-space displacement.
fn wave_displacement(world_xz: vec2<f32>, time: f32) -> vec3<f32> {
    // Four octaves with decreasing amplitude and increasing frequency.
    // Directions are hand-picked for organic-looking interference patterns.
    let dirs = array<vec2<f32>, 4>(
        normalize(vec2(1.0, 0.6)),
        normalize(vec2(-0.7, 1.0)),
        normalize(vec2(0.3, -0.8)),
        normalize(vec2(-0.5, -0.4))
    );
    let freqs = array<f32, 4>(1.2, 2.5, 4.1, 6.8);
    let amps  = array<f32, 4>(0.08, 0.04, 0.02, 0.01);
    let speeds = array<f32, 4>(1.0, 1.3, 0.9, 1.6);

    var displacement = vec3(0.0, 0.0, 0.0);
    for (var i = 0; i < WAVE_OCTAVE_COUNT; i++) {
        let phase = dot(dirs[i], world_xz) * freqs[i] + time * speeds[i];
        let s = sin(phase);
        let c = cos(phase);
        // Vertical displacement
        displacement.y += s * amps[i];
        // Horizontal displacement (Gerstner-style lateral motion)
        displacement.x += dirs[i].x * c * amps[i] * 0.3;
        displacement.z += dirs[i].y * c * amps[i] * 0.3;
    }
    return displacement;
}

/// Compute the wave-displaced normal by finite-difference sampling.
fn wave_normal(world_xz: vec2<f32>, time: f32) -> vec3<f32> {
    let eps = 0.1;
    let hc = wave_displacement(world_xz, time).y;
    let hx = wave_displacement(world_xz + vec2(eps, 0.0), time).y;
    let hz = wave_displacement(world_xz + vec2(0.0, eps), time).y;
    // Tangent vectors along X and Z, cross product gives the normal.
    let tx = vec3(eps, hx - hc, 0.0);
    let tz = vec3(0.0, hz - hc, eps);
    return normalize(cross(tz, tx));
}

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

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) iid: u32) -> VertexOutput {
    let model = instances[iid].model;
    let time = scene.elapsed_time.x;

    let world_pos_flat = (model * vec4<f32>(in.position, 1.0)).xyz;
    let disp = wave_displacement(world_pos_flat.xz, time);
    let world_pos = world_pos_flat + disp;

    var out: VertexOutput;
    out.clip_position = scene.camera_vp * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = wave_normal(world_pos_flat.xz, time);
    out.uv = in.uv;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment stage
// ---------------------------------------------------------------------------

/// Schlick Fresnel approximation for water (F0 ~0.02 for water at normal incidence).
fn fresnel_water(cos_theta: f32) -> f32 {
    let f0 = 0.02;
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let time = scene.elapsed_time.x;
    let N = normalize(in.world_normal);
    let V = normalize(scene.camera_pos.xyz - in.world_pos);
    let L = normalize(-scene.light_direction.xyz);
    let H = normalize(V + L);

    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let NdotH = max(dot(N, H), 0.0);

    // --- Fresnel ---
    let fresnel = fresnel_water(NdotV);

    // --- Water color: blend shallow/deep based on view angle ---
    // At grazing angles, see more of the surface (reflections). Looking straight
    // down, see deeper into the water.
    let depth_factor = 1.0 - NdotV;
    let water_color = mix(WATER_SHALLOW, WATER_DEEP, depth_factor * 0.6);

    // --- Animated caustic-like color variation ---
    // Subtle color shift from overlapping sine waves to suggest subsurface caustics.
    let caustic_phase = dot(in.world_pos.xz, vec2(3.7, 2.9)) + time * 0.8;
    let caustic = 0.03 * sin(caustic_phase) * sin(caustic_phase * 0.7 + 1.3);
    let base_color = water_color + vec3(caustic * 0.5, caustic, caustic * 0.3);

    // --- Diffuse lighting ---
    let light_intensity = scene.light_color.w;
    let radiance = scene.light_color.rgb * light_intensity;
    let ambient_intensity = scene.ambient_color.w;
    let ambient = scene.ambient_color.rgb * ambient_intensity;
    let diffuse = base_color * (ambient + radiance * NdotL * 0.6);

    // --- Specular highlight (Blinn-Phong for water — sharper than GGX at low roughness) ---
    let spec_power = 256.0;
    let spec = pow(NdotH, spec_power) * fresnel;
    let specular = WATER_SPECULAR_COLOR * radiance * spec;

    // --- Combine ---
    // Fresnel controls the mix: at grazing angles, more reflection (brighter);
    // at normal incidence, more transmission (see-through).
    let color = diffuse + specular;

    // Alpha: minimum ~0.4 (water is never fully invisible), rises with fresnel.
    let alpha = mix(0.4, 0.75, fresnel);

    return vec4<f32>(color, alpha);
}
