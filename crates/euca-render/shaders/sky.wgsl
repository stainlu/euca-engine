// Procedural sky shader.
// Renders an atmospheric sky dome with sun disk, glow, and horizon effects.
// Uses a fullscreen triangle driven by vertex_index.

struct SceneUniforms {
    camera_pos: vec4<f32>,
    light_direction: vec4<f32>,
    light_color: vec4<f32>,
    ambient_color: vec4<f32>,
    camera_vp: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    inv_vp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> scene: SceneUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    let x = f32(i32(id) / 2) * 4.0 - 1.0;
    let y = f32(i32(id) % 2) * 4.0 - 1.0;
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 1.0, 1.0);
    out.ndc = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct world-space ray direction from NDC.
    let clip = vec4<f32>(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let world_h = scene.inv_vp * clip;
    let world_dir = normalize(world_h.xyz / world_h.w - scene.camera_pos.xyz);

    let up = max(world_dir.y, 0.0);
    let down = max(-world_dir.y, 0.0);

    // Sky gradient colors.
    let sky_zenith = vec3<f32>(0.15, 0.3, 0.65);
    let sky_horizon = vec3<f32>(0.55, 0.7, 0.9);
    let ground_color = vec3<f32>(0.15, 0.13, 0.12);

    var color: vec3<f32>;
    if world_dir.y >= 0.0 {
        let t = pow(up, 0.5);
        color = mix(sky_horizon, sky_zenith, t);
    } else {
        let t = pow(down, 0.8);
        color = mix(sky_horizon, ground_color, t);
    }

    // Sun disk and glow.
    let sun_dir = normalize(-scene.light_direction.xyz);
    let sun_dot = max(dot(world_dir, sun_dir), 0.0);
    let sun_disk = smoothstep(0.9995, 0.9999, sun_dot);
    let sun_color = vec3<f32>(1.0, 0.95, 0.85);
    color = mix(color, sun_color * 3.0, sun_disk);

    let glow = pow(sun_dot, 64.0) * 0.6;
    color += sun_color * glow;

    // Horizon glow near the sun.
    let horizon_glow = pow(sun_dot, 8.0) * (1.0 - up) * 0.3;
    color += vec3<f32>(1.0, 0.6, 0.3) * horizon_glow;

    return vec4<f32>(color, 1.0);
}
