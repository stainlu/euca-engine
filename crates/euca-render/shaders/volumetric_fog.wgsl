// Volumetric fog / god-ray compute shader.
//
// For each screen pixel we ray-march from the camera through an exponential
// height-fog volume, accumulating extinction and in-scattering from a single
// directional light. The result is written to an Rgba16Float storage texture
// (rgb = fog color, a = transmittance inverse i.e. opacity).
//
// Bindings (group 0):
//   @binding(0) fog_params  -- uniform FogParams
//   @binding(1) output_tex  -- storage texture (write), rgba16float

struct FogParams {
    // Camera
    camera_pos:         vec4<f32>,  // xyz = eye position
    inv_vp:             mat4x4<f32>,
    // Light
    light_direction:    vec4<f32>,  // xyz = normalised direction TO light
    light_color:        vec4<f32>,  // xyz = light color, w unused
    // Fog properties
    fog_color:          vec4<f32>,  // xyz = tint, w = light_contribution (god ray strength)
    fog_params:         vec4<f32>,  // x = density, y = scattering, z = absorption, w = height_falloff
    fog_params2:        vec4<f32>,  // x = max_distance, y = screen_width, z = screen_height, w unused
}

@group(0) @binding(0) var<uniform> params: FogParams;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;

// Henyey-Greenstein phase function.
// g = 0 -> isotropic, g > 0 -> forward scattering.
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(denom, 1.5));
}

// Reconstruct a world-space ray direction from pixel coordinates.
fn pixel_to_world_dir(pixel: vec2<f32>) -> vec3<f32> {
    let screen = vec2<f32>(params.fog_params2.y, params.fog_params2.z);
    let ndc = vec2<f32>(
        (pixel.x / screen.x) * 2.0 - 1.0,
         1.0 - (pixel.y / screen.y) * 2.0,
    );
    let clip = vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
    let world_h = params.inv_vp * clip;
    return normalize(world_h.xyz / world_h.w - params.camera_pos.xyz);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let screen_w = u32(params.fog_params2.y);
    let screen_h = u32(params.fog_params2.z);
    if gid.x >= screen_w || gid.y >= screen_h {
        return;
    }

    let pixel = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let ray_dir = pixel_to_world_dir(pixel);
    let ray_origin = params.camera_pos.xyz;

    let density      = params.fog_params.x;
    let scattering    = params.fog_params.y;
    let absorption    = params.fog_params.z;
    let height_falloff = params.fog_params.w;
    let max_distance  = params.fog_params2.x;
    let light_contribution = params.fog_color.w;

    let extinction = scattering + absorption;
    let light_dir = normalize(params.light_direction.xyz); // direction toward light

    // Phase function: cos(angle between ray and light direction).
    let cos_theta = dot(ray_dir, light_dir);
    let phase = henyey_greenstein(cos_theta, 0.7); // forward-scattering bias for god rays

    // Ray-march parameters.
    let num_steps = 64u;
    let step_size = max_distance / f32(num_steps);

    var accumulated_color = vec3<f32>(0.0);
    var transmittance = 1.0;

    for (var i = 0u; i < num_steps; i = i + 1u) {
        let t = (f32(i) + 0.5) * step_size;
        let sample_pos = ray_origin + ray_dir * t;

        // Exponential height-based density.
        let height = sample_pos.y;
        let local_density = density * exp(-height_falloff * max(height, 0.0));

        // Extinction at this step.
        let step_extinction = extinction * local_density * step_size;
        let step_transmittance = exp(-step_extinction);

        // In-scattering from directional light.
        let in_scatter = scattering * local_density * phase * light_contribution;
        let light_color = params.light_color.xyz;

        // Energy-conserving integration (Beer-Lambert).
        let scatter_color = params.fog_color.xyz * light_color * in_scatter;

        // Integrate: color += transmittance * scatter * step_size
        accumulated_color += transmittance * scatter_color * step_size;
        transmittance *= step_transmittance;

        // Early exit when fog is nearly opaque.
        if transmittance < 0.01 {
            break;
        }
    }

    let opacity = 1.0 - transmittance;
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(accumulated_color, opacity));
}
