// Procedural sky shader — Metal Shading Language.
// Renders an atmospheric sky dome with sun disk, glow, and horizon effects.
// Uses a fullscreen triangle driven by vertex_id.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Structures — only the fields needed for sky rendering.
// ---------------------------------------------------------------------------

struct SceneUniforms {
    float4   camera_pos;
    float4   light_direction;
    float4   light_color;
    float4   ambient_color;
    float4x4 camera_vp;
    float4x4 light_vp;
    float4x4 inv_vp;
};

struct VertexOutput {
    float4 position [[position]];
    float2 ndc;
};

// ---------------------------------------------------------------------------
// Vertex shader — fullscreen triangle from vertex_id.
// ---------------------------------------------------------------------------

vertex VertexOutput vertex_main(
    uint                    id    [[vertex_id]],
    constant SceneUniforms& scene [[buffer(0)]]
) {
    float x = float(int(id) / 2) * 4.0 - 1.0;
    float y = float(int(id) % 2) * 4.0 - 1.0;

    VertexOutput out;
    out.position = float4(x, y, 1.0, 1.0);
    out.ndc      = float2(x, y);
    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — procedural atmospheric sky.
// ---------------------------------------------------------------------------

fragment float4 fragment_main(
    VertexOutput            in    [[stage_in]],
    constant SceneUniforms& scene [[buffer(0)]]
) {
    // Reconstruct world-space ray direction from NDC.
    float4 clip      = float4(in.ndc.x, in.ndc.y, 1.0, 1.0);
    float4 world_h   = scene.inv_vp * clip;
    float3 world_dir = normalize(world_h.xyz / world_h.w - scene.camera_pos.xyz);

    float up   = max(world_dir.y, 0.0);
    float down = max(-world_dir.y, 0.0);

    // Sky gradient colors.
    float3 sky_zenith    = float3(0.15, 0.3, 0.65);
    float3 sky_horizon   = float3(0.55, 0.7, 0.9);
    float3 ground_color  = float3(0.15, 0.13, 0.12);

    float3 color;
    if (world_dir.y >= 0.0) {
        float t = pow(up, 0.5);
        color = mix(sky_horizon, sky_zenith, t);
    } else {
        float t = pow(down, 0.8);
        color = mix(sky_horizon, ground_color, t);
    }

    // Sun disk and glow.
    float3 sun_dir   = normalize(-scene.light_direction.xyz);
    float  sun_dot   = max(dot(world_dir, sun_dir), 0.0);
    float  sun_disk  = smoothstep(0.9995, 0.9999, sun_dot);
    float3 sun_color = float3(1.0, 0.95, 0.85);
    color = mix(color, sun_color * 3.0, sun_disk);

    float glow = pow(sun_dot, 64.0) * 0.6;
    color += sun_color * glow;

    // Horizon glow near the sun.
    float horizon_glow = pow(sun_dot, 8.0) * (1.0 - up) * 0.3;
    color += float3(1.0, 0.6, 0.3) * horizon_glow;

    return float4(color, 1.0);
}
