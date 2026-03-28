// Shadow map depth-only pass — Metal Shading Language.
// Renders geometry from the light's perspective, outputting only depth.
// The light VP matrix is applied via the render pipeline (not in this shader).

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Structures — must match Rust #[repr(C)] and WGSL definitions.
// ---------------------------------------------------------------------------

struct InstanceData {
    float4x4 model;
    float4x4 normal_matrix;
    uint      material_id;
    uint      _pad0;
    uint      _pad1;
    uint      _pad2;
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

// ---------------------------------------------------------------------------
// Vertex shader — depth-only, no fragment shader needed.
// ---------------------------------------------------------------------------

vertex float4 vertex_main(
    VertexInput                in        [[stage_in]],
    uint                       iid       [[instance_id]],
    const device InstanceData* instances [[buffer(0)]]
) {
    return instances[iid].model * float4(in.position, 1.0);
}
