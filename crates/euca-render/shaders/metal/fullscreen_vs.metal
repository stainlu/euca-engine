// Fullscreen vertex shader — Metal Shading Language.
// Generates a fullscreen triangle from vertex_id (0, 1, 2).
// Outputs UV coordinates for post-processing passes.

#include <metal_stdlib>
using namespace metal;

struct VertexOutput {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOutput vertex_main(uint id [[vertex_id]]) {
    float x = float(int(id) / 2) * 4.0 - 1.0;
    float y = float(int(id) % 2) * 4.0 - 1.0;

    VertexOutput out;
    out.position = float4(x, y, 0.0, 1.0);
    out.uv       = float2(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}
