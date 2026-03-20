// Shadow map depth-only pass.
// Renders geometry from the light's perspective, outputting only depth.

struct InstanceData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) iid: u32) -> @builtin(position) vec4<f32> {
    return instances[iid].model * vec4<f32>(in.position, 1.0);
}
