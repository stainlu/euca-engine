// Velocity buffer pass shader.
//
// Computes per-pixel screen-space motion vectors by comparing the current
// frame's clip position against the previous frame's clip position.
//
// Output: Rg16Float — (velocity_x, velocity_y) in screen UV space.
// Static objects (same model matrix both frames) naturally produce zero velocity.

struct InstanceData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

struct PrevModelData {
    prev_model: mat4x4<f32>,
}

struct VelocitySceneUniforms {
    view_projection: mat4x4<f32>,
    prev_view_projection: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(1) @binding(0) var<uniform> scene: VelocitySceneUniforms;
@group(2) @binding(0) var<storage, read> prev_models: array<PrevModelData>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) current_ndc: vec2<f32>,
    @location(1) previous_ndc: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) iid: u32) -> VertexOutput {
    let current_clip = scene.view_projection * instances[iid].model * vec4<f32>(in.position, 1.0);
    let previous_clip = scene.prev_view_projection * prev_models[iid].prev_model * vec4<f32>(in.position, 1.0);

    var out: VertexOutput;
    out.clip_position = current_clip;
    out.current_ndc = current_clip.xy / current_clip.w;
    out.previous_ndc = previous_clip.xy / previous_clip.w;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<f32> {
    // Convert NDC delta to screen UV delta: NDC is [-1,1], UV is [0,1], so scale by 0.5.
    return (in.current_ndc - in.previous_ndc) * 0.5;
}
