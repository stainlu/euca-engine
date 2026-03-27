// Depth + Normal pre-pass shader.
//
// Renders all opaque geometry with minimal work: no lighting, no textures.
// Outputs view-space normals to a color target and depth to the depth buffer.
//
// The view-space normal is encoded as (N * 0.5 + 0.5) so each component
// maps from [-1, 1] to [0, 1], stored in an Rgba16Float render target.

struct InstanceData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    material_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct PrepassSceneUniforms {
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(1) @binding(0) var<uniform> scene: PrepassSceneUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) iid: u32) -> VertexOutput {
    let model = instances[iid].model;
    let normal_mat = instances[iid].normal_matrix;
    let world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    let world_normal = normalize((normal_mat * vec4<f32>(in.normal, 0.0)).xyz);
    let view_normal = normalize((scene.view * vec4<f32>(world_normal, 0.0)).xyz);
    var out: VertexOutput;
    out.clip_position = scene.view_projection * vec4<f32>(world_pos, 1.0);
    out.view_normal = view_normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.view_normal);
    return vec4<f32>(n * 0.5 + 0.5, 1.0);
}
