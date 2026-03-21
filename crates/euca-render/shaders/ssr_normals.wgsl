// SSR Normals reconstruction from depth buffer.
// Reconstructs view-space normals from the depth buffer using cross-product of
// partial derivatives of the view-space position.

struct SsrNormalsUniforms {
    inv_projection: mat4x4<f32>,
    texel_size: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: SsrNormalsUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    let x = f32(i32(id) / 2) * 4.0 - 1.0;
    let y = f32(i32(id) % 2) * 4.0 - 1.0;
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}

fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view_pos = uniforms.inv_projection * ndc;
    return view_pos.xyz / view_pos.w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, tex_sampler, in.uv).r;
    if depth >= 1.0 {
        return vec4<f32>(0.5, 0.5, 1.0, 1.0); // sky: up normal
    }

    let pos_c = reconstruct_view_pos(in.uv, depth);

    // Sample neighboring depths for cross-product normal reconstruction
    let ts = uniforms.texel_size;
    let depth_r = textureSample(depth_tex, tex_sampler, in.uv + vec2<f32>(ts.x, 0.0)).r;
    let depth_u = textureSample(depth_tex, tex_sampler, in.uv + vec2<f32>(0.0, ts.y)).r;

    let pos_r = reconstruct_view_pos(in.uv + vec2<f32>(ts.x, 0.0), depth_r);
    let pos_u = reconstruct_view_pos(in.uv + vec2<f32>(0.0, ts.y), depth_u);

    let normal = normalize(cross(pos_r - pos_c, pos_u - pos_c));

    // Encode: [-1,1] -> [0,1]
    return vec4<f32>(normal * 0.5 + 0.5, 1.0);
}
