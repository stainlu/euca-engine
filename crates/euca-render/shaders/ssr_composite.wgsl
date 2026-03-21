// SSR Composite — blends screen-space reflection output over scene color.
// Uses alpha from the SSR texture to control blend strength.

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var ssr_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_tex, tex_sampler, in.uv);
    let reflection = textureSample(ssr_tex, tex_sampler, in.uv);

    // Blend: scene + reflection * alpha
    let result = scene.rgb + reflection.rgb * reflection.a;
    return vec4<f32>(result, scene.a);
}
