// Post-processing shader: bloom + ACES tonemapping + gamma + vignette.
// Fullscreen triangle driven by vertex_index.

diagnostic(off, derivative_uniformity);

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    let x = f32(i32(id) / 2) * 4.0 - 1.0;
    let y = f32(i32(id) % 2) * 4.0 - 1.0;
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}

// ACES filmic tonemapping curve.
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp(
        (x * (a * x + b)) / (x * (c * x + d) + e),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

// Sample surrounding texels and extract bright areas for a simple bloom effect.
fn bloom_sample(uv: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    var bloom = vec3<f32>(0.0);
    let center = textureSample(hdr_tex, hdr_sampler, uv).rgb;

    let offsets = array<vec2<f32>, 12>(
        vec2<f32>(-1.0,  0.0),
        vec2<f32>( 1.0,  0.0),
        vec2<f32>( 0.0, -1.0),
        vec2<f32>( 0.0,  1.0),
        vec2<f32>(-0.7, -0.7),
        vec2<f32>( 0.7, -0.7),
        vec2<f32>(-0.7,  0.7),
        vec2<f32>( 0.7,  0.7),
        vec2<f32>(-2.0,  0.0),
        vec2<f32>( 2.0,  0.0),
        vec2<f32>( 0.0, -2.0),
        vec2<f32>( 0.0,  2.0),
    );

    let radius = 4.0;
    for (var i = 0u; i < 12u; i++) {
        let sample_uv = uv + offsets[i] * texel * radius;
        let s = textureSample(hdr_tex, hdr_sampler, sample_uv).rgb;
        let luminance = dot(s, vec3<f32>(0.2126, 0.7152, 0.0722));
        let bright = max(luminance - 0.8, 0.0) / max(luminance, 0.001);
        bloom += s * bright;
    }

    return center + bloom * 0.08;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(hdr_tex));
    let texel = 1.0 / dims;

    let hdr = bloom_sample(in.uv, texel);
    let mapped = aces_tonemap(hdr);
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));

    // Vignette.
    let center_dist = length(in.uv - 0.5) * 1.4;
    let vignette = 1.0 - center_dist * center_dist * 0.35;
    let final_color = gamma * vignette;

    return vec4<f32>(final_color, 1.0);
}
