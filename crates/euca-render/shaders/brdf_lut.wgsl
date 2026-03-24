// BRDF integration LUT compute shader.
//
// Precomputes the split-sum BRDF lookup table for GGX specular IBL.
// For each texel (NdotV, roughness), integrates the GGX BRDF split-sum
// to produce a scale factor (R) and bias factor (G) for Schlick-Fresnel.
//
// Output: Rg16Float 512x512 texture
//   R = integral of F0 * scale (Fresnel scale)
//   G = integral of bias (Fresnel bias)
//
// Bindings (group 0):
//   @binding(0) output_tex -- storage texture (write), rg32float (written as vec4, rg used)

@group(0) @binding(0) var output_tex: texture_storage_2d<rg32float, write>;

const PI: f32 = 3.14159265359;
const SAMPLE_COUNT: u32 = 1024u;
const TEX_SIZE: f32 = 512.0;

// Radical inverse (Van der Corput sequence) for Hammersley sampling.
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

// GGX importance sampling: generates a half-vector H in tangent space
// biased towards the GGX NDF for the given roughness.
fn importance_sample_ggx(xi: vec2<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// Schlick-GGX geometry function for IBL (uses k = a^2 / 2).
fn geometry_schlick_ggx_ibl(n_dot_v: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let k = a / 2.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith_ibl(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx_ibl(n_dot_v, roughness)
         * geometry_schlick_ggx_ibl(n_dot_l, roughness);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u32(TEX_SIZE) || gid.y >= u32(TEX_SIZE) {
        return;
    }

    // Map texel to (NdotV, roughness). Offset by 0.5 to sample texel centers.
    // Clamp NdotV away from 0 to avoid division by zero.
    let n_dot_v = max((f32(gid.x) + 0.5) / TEX_SIZE, 0.001);
    let roughness = max((f32(gid.y) + 0.5) / TEX_SIZE, 0.001);

    // View vector in tangent space (N = (0,0,1)).
    let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let n = vec3<f32>(0.0, 0.0, 1.0);

    var scale = 0.0;
    var bias = 0.0;

    for (var i = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = importance_sample_ggx(xi, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);

        let n_dot_l = max(l.z, 0.0);
        let n_dot_h = max(h.z, 0.0);
        let v_dot_h = max(dot(v, h), 0.0);

        if n_dot_l > 0.0 {
            let g = geometry_smith_ibl(n_dot_v, n_dot_l, roughness);
            // G_Vis = G * VdotH / (NdotH * NdotV)
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v);
            let fc = pow(1.0 - v_dot_h, 5.0);

            scale += (1.0 - fc) * g_vis;
            bias += fc * g_vis;
        }
    }

    scale = scale / f32(SAMPLE_COUNT);
    bias = bias / f32(SAMPLE_COUNT);

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(scale, bias, 0.0, 1.0));
}
