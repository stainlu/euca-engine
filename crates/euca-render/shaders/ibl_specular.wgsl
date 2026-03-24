// Specular pre-filter compute shader.
//
// Pre-filters an environment cubemap for split-sum specular IBL. Each mip
// level corresponds to a roughness value (roughness = mip / (mip_count - 1)).
// Higher mip levels are blurrier, representing rougher surfaces.
//
// Uses GGX importance sampling to convolve the environment map with the
// GGX NDF at the target roughness.
//
// Dispatch: per face, per mip level.
//
// Bindings (group 0):
//   @binding(0) params       -- uniform { face, mip_level, mip_count, size }
//   @binding(1) env_cubemap  -- texture_cube<f32> (source environment)
//   @binding(2) env_sampler  -- sampler
//   @binding(3) output_tex   -- storage texture 2d array (write), rgba16float

struct SpecularParams {
    face: u32,
    mip_level: u32,
    mip_count: u32,
    size: u32,
}

@group(0) @binding(0) var<uniform> params: SpecularParams;
@group(0) @binding(1) var env_cubemap: texture_cube<f32>;
@group(0) @binding(2) var env_sampler: sampler;
@group(0) @binding(3) var output_tex: texture_storage_2d_array<rgba16float, write>;

const PI: f32 = 3.14159265359;
const SAMPLE_COUNT: u32 = 1024u;

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

fn importance_sample_ggx(xi: vec2<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

fn face_uv_to_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x;
    let v = uv.y;
    switch face {
        case 0u: { return normalize(vec3<f32>( 1.0,   -v,   -u)); } // +X
        case 1u: { return normalize(vec3<f32>(-1.0,   -v,    u)); } // -X
        case 2u: { return normalize(vec3<f32>(   u,  1.0,    v)); } // +Y
        case 3u: { return normalize(vec3<f32>(   u, -1.0,   -v)); } // -Y
        case 4u: { return normalize(vec3<f32>(   u,   -v,  1.0)); } // +Z
        default: { return normalize(vec3<f32>(  -u,   -v, -1.0)); } // -Z
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if gid.x >= size || gid.y >= size {
        return;
    }

    let roughness = f32(params.mip_level) / f32(params.mip_count - 1u);

    // Map texel to UV in [-1, 1].
    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(size) * 2.0 - 1.0,
        (f32(gid.y) + 0.5) / f32(size) * 2.0 - 1.0,
    );
    let n = face_uv_to_direction(params.face, uv);
    // For the pre-filter, we assume V = R = N (the common approximation).
    let v = n;

    // Build a tangent frame.
    var up_vec = vec3<f32>(0.0, 1.0, 0.0);
    if abs(n.y) > 0.999 {
        up_vec = vec3<f32>(0.0, 0.0, 1.0);
    }
    let tangent = normalize(cross(up_vec, n));
    let bitangent = cross(n, tangent);

    var prefiltered_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = importance_sample_ggx(xi, roughness);
        // Transform H from tangent space to world space.
        let h_world = h.x * tangent + h.y * bitangent + h.z * n;
        let l = normalize(2.0 * dot(v, h_world) * h_world - v);

        let n_dot_l = max(dot(n, l), 0.0);
        if n_dot_l > 0.0 {
            let sample_color = textureSampleLevel(env_cubemap, env_sampler, l, 0.0).rgb;
            prefiltered_color += sample_color * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    if total_weight > 0.0 {
        prefiltered_color = prefiltered_color / total_weight;
    }

    textureStore(output_tex, vec2<i32>(gid.xy), i32(params.face), vec4<f32>(prefiltered_color, 1.0));
}
