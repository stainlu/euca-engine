// Irradiance convolution compute shader.
//
// Convolves an environment cubemap with a cosine-weighted hemisphere kernel
// to produce a diffuse irradiance cubemap. Each output texel stores the
// integral of incoming radiance * cos(theta) over the hemisphere oriented
// along the texel's world-space direction.
//
// Dispatch: 6 faces, each (face_size / 8) x (face_size / 8) workgroups.
// The face index is passed via the Z component of the dispatch (or uniform).
//
// Bindings (group 0):
//   @binding(0) params       -- uniform { face: u32, size: u32 }
//   @binding(1) env_cubemap  -- texture_cube<f32> (source environment)
//   @binding(2) env_sampler  -- sampler
//   @binding(3) output_tex   -- storage texture 2d array (write), rgba16float

struct IrradianceParams {
    face: u32,
    size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: IrradianceParams;
@group(0) @binding(1) var env_cubemap: texture_cube<f32>;
@group(0) @binding(2) var env_sampler: sampler;
@group(0) @binding(3) var output_tex: texture_storage_2d_array<rgba16float, write>;

const PI: f32 = 3.14159265359;
const SAMPLE_DELTA: f32 = 0.05;

// Convert a cubemap face index + UV to a world-space direction.
fn face_uv_to_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    // uv in [-1, 1]
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

    // Map texel to UV in [-1, 1].
    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(size) * 2.0 - 1.0,
        (f32(gid.y) + 0.5) / f32(size) * 2.0 - 1.0,
    );
    let normal = face_uv_to_direction(params.face, uv);

    // Build a tangent frame from the normal.
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(normal.y) > 0.999 {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    // Integrate cosine-weighted hemisphere using uniform sampling on
    // spherical coordinates (phi, theta).
    var irradiance = vec3<f32>(0.0);
    var sample_count = 0.0;

    var phi = 0.0;
    while phi < 2.0 * PI {
        var theta = 0.0;
        while theta < 0.5 * PI {
            // Spherical to tangent-space Cartesian.
            let tangent_sample = vec3<f32>(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta),
            );
            // Transform to world space.
            let sample_dir = tangent_sample.x * tangent
                           + tangent_sample.y * bitangent
                           + tangent_sample.z * normal;

            let sample_color = textureSampleLevel(env_cubemap, env_sampler, sample_dir, 0.0).rgb;
            // cos(theta) * sin(theta) is the Jacobian for hemisphere integration.
            irradiance += sample_color * cos(theta) * sin(theta);
            sample_count += 1.0;

            theta += SAMPLE_DELTA;
        }
        phi += SAMPLE_DELTA;
    }

    irradiance = PI * irradiance / sample_count;

    textureStore(output_tex, vec2<i32>(gid.xy), i32(params.face), vec4<f32>(irradiance, 1.0));
}
