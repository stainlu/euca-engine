// GPU foliage instance culling compute shader.
//
// Performs per-instance frustum and distance culling on the GPU.
// Visible instances are compacted into an output model matrix buffer
// and an atomic draw count is incremented.
//
// Bindings (group 0):
//   @binding(0) instances        -- storage (read)  array<FoliageInstance>
//   @binding(1) uniforms         -- uniform         CullUniforms
//   @binding(2) visible_matrices -- storage (rw)    array<ModelMatrix>
//   @binding(3) draw_count       -- storage (rw)    atomic<u32>

struct FoliageInstance {
    // Position (xyz) + rotation angle (w)
    position_rotation: vec4<f32>,
    // Scale (xyz) + max_distance (w)
    scale_distance: vec4<f32>,
};

struct CullUniforms {
    frustum_planes: array<vec4<f32>, 6>,
    camera_position: vec4<f32>,
    instance_count: u32,
    _pad: vec3<u32>,
};

struct ModelMatrix {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
    col3: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> instances: array<FoliageInstance>;
@group(0) @binding(1) var<uniform> uniforms: CullUniforms;
@group(0) @binding(2) var<storage, read_write> visible_matrices: array<ModelMatrix>;
@group(0) @binding(3) var<storage, read_write> draw_count: atomic<u32>;

// Frustum test for a point with radius (sphere test).
fn frustum_test_sphere(center: vec3<f32>, radius: f32, planes: array<vec4<f32>, 6>) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = planes[i];
        let dist = dot(plane.xyz, center) + plane.w;
        if (dist < -radius) {
            return false;
        }
    }
    return true;
}

// Build rotation matrix from Y-axis angle.
fn rotation_y(angle: f32) -> mat3x3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat3x3<f32>(
        vec3<f32>(c, 0.0, s),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(-s, 0.0, c),
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.instance_count) {
        return;
    }

    let inst = instances[idx];
    let position = inst.position_rotation.xyz;
    let rotation = inst.position_rotation.w;
    let scale = inst.scale_distance.xyz;
    let max_distance = inst.scale_distance.w;

    // Distance culling
    let to_camera = position - uniforms.camera_position.xyz;
    let dist_sq = dot(to_camera, to_camera);
    if (dist_sq > max_distance * max_distance) {
        return;
    }

    // Frustum culling (use max scale as bounding sphere radius)
    let radius = max(max(scale.x, scale.y), scale.z);
    if (!frustum_test_sphere(position, radius, uniforms.frustum_planes)) {
        return;
    }

    // Build model matrix: T * R * S
    let rot = rotation_y(rotation);
    let scaled_rot = mat3x3<f32>(
        rot[0] * scale.x,
        rot[1] * scale.y,
        rot[2] * scale.z,
    );

    // Append to output
    let out_idx = atomicAdd(&draw_count, 1u);
    visible_matrices[out_idx] = ModelMatrix(
        vec4<f32>(scaled_rot[0], 0.0),
        vec4<f32>(scaled_rot[1], 0.0),
        vec4<f32>(scaled_rot[2], 0.0),
        vec4<f32>(position, 1.0),
    );
}
