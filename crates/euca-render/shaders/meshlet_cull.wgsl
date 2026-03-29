// Meshlet GPU culling compute shader.
//
// For each meshlet, this shader:
//   1. Transforms the meshlet AABB to world space.
//   2. Tests the AABB against 6 frustum planes (frustum culling).
//   3. Tests the normal cone against the view direction (backface culling).
//   4. Optionally tests the projected AABB against the hierarchical Z-buffer
//      (occlusion culling).
//   5. Atomically appends visible meshlets to the output indirect draw buffer.
//
// Normal cone backface culling is the key Nanite-style optimisation: if the
// camera lies entirely behind the bounding cone of all triangle normals in a
// meshlet, the entire cluster is back-facing and can be discarded.
//
// HZB occlusion culling is gated by `uniforms.hzb_enabled` because the HZB
// is unavailable on the first frame (no previous depth buffer yet).
//
// Bindings (group 0):
//   @binding(0) meshlets            -- storage (read)  array<GpuMeshlet>
//   @binding(1) instance_transforms -- storage (read)  array<InstanceTransform>
//   @binding(2) uniforms            -- uniform         CullUniforms
//   @binding(3) indirect_args       -- storage (rw)    array<DrawIndexedIndirect>
//   @binding(4) draw_count          -- storage (rw)    atomic<u32>
//   @binding(5) hzb_texture         -- texture_2d<f32> hierarchical Z-buffer
//   @binding(6) hzb_sampler         -- sampler         for hzb_texture

// -- Per-meshlet cluster descriptor, uploaded once per mesh --
struct GpuMeshlet {
    vertex_offset:    u32,
    vertex_count:     u32,
    triangle_offset:  u32,
    triangle_count:   u32,
    aabb_center:       vec4<f32>, // xyz = center, w = unused
    aabb_half_extents: vec4<f32>, // xyz = half-extents, w = unused
    cone_axis_cutoff:  vec4<f32>, // xyz = cone axis (unit), w = cos(cone half-angle)
}

// -- Per-instance model transform --
struct InstanceTransform {
    model_col0: vec4<f32>,
    model_col1: vec4<f32>,
    model_col2: vec4<f32>,
    model_col3: vec4<f32>,
}

// -- Uniform parameters for the cull pass --
struct CullUniforms {
    frustum_planes:  array<vec4<f32>, 6>,
    camera_position: vec4<f32>,
    view_proj:       mat4x4<f32>,
    hzb_size:        vec2<f32>, // HZB texture dimensions (width, height)
    meshlet_count:   u32,
    hzb_enabled:     u32,       // 0 = skip HZB, 1 = use HZB
}

// -- Output: maps directly to wgpu DrawIndexedIndirect --
struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

@group(0) @binding(0) var<storage, read>       meshlets:            array<GpuMeshlet>;
@group(0) @binding(1) var<storage, read>       instance_transforms: array<InstanceTransform>;
@group(0) @binding(2) var<uniform>             uniforms:            CullUniforms;
@group(0) @binding(3) var<storage, read_write> indirect_args:       array<DrawIndexedIndirect>;
@group(0) @binding(4) var<storage, read_write> draw_count:          atomic<u32>;
@group(0) @binding(5) var hzb_texture: texture_2d<f32>;
@group(0) @binding(6) var hzb_sampler: sampler;

// ── Helper functions ───────────────────────────────────────────────────

/// Test AABB against six frustum planes.
/// Returns true when the AABB is at least partially inside the frustum.
fn frustum_test(center: vec3<f32>, half_ext: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = uniforms.frustum_planes[i];
        let n = plane.xyz;
        let d = plane.w;
        // Project half-extents onto plane normal for the effective radius.
        let r = dot(half_ext, abs(n));
        let dist = dot(n, center) + d;
        if dist < -r {
            return false;
        }
    }
    return true;
}

/// Normal cone backface culling.
///
/// If the camera lies entirely behind the bounding cone of all triangle
/// normals in the meshlet, the cluster is fully back-facing.
///
/// Returns true when the meshlet should be CULLED (back-facing).
fn cone_cull(
    cone_axis: vec3<f32>,
    cone_cutoff: f32,
    center: vec3<f32>,
    camera_pos: vec3<f32>,
) -> bool {
    // A negative cutoff means the cone spans more than 180 degrees --
    // the meshlet cannot be reliably culled.
    if cone_cutoff < 0.0 {
        return false;
    }
    let view_dir = normalize(center - camera_pos);
    // When the view direction aligns with the cone axis more than the
    // cutoff, the camera is behind every triangle in the meshlet.
    return dot(view_dir, cone_axis) > cone_cutoff;
}

/// Transform an object-space AABB by an instance model matrix.
///
/// Produces a conservative world-space AABB by projecting the absolute
/// values of the rotation/scale columns against the half-extents.
fn transform_aabb(
    center: vec3<f32>,
    half_ext: vec3<f32>,
    transform: InstanceTransform,
) -> array<vec3<f32>, 2> {
    let model = mat4x4<f32>(
        transform.model_col0,
        transform.model_col1,
        transform.model_col2,
        transform.model_col3,
    );
    let world_center = (model * vec4<f32>(center, 1.0)).xyz;
    // Conservative world-space half-extents via absolute rotation columns.
    let abs_model = mat3x3<f32>(
        abs(model[0].xyz),
        abs(model[1].xyz),
        abs(model[2].xyz),
    );
    let world_half = abs_model * half_ext;
    return array<vec3<f32>, 2>(world_center, world_half);
}

/// Project a world-space AABB to NDC and return the bounding rect.
///
/// Returns `(ndc_min_x, ndc_min_y, ndc_max_x, ndc_max_y)` in the xy
/// components, and `ndc_min_z` (closest depth) in the result's `.x`
/// component of a second return. To keep things in one vec4 + a float,
/// we encode: `vec4(min_x, min_y, max_x, max_y)` and a separate min_z.
///
/// When any corner is behind the near plane we return a sentinel that
/// marks the AABB as conservatively visible.
struct ProjectedRect {
    rect:  vec4<f32>, // (ndc_min_x, ndc_min_y, ndc_max_x, ndc_max_y)
    min_z: f32,
    behind_camera: bool,
}

fn project_aabb(
    center: vec3<f32>,
    half_ext: vec3<f32>,
    vp: mat4x4<f32>,
) -> ProjectedRect {
    var min_xy = vec2<f32>(1.0, 1.0);
    var max_xy = vec2<f32>(-1.0, -1.0);
    var min_z = 1.0;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let corner = center + half_ext * vec3<f32>(
            select(-1.0, 1.0, (i & 1u) != 0u),
            select(-1.0, 1.0, (i & 2u) != 0u),
            select(-1.0, 1.0, (i & 4u) != 0u),
        );
        let clip = vp * vec4<f32>(corner, 1.0);
        if clip.w <= 0.0 {
            // Corner behind near plane -- conservatively visible.
            return ProjectedRect(vec4<f32>(0.0), 0.0, true);
        }
        let ndc = clip.xyz / clip.w;
        min_xy = min(min_xy, ndc.xy);
        max_xy = max(max_xy, ndc.xy);
        min_z  = min(min_z, ndc.z);
    }

    return ProjectedRect(
        vec4<f32>(min_xy.x, min_xy.y, max_xy.x, max_xy.y),
        min_z,
        false,
    );
}

/// HZB occlusion test.
///
/// Projects the AABB onto screen, picks the HZB mip level that covers
/// the projected footprint, and compares the closest AABB depth against
/// the hierarchical depth value.
///
/// Returns true when the meshlet is OCCLUDED (can be culled).
fn hzb_occluded(
    center: vec3<f32>,
    half_ext: vec3<f32>,
    vp: mat4x4<f32>,
    hzb_size: vec2<f32>,
) -> bool {
    let proj = project_aabb(center, half_ext, vp);

    // If any corner was behind the camera, conservatively keep visible.
    if proj.behind_camera {
        return false;
    }

    // Convert NDC [-1,1] to UV [0,1].
    let uv_min = proj.rect.xy * 0.5 + 0.5;
    let uv_max = proj.rect.zw * 0.5 + 0.5;

    // Screen-space extent in pixels.
    let screen_size = (uv_max - uv_min) * hzb_size;
    let max_dim = max(screen_size.x, screen_size.y);

    // Select the mip level that makes the footprint ~1 pixel.
    let mip = max(0.0, ceil(log2(max_dim)));

    // Sample HZB at the centre of the projected rect.
    let uv_center = (uv_min + uv_max) * 0.5;
    let hzb_depth = textureSampleLevel(hzb_texture, hzb_sampler, uv_center, mip).r;

    // The AABB is occluded if its closest depth is behind the HZB depth.
    return proj.min_z > hzb_depth;
}

// ── Entry point ────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.meshlet_count {
        return;
    }

    let meshlet = meshlets[idx];

    // For now, assume instance 0 (single instance per meshlet).
    // TODO: extend with instance_id per meshlet for instanced meshlet rendering.
    let transform = instance_transforms[0u];

    // Transform AABB to world space.
    let world_aabb = transform_aabb(
        meshlet.aabb_center.xyz,
        meshlet.aabb_half_extents.xyz,
        transform,
    );
    let world_center = world_aabb[0];
    let world_half   = world_aabb[1];

    // 1. Frustum culling
    if !frustum_test(world_center, world_half) {
        return;
    }

    // 2. Normal cone backface culling
    let cone_axis   = meshlet.cone_axis_cutoff.xyz;
    let cone_cutoff = meshlet.cone_axis_cutoff.w;
    if cone_cull(cone_axis, cone_cutoff, world_center, uniforms.camera_position.xyz) {
        return;
    }

    // 3. HZB occlusion culling (optional -- skipped when HZB not yet available)
    if uniforms.hzb_enabled != 0u {
        if hzb_occluded(world_center, world_half, uniforms.view_proj, uniforms.hzb_size) {
            return;
        }
    }

    // Meshlet is visible -- append to indirect draw buffer.
    let draw_idx = atomicAdd(&draw_count, 1u);
    indirect_args[draw_idx] = DrawIndexedIndirect(
        meshlet.triangle_count * 3u,  // index_count (3 indices per triangle)
        1u,                            // instance_count
        meshlet.triangle_offset,       // first_index
        i32(meshlet.vertex_offset),    // base_vertex
        0u,                            // first_instance
    );
}
