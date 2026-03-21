// GPU-driven culling and indirect draw argument generation.
//
// For each entity, this shader:
//   1. Tests the entity's AABB against the camera frustum (6 planes).
//   2. Computes camera distance and selects an LOD level.
//   3. Writes a DrawIndexedIndirect struct into the output buffer.
//   4. Atomically increments a draw count for visible entities.
//
// Invisible entities get zero-ed DrawIndexedIndirect arguments so they
// produce no triangles, keeping the indirect buffer at a fixed stride
// (one slot per entity).
//
// Bindings (group 0):
//   @binding(0) draw_commands  -- storage (read)  array<DrawCommandGpu>
//   @binding(1) frustum        -- uniform         FrustumData (6 planes + camera pos)
//   @binding(2) indirect_args  -- storage (rw)    array<DrawIndexedIndirect>
//   @binding(3) draw_count     -- storage (rw)    atomic<u32> visible entity count
//   @binding(4) params         -- uniform         GpuCullParams { entity_count }

// -- Input: one per entity, uploaded by the CPU each frame --
struct DrawCommandGpu {
    model_col0: vec4<f32>,
    model_col1: vec4<f32>,
    model_col2: vec4<f32>,
    model_col3: vec4<f32>,
    aabb_center:       vec4<f32>,
    aabb_half_extents: vec4<f32>,
    mesh_id:     u32,
    material_id: u32,
    index_count:    u32,
    first_index:    u32,
    vertex_offset:  i32,
    lod_count:      u32,
    lod_index_counts:      vec4<u32>,
    lod_first_indices:     vec4<u32>,
    lod_vertex_offsets:    vec4<i32>,
    lod_distance_sq:       vec4<f32>,
}

// -- Output: maps directly to wgpu DrawIndexedIndirect --
struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

struct FrustumData {
    planes: array<vec4<f32>, 6>,
    camera_position: vec4<f32>,
}

struct GpuCullParams {
    entity_count: u32,
}

@group(0) @binding(0) var<storage, read>       draw_commands: array<DrawCommandGpu>;
@group(0) @binding(1) var<uniform>             frustum:       FrustumData;
@group(0) @binding(2) var<storage, read_write> indirect_args: array<DrawIndexedIndirect>;
@group(0) @binding(3) var<storage, read_write> draw_count:    atomic<u32>;
@group(0) @binding(4) var<uniform>             params:        GpuCullParams;

fn frustum_test(center: vec3<f32>, half_ext: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = frustum.planes[i];
        let n = plane.xyz;
        let d = plane.w;
        let r = dot(half_ext, abs(n));
        let dist = dot(n, center) + d;
        if dist < -r {
            return false;
        }
    }
    return true;
}

fn select_lod(cmd: DrawCommandGpu, dist_sq: f32) -> u32 {
    let count = cmd.lod_count;
    for (var i = 0u; i < count; i = i + 1u) {
        if dist_sq <= cmd.lod_distance_sq[i] {
            return i;
        }
    }
    return count - 1u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.entity_count {
        return;
    }

    let cmd = draw_commands[idx];
    let center = cmd.aabb_center.xyz;
    let half_ext = cmd.aabb_half_extents.xyz;

    let visible = frustum_test(center, half_ext);

    if visible {
        let to_camera = frustum.camera_position.xyz - center;
        let dist_sq = dot(to_camera, to_camera);

        var index_count: u32;
        var first_index: u32;
        var base_vertex: i32;

        if cmd.lod_count > 1u {
            let lod = select_lod(cmd, dist_sq);
            index_count = cmd.lod_index_counts[lod];
            first_index = cmd.lod_first_indices[lod];
            base_vertex = cmd.lod_vertex_offsets[lod];
        } else {
            index_count = cmd.index_count;
            first_index = cmd.first_index;
            base_vertex = cmd.vertex_offset;
        }

        indirect_args[idx].index_count    = index_count;
        indirect_args[idx].instance_count = 1u;
        indirect_args[idx].first_index    = first_index;
        indirect_args[idx].base_vertex    = base_vertex;
        indirect_args[idx].first_instance = idx;

        atomicAdd(&draw_count, 1u);
    } else {
        indirect_args[idx].index_count    = 0u;
        indirect_args[idx].instance_count = 0u;
        indirect_args[idx].first_index    = 0u;
        indirect_args[idx].base_vertex    = 0i;
        indirect_args[idx].first_instance = 0u;
    }
}
