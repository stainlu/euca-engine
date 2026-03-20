// GPU frustum culling compute shader.
// Tests each entity's AABB against 6 frustum planes and writes a visibility bitset.
//
// Bindings (group 0):
//   @binding(0) frustum_planes -- uniform FrustumPlanes (6 x vec4<f32>)
//   @binding(1) aabbs          -- storage (read) array of Aabb structs
//   @binding(2) visibility     -- storage (read_write) array of atomic<u32> bitset
//   @binding(3) params         -- uniform CullParams { entity_count: u32 }

struct FrustumPlanes {
    planes: array<vec4<f32>, 6>,
}

struct Aabb {
    center: vec4<f32>,       // xyz = center, w unused
    half_extents: vec4<f32>, // xyz = half-extents, w unused
}

struct CullParams {
    entity_count: u32,
}

@group(0) @binding(0) var<uniform> frustum: FrustumPlanes;
@group(0) @binding(1) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(2) var<storage, read_write> visibility: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: CullParams;

fn is_visible(entity: u32) -> bool {
    let c = aabbs[entity].center.xyz;
    let h = aabbs[entity].half_extents.xyz;

    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = frustum.planes[i];
        let n = plane.xyz;
        let d = plane.w;

        // Effective radius projected onto the plane normal
        let r = dot(h, abs(n));
        // Signed distance from center to plane
        let dist = dot(n, c) + d;

        if dist < -r {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let entity = gid.x;
    if entity >= params.entity_count {
        return;
    }

    let word_index = entity / 32u;
    let bit_index = entity % 32u;

    if is_visible(entity) {
        // Atomically set the bit.
        atomicOr(&visibility[word_index], 1u << bit_index);
    }
    // Bits default to 0 (invisible) -- caller must clear the buffer before dispatch.
}
