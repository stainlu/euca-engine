// Clustered light assignment compute shader.
//
// Divides the view frustum into a 3D grid of clusters (tiles_x * tiles_y * depth_slices).
// For each cluster, tests every active light against the cluster's view-space AABB.
// Writes per-cluster light index lists for the PBR shader to read.

const TILES_X: u32 = 16u;
const TILES_Y: u32 = 9u;
const DEPTH_SLICES: u32 = 24u;
const MAX_LIGHTS_PER_CLUSTER: u32 = 32u;
const MAX_LIGHTS: u32 = 256u;

struct ClusterConfig {
    view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    screen_size: vec2<f32>,
    near_z: f32,
    far_z: f32,
    num_lights: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct LightData {
    position_range: vec4<f32>,
    color_intensity: vec4<f32>,
    direction_type: vec4<f32>,
    cone_angles: vec4<f32>,
}

@group(0) @binding(0) var<uniform> config: ClusterConfig;
@group(0) @binding(1) var<storage, read> lights: array<LightData>;
@group(0) @binding(2) var<storage, read_write> light_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> cluster_light_counts: array<atomic<u32>>;

fn screen_to_view(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view_pos = config.inv_proj * ndc;
    return view_pos.xyz / view_pos.w;
}

fn cluster_aabb_min_max(cluster_id: vec3<u32>) -> array<vec3<f32>, 2> {
    let tile_size_x = 1.0 / f32(TILES_X);
    let tile_size_y = 1.0 / f32(TILES_Y);

    let uv_min = vec2<f32>(f32(cluster_id.x) * tile_size_x, f32(cluster_id.y) * tile_size_y);
    let uv_max = vec2<f32>(f32(cluster_id.x + 1u) * tile_size_x, f32(cluster_id.y + 1u) * tile_size_y);

    let log_ratio = log(config.far_z / config.near_z);
    let near_depth = config.near_z * exp(log_ratio * f32(cluster_id.z) / f32(DEPTH_SLICES));
    let far_depth  = config.near_z * exp(log_ratio * f32(cluster_id.z + 1u) / f32(DEPTH_SLICES));

    let depth_range = config.far_z - config.near_z;
    let near_ndc = (near_depth - config.near_z) / depth_range;
    let far_ndc  = (far_depth - config.near_z) / depth_range;

    let c00n = screen_to_view(uv_min, near_ndc);
    let c10n = screen_to_view(vec2<f32>(uv_max.x, uv_min.y), near_ndc);
    let c01n = screen_to_view(vec2<f32>(uv_min.x, uv_max.y), near_ndc);
    let c11n = screen_to_view(uv_max, near_ndc);

    let c00f = screen_to_view(uv_min, far_ndc);
    let c10f = screen_to_view(vec2<f32>(uv_max.x, uv_min.y), far_ndc);
    let c01f = screen_to_view(vec2<f32>(uv_min.x, uv_max.y), far_ndc);
    let c11f = screen_to_view(uv_max, far_ndc);

    var aabb_min = min(c00n, c10n);
    aabb_min = min(aabb_min, c01n);
    aabb_min = min(aabb_min, c11n);
    aabb_min = min(aabb_min, c00f);
    aabb_min = min(aabb_min, c10f);
    aabb_min = min(aabb_min, c01f);
    aabb_min = min(aabb_min, c11f);

    var aabb_max = max(c00n, c10n);
    aabb_max = max(aabb_max, c01n);
    aabb_max = max(aabb_max, c11n);
    aabb_max = max(aabb_max, c00f);
    aabb_max = max(aabb_max, c10f);
    aabb_max = max(aabb_max, c01f);
    aabb_max = max(aabb_max, c11f);

    return array<vec3<f32>, 2>(aabb_min, aabb_max);
}

fn sphere_aabb_intersect(center: vec3<f32>, radius: f32, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    let closest = clamp(center, aabb_min, aabb_max);
    let diff = center - closest;
    let dist_sq = dot(diff, diff);
    return dist_sq <= (radius * radius);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= TILES_X || gid.y >= TILES_Y || gid.z >= DEPTH_SLICES {
        return;
    }

    let cluster_idx = gid.x + gid.y * TILES_X + gid.z * TILES_X * TILES_Y;
    let base_offset = cluster_idx * MAX_LIGHTS_PER_CLUSTER;

    atomicStore(&cluster_light_counts[cluster_idx], 0u);

    let aabb = cluster_aabb_min_max(gid);
    let aabb_min = aabb[0];
    let aabb_max = aabb[1];

    let num_lights = min(config.num_lights, MAX_LIGHTS);

    for (var i = 0u; i < num_lights; i++) {
        let light = lights[i];
        let light_pos_world = light.position_range.xyz;
        let light_range = light.position_range.w;

        let light_pos_view = (config.view * vec4<f32>(light_pos_world, 1.0)).xyz;

        if sphere_aabb_intersect(light_pos_view, light_range, aabb_min, aabb_max) {
            let count = atomicAdd(&cluster_light_counts[cluster_idx], 1u);
            if count < MAX_LIGHTS_PER_CLUSTER {
                light_indices[base_offset + count] = i;
            }
        }
    }
}
