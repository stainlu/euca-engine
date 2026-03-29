// GPU terrain mesh generation compute shader.
//
// Reads a heightmap from a storage buffer and generates terrain vertex
// and index data directly on the GPU. Each thread produces one vertex
// (position + normal + tangent + UV) and the corresponding index topology.
//
// IMPORTANT: The vertex output is written as a flat array of f32 values
// rather than a struct with vec3 fields. WGSL storage buffer layout rules
// align vec3<f32> to 16 bytes, but the engine's CPU-side Vertex is tightly
// packed at 44 bytes (3+3+3+2 floats). Writing raw floats ensures exact
// byte-level compatibility.

struct TerrainParams {
    grid_cols: u32,
    grid_rows: u32,
    cell_size: f32,
    step: u32,
    origin_x: f32,
    origin_z: f32,
    heightmap_width: u32,
    heightmap_height: u32,
    height_scale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> heightmap: array<f32>;
@group(0) @binding(1) var<uniform> params: TerrainParams;
// Flat f32 array: 11 floats per vertex (pos.xyz + normal.xyz + tangent.xyz + uv.xy).
@group(0) @binding(2) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(3) var<storage, read_write> indices: array<u32>;

/// Sample heightmap with bilinear interpolation.
/// `x` and `z` are in heightmap grid coordinates (not world space).
fn sample_height(x: f32, z: f32) -> f32 {
    let max_col = f32(params.heightmap_width - 1u);
    let max_row = f32(params.heightmap_height - 1u);

    let fx = clamp(x, 0.0, max_col);
    let fz = clamp(z, 0.0, max_row);

    let ix = u32(floor(min(fx, max_col - 1.0)));
    let iz = u32(floor(min(fz, max_row - 1.0)));
    let dx = fx - f32(ix);
    let dz = fz - f32(iz);

    let w = params.heightmap_width;
    let h00 = heightmap[iz * w + ix];
    let h10 = heightmap[iz * w + min(ix + 1u, w - 1u)];
    let h01 = heightmap[min(iz + 1u, params.heightmap_height - 1u) * w + ix];
    let h11 = heightmap[min(iz + 1u, params.heightmap_height - 1u) * w + min(ix + 1u, w - 1u)];

    let h0 = mix(h00, h10, dx);
    let h1 = mix(h01, h11, dx);
    return mix(h0, h1, dz) * params.height_scale;
}

/// Compute surface normal via central differences on the heightmap.
fn compute_normal(x: f32, z: f32) -> vec3<f32> {
    let eps = 1.0;
    let hL = sample_height(x - eps, z);
    let hR = sample_height(x + eps, z);
    let hD = sample_height(x, z - eps);
    let hU = sample_height(x, z + eps);
    return normalize(vec3<f32>(hL - hR, 2.0 * eps, hD - hU));
}

/// Generate one vertex per thread: position, normal, tangent, UV.
@compute @workgroup_size(64)
fn generate_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.grid_cols * params.grid_rows;
    if idx >= total {
        return;
    }

    let col = idx % params.grid_cols;
    let row = idx / params.grid_cols;

    // Heightmap grid coordinates (accounting for LOD step).
    let hm_x = f32(col * params.step);
    let hm_z = f32(row * params.step);

    // World-space position.
    let world_x = params.origin_x + f32(col) * params.cell_size;
    let world_z = params.origin_z + f32(row) * params.cell_size;
    let height = sample_height(hm_x, hm_z);

    let normal = compute_normal(hm_x, hm_z);

    // Tangent along the X axis (terrain-space).
    let tangent = vec3<f32>(1.0, 0.0, 0.0);

    // UV: normalised [0, 1] across the grid.
    let u = f32(col) / f32(max(params.grid_cols - 1u, 1u));
    let v = f32(row) / f32(max(params.grid_rows - 1u, 1u));

    // Write 11 f32 values per vertex (44 bytes, matching CPU Vertex layout).
    let base = idx * 11u;
    // position
    vertices[base + 0u] = world_x;
    vertices[base + 1u] = height;
    vertices[base + 2u] = world_z;
    // normal
    vertices[base + 3u] = normal.x;
    vertices[base + 4u] = normal.y;
    vertices[base + 5u] = normal.z;
    // tangent
    vertices[base + 6u] = tangent.x;
    vertices[base + 7u] = tangent.y;
    vertices[base + 8u] = tangent.z;
    // uv
    vertices[base + 9u] = u;
    vertices[base + 10u] = v;
}

/// Generate index topology: two triangles per quad, six indices per quad.
@compute @workgroup_size(64)
fn generate_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let quad_cols = params.grid_cols - 1u;
    let quad_rows = params.grid_rows - 1u;
    let total_quads = quad_cols * quad_rows;
    if idx >= total_quads {
        return;
    }

    let col = idx % quad_cols;
    let row = idx / quad_cols;

    let tl = row * params.grid_cols + col;
    let tr = tl + 1u;
    let bl = tl + params.grid_cols;
    let br = bl + 1u;

    let base = idx * 6u;
    // First triangle (top-left, bottom-left, top-right) — matches CPU winding.
    indices[base + 0u] = tl;
    indices[base + 1u] = bl;
    indices[base + 2u] = tr;
    // Second triangle (top-right, bottom-left, bottom-right).
    indices[base + 3u] = tr;
    indices[base + 4u] = bl;
    indices[base + 5u] = br;
}
