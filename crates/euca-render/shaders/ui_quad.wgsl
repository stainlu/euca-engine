// UI quad shader — draws screen-space colored rectangles.
//
// Vertices are generated from instance data (position, size, color).
// Each instance produces a quad (2 triangles, 6 vertices via index buffer).

struct UiQuadInstance {
    @location(0) pos_size: vec4<f32>,   // xy = top-left position in NDC, zw = size in NDC
    @location(1) color: vec4<f32>,      // RGBA
}

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    instance: UiQuadInstance,
) -> VsOut {
    // 6 vertices per quad: 2 triangles
    // Triangle 1: 0,1,2  Triangle 2: 2,1,3
    // 0--1     UV: (0,0) (1,0)
    // |\ |          (0,1) (1,1)
    // 2--3
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let indices = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let idx = indices[vi];
    let corner = corners[idx];

    let ndc_x = instance.pos_size.x + corner.x * instance.pos_size.z;
    let ndc_y = instance.pos_size.y + corner.y * instance.pos_size.w;

    var out: VsOut;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.color = instance.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
