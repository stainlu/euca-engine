// GPU particle render shader — billboard quads facing the camera.

struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    camera_right: vec3<f32>,
    _pad0: f32,
    camera_up: vec3<f32>,
    _pad1: f32,
};

struct Particle {
    position: vec3<f32>,
    age: f32,
    velocity: vec3<f32>,
    lifetime: f32,
    size: f32,
    color: vec4<f32>,
    _pad: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> counters: array<u32>;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

// Each particle renders as a quad (2 triangles = 6 vertices).
// vertex_index 0..5 maps to the two triangles of the quad.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    var out: VertexOutput;

    if iid >= counters[0] {
        // Beyond alive count — degenerate triangle
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    let p = particles[iid];

    // Quad corners: two triangles (0,1,2) and (2,3,0)
    let corner_id = vid % 6u;
    var offset: vec2<f32>;
    var uv: vec2<f32>;
    switch corner_id {
        case 0u: { offset = vec2<f32>(-0.5, -0.5); uv = vec2<f32>(0.0, 1.0); }
        case 1u: { offset = vec2<f32>( 0.5, -0.5); uv = vec2<f32>(1.0, 1.0); }
        case 2u: { offset = vec2<f32>( 0.5,  0.5); uv = vec2<f32>(1.0, 0.0); }
        case 3u: { offset = vec2<f32>( 0.5,  0.5); uv = vec2<f32>(1.0, 0.0); }
        case 4u: { offset = vec2<f32>(-0.5,  0.5); uv = vec2<f32>(0.0, 0.0); }
        case 5u: { offset = vec2<f32>(-0.5, -0.5); uv = vec2<f32>(0.0, 1.0); }
        default: { offset = vec2<f32>(0.0); uv = vec2<f32>(0.0); }
    }

    let world_pos = p.position
        + camera.camera_right * offset.x * p.size
        + camera.camera_up * offset.y * p.size;

    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = uv;
    out.color = p.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Soft circular particle (distance from center)
    let dist = length(in.uv - vec2<f32>(0.5));
    let alpha = 1.0 - smoothstep(0.3, 0.5, dist);

    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
