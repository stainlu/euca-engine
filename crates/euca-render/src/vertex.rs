use bytemuck::{Pod, Zeroable};

/// Interleaved vertex layout used by the PBR forward pipeline.
///
/// Fields are tightly packed in C layout (`#[repr(C)]`) and directly
/// uploadable to the GPU via `bytemuck`. The corresponding wgpu vertex
/// buffer layout is available as [`Vertex::LAYOUT`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// Object-space position (shader location 0).
    pub position: [f32; 3],
    /// Unit-length surface normal (shader location 1).
    pub normal: [f32; 3],
    /// Tangent vector for normal mapping, aligned with the U texture axis
    /// (shader location 2).
    pub tangent: [f32; 3],
    /// Texture coordinates (shader location 3).
    pub uv: [f32; 2],
}

impl Vertex {
    /// Vertex buffer layout descriptor matching the field offsets above.
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            // position
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            // normal
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            // tangent
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 24,
                shader_location: 2,
            },
            // uv
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 36,
                shader_location: 3,
            },
        ],
    };

    /// RHI-agnostic vertex buffer layout descriptor matching the field offsets above.
    pub const RHI_LAYOUT: euca_rhi::VertexBufferLayout<'static> = euca_rhi::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: euca_rhi::VertexStepMode::Vertex,
        attributes: &[
            // position
            euca_rhi::VertexAttribute {
                format: euca_rhi::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            // normal
            euca_rhi::VertexAttribute {
                format: euca_rhi::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            // tangent
            euca_rhi::VertexAttribute {
                format: euca_rhi::VertexFormat::Float32x3,
                offset: 24,
                shader_location: 2,
            },
            // uv
            euca_rhi::VertexAttribute {
                format: euca_rhi::VertexFormat::Float32x2,
                offset: 36,
                shader_location: 3,
            },
        ],
    };
}
