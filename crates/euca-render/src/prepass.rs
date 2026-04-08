//! Depth + normal pre-pass for screen-space effects.
//!
//! Renders all opaque geometry with a minimal shader that writes:
//! - **Depth** to a `Depth32Float` depth attachment (shared with the main pass)
//! - **View-space normals** to an `Rgba16Float` color attachment
//!
//! Screen-space effects (SSAO, SSR) can read these textures directly instead
//! of reconstructing normals from the depth buffer (lossy) or paying the
//! bandwidth cost of a full G-buffer.
//!
//! # Normal encoding
//! View-space normals are encoded as `N * 0.5 + 0.5` so each component maps
//! from `[-1, 1]` to `[0, 1]`. The fourth channel is always `1.0`.
//!
//! # Usage
//! ```ignore
//! let textures = PrepassTextures::new(&device, width, height);
//! let pipeline = PrepassPipeline::new(&device, unified_memory);
//! // ... each frame:
//! pipeline.write_scene(&device, view_projection, view);
//! pipeline.write_instances(&device, &instance_data);
//! pipeline.execute(&device, &mut encoder, &textures, |pass| {
//!     // draw opaque geometry
//! });
//! ```

use crate::buffer::{BufferKind, SmartBuffer};
use crate::vertex::Vertex;
use euca_rhi::RenderPassOps;

/// Texture format for the depth attachment.
pub const PREPASS_DEPTH_FORMAT: euca_rhi::TextureFormat = euca_rhi::TextureFormat::Depth32Float;

/// Texture format for the normal attachment (view-space XYZ + spare channel).
pub const PREPASS_NORMAL_FORMAT: euca_rhi::TextureFormat = euca_rhi::TextureFormat::Rgba16Float;

/// Initial pre-pass instance buffer capacity. Grows dynamically when exceeded.
const INITIAL_PREPASS_INSTANCE_CAPACITY: usize = 16384;

const PREPASS_SHADER: &str = include_str!("../shaders/prepass.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrepassSceneUniforms {
    /// Combined view-projection matrix (column-major).
    pub view_projection: [[f32; 4]; 4],
    /// View matrix (column-major) — used to transform normals to view space.
    pub view: [[f32; 4]; 4],
}

/// Resolution-dependent textures produced by the depth+normal pre-pass.
pub struct PrepassTextures<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pub depth_texture: D::Texture,
    pub depth_view: D::TextureView,
    pub normal_texture: D::Texture,
    pub normal_view: D::TextureView,
    pub width: u32,
    pub height: u32,
}

impl<D: euca_rhi::RenderDevice> PrepassTextures<D> {
    /// Create depth + normal textures for the given resolution.
    pub fn new(device: &D, width: u32, height: u32) -> Self {
        let w = width.max(1);
        let h = height.max(1);

        let depth_texture = device.create_texture(&euca_rhi::TextureDesc {
            label: Some("Prepass Depth"),
            size: euca_rhi::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: euca_rhi::TextureDimension::D2,
            format: PREPASS_DEPTH_FORMAT,
            usage: euca_rhi::TextureUsages::RENDER_ATTACHMENT
                | euca_rhi::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view =
            device.create_texture_view(&depth_texture, &euca_rhi::TextureViewDesc::default());

        let normal_texture = device.create_texture(&euca_rhi::TextureDesc {
            label: Some("Prepass Normal"),
            size: euca_rhi::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: euca_rhi::TextureDimension::D2,
            format: PREPASS_NORMAL_FORMAT,
            usage: euca_rhi::TextureUsages::RENDER_ATTACHMENT
                | euca_rhi::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let normal_view =
            device.create_texture_view(&normal_texture, &euca_rhi::TextureViewDesc::default());

        Self {
            depth_texture,
            depth_view,
            normal_texture,
            normal_view,
            width: w,
            height: h,
        }
    }

    /// Recreate textures at a new resolution (e.g. window resize).
    pub fn resize(&mut self, device: &D, width: u32, height: u32) {
        *self = Self::new(device, width, height);
    }
}

/// Render pipeline that writes depth + view-space normals in a single pass.
pub struct PrepassPipeline<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pipeline: D::RenderPipeline,
    instance_bgl: D::BindGroupLayout,
    scene_bgl: D::BindGroupLayout,
    instance_buffer: SmartBuffer<D>,
    instance_bind_group: D::BindGroup,
    scene_buffer: SmartBuffer<D>,
    scene_bind_group: D::BindGroup,
    /// Current capacity (in instances) of the instance buffer.
    instance_capacity: usize,
    /// Whether the GPU uses unified memory (needed for buffer re-creation).
    unified_memory: bool,
}

impl<D: euca_rhi::RenderDevice> PrepassPipeline<D> {
    /// Create the pre-pass pipeline and allocate GPU buffers.
    pub fn new(device: &D, unified_memory: bool) -> Self {
        let instance_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("Prepass Instance BGL"),
            entries: &[euca_rhi::BindGroupLayoutEntry {
                binding: 0,
                visibility: euca_rhi::ShaderStages::VERTEX,
                ty: euca_rhi::BindingType::Buffer {
                    ty: euca_rhi::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let scene_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("Prepass Scene BGL"),
            entries: &[euca_rhi::BindGroupLayoutEntry {
                binding: 0,
                visibility: euca_rhi::ShaderStages::VERTEX,
                ty: euca_rhi::BindingType::Buffer {
                    ty: euca_rhi::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(std::mem::size_of::<PrepassSceneUniforms>() as u64),
                },
                count: None,
            }],
        });

        let instance_size =
            (INITIAL_PREPASS_INSTANCE_CAPACITY * std::mem::size_of::<[[f32; 4]; 8]>()) as u64;
        let instance_buffer = SmartBuffer::new(
            device,
            instance_size,
            BufferKind::Storage,
            unified_memory,
            "Prepass Instance SSBO",
        );

        let scene_buffer = SmartBuffer::new(
            device,
            std::mem::size_of::<PrepassSceneUniforms>() as u64,
            BufferKind::Uniform,
            unified_memory,
            "Prepass Scene UBO",
        );

        let instance_bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("Prepass Instance BG"),
            layout: &instance_bgl,
            entries: &[euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                    buffer: instance_buffer.raw(),
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let scene_bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("Prepass Scene BG"),
            layout: &scene_bgl,
            entries: &[euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                    buffer: scene_buffer.raw(),
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let shader = device.create_shader(&euca_rhi::ShaderDesc {
            label: Some("Prepass Shader"),
            source: euca_rhi::ShaderSource::Wgsl(PREPASS_SHADER.into()),
        });

        let pipeline = device.create_render_pipeline(&euca_rhi::RenderPipelineDesc {
            label: Some("Prepass Pipeline"),
            layout: &[&instance_bgl, &scene_bgl],
            vertex: euca_rhi::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::RHI_LAYOUT],
            },
            fragment: Some(euca_rhi::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(euca_rhi::ColorTargetState {
                    format: PREPASS_NORMAL_FORMAT,
                    blend: Some(euca_rhi::BlendState::REPLACE),
                    write_mask: euca_rhi::ColorWrites::ALL,
                })],
            }),
            primitive: euca_rhi::PrimitiveState {
                topology: euca_rhi::PrimitiveTopology::TriangleList,
                front_face: euca_rhi::FrontFace::Ccw,
                cull_mode: Some(euca_rhi::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(euca_rhi::DepthStencilState {
                format: PREPASS_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: euca_rhi::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
        });

        Self {
            pipeline,
            instance_bgl,
            scene_bgl,
            instance_buffer,
            instance_bind_group,
            scene_buffer,
            scene_bind_group,
            instance_capacity: INITIAL_PREPASS_INSTANCE_CAPACITY,
            unified_memory,
        }
    }

    /// Upload per-frame scene matrices (view-projection and view).
    pub fn write_scene(&self, device: &D, view_projection: [[f32; 4]; 4], view: [[f32; 4]; 4]) {
        let uniforms = PrepassSceneUniforms {
            view_projection,
            view,
        };
        self.scene_buffer
            .write_bytes(device, bytemuck::bytes_of(&uniforms));
    }

    /// Grow the instance buffer if `count` exceeds capacity.
    pub fn ensure_instance_capacity(&mut self, device: &D, count: usize) {
        if count <= self.instance_capacity {
            return;
        }
        self.instance_capacity = count.next_power_of_two();
        let size = (self.instance_capacity * std::mem::size_of::<[[f32; 4]; 8]>()) as u64;
        self.instance_buffer = SmartBuffer::new(
            device,
            size,
            BufferKind::Storage,
            self.unified_memory,
            "Prepass Instance SSBO",
        );
        self.instance_bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("Prepass Instance BG"),
            layout: &self.instance_bgl,
            entries: &[euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                    buffer: self.instance_buffer.raw(),
                    offset: 0,
                    size: None,
                }),
            }],
        });
    }

    /// Upload instance data (model + normal matrices) for the current frame.
    pub fn write_instances<T: bytemuck::Pod>(&self, device: &D, data: &[T]) {
        self.instance_buffer.write(device, data);
    }

    /// Execute the depth+normal pre-pass.
    pub fn execute<'a, F>(
        &'a self,
        device: &'a D,
        encoder: &'a mut D::CommandEncoder,
        textures: &'a PrepassTextures<D>,
        draw_fn: F,
    ) where
        F: FnOnce(&mut D::RenderPass<'a>),
    {
        let mut pass = device.begin_render_pass(
            encoder,
            &euca_rhi::RenderPassDesc {
                label: Some("Depth+Normal Prepass"),
                color_attachments: &[Some(euca_rhi::RenderPassColorAttachment {
                    view: &textures.normal_view,
                    resolve_target: None,
                    ops: euca_rhi::Operations {
                        load: euca_rhi::LoadOp::Clear(euca_rhi::Color {
                            r: 0.5,
                            g: 0.5,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: euca_rhi::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(euca_rhi::RenderPassDepthStencilAttachment {
                    view: &textures.depth_view,
                    depth_ops: Some(euca_rhi::Operations {
                        load: euca_rhi::LoadOp::Clear(1.0),
                        store: euca_rhi::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
            },
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.instance_bind_group, &[]);
        pass.set_bind_group(1, &self.scene_bind_group, &[]);
        draw_fn(&mut pass);
    }

    /// Access the instance bind group layout.
    pub fn instance_bgl(&self) -> &D::BindGroupLayout {
        &self.instance_bgl
    }

    /// Access the scene bind group layout.
    pub fn scene_bgl(&self) -> &D::BindGroupLayout {
        &self.scene_bgl
    }
}

/// Encode a view-space normal into the `[0, 1]` range used by the prepass shader.
pub fn encode_view_normal(nx: f32, ny: f32, nz: f32) -> [f32; 4] {
    [nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5, 1.0]
}

/// Decode a prepass normal back from `[0, 1]` to `[-1, 1]` range.
pub fn decode_view_normal(encoded: [f32; 4]) -> [f32; 3] {
    [
        encoded[0] * 2.0 - 1.0,
        encoded[1] * 2.0 - 1.0,
        encoded[2] * 2.0 - 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepass_texture_formats() {
        assert_eq!(PREPASS_DEPTH_FORMAT, euca_rhi::TextureFormat::Depth32Float);
        assert_eq!(PREPASS_NORMAL_FORMAT, euca_rhi::TextureFormat::Rgba16Float);
        // Prepass depth format must match G-buffer depth format.
        // GBufferFormats::DEPTH is still wgpu-typed, so assert the underlying
        // format value matches instead of a direct cross-type comparison.
        assert_eq!(
            crate::deferred::GBufferFormats::DEPTH,
            euca_rhi::TextureFormat::Depth32Float,
            "G-buffer depth format must be Depth32Float (matching prepass)"
        );
    }

    #[test]
    fn prepass_scene_uniforms_gpu_aligned() {
        let size = std::mem::size_of::<PrepassSceneUniforms>();
        assert_eq!(size, 128);
        assert_eq!(
            size % 16,
            0,
            "PrepassSceneUniforms size ({size}) must be 16-byte aligned"
        );
        let u = PrepassSceneUniforms {
            view_projection: [[1.0, 0.0, 0.0, 0.0]; 4],
            view: [[0.0; 4]; 4],
        };
        assert_eq!(bytemuck::bytes_of(&u).len(), size);
    }

    #[test]
    fn normal_encoding_roundtrip() {
        let test_normals: &[[f32; 3]] = &[
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ];
        for &n in test_normals {
            let encoded = encode_view_normal(n[0], n[1], n[2]);
            for &c in &encoded[..3] {
                assert!(
                    (0.0..=1.0).contains(&c),
                    "Encoded component {c} out of [0,1] for normal {n:?}"
                );
            }
            assert!(
                (encoded[3] - 1.0).abs() < f32::EPSILON,
                "Fourth channel must be 1.0"
            );
            let decoded = decode_view_normal(encoded);
            for i in 0..3 {
                assert!(
                    (decoded[i] - n[i]).abs() < 1e-6,
                    "Roundtrip failed for component {i}: expected {}, got {}",
                    n[i],
                    decoded[i]
                );
            }
        }
    }

    #[test]
    fn normal_encoding_zero_vector() {
        let encoded = encode_view_normal(0.0, 0.0, 0.0);
        assert!((encoded[0] - 0.5).abs() < f32::EPSILON);
        assert!((encoded[1] - 0.5).abs() < f32::EPSILON);
        assert!((encoded[2] - 0.5).abs() < f32::EPSILON);
        assert!((encoded[3] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn initial_capacity_matches_deferred() {
        assert_eq!(INITIAL_PREPASS_INSTANCE_CAPACITY, 16384);
    }
}
