//! Deferred rendering infrastructure.
//!
//! Adds G-buffer render targets and a deferred lighting pass alongside
//! the existing forward renderer. The renderer can be configured to use
//! either path via [`RenderPath`].
//!
//! # G-Buffer layout
//! - RT0: Rgba8Unorm  — Albedo RGB + Alpha
//! - RT1: Rgba16Float — Normal XYZ + Roughness
//! - RT2: Rgba8Unorm  — Metallic (R) + AO (G) + Emissive flag (B)
//! - Depth: Depth32Float (shared with forward pass)
//!
//! # Architecture
//! 1. G-buffer pass: render opaque geometry, output material properties
//! 2. Lighting pass: fullscreen quad reads G-buffer + depth, PBR lighting
//! 3. Forward pass: render transparent objects on top (always forward)

use crate::buffer::{BufferKind, SmartBuffer};
use crate::vertex::Vertex;
use euca_rhi::{RenderDevice, pass::RenderPassOps};

/// Which rendering path the engine uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderPath {
    /// Traditional forward rendering (current default).
    Forward,
    /// Deferred rendering with G-buffer.
    Deferred,
}
impl Default for RenderPath {
    fn default() -> Self {
        Self::Forward
    }
}

/// G-buffer texture format definitions.
pub struct GBufferFormats;
impl GBufferFormats {
    pub const ALBEDO: euca_rhi::TextureFormat = euca_rhi::TextureFormat::Rgba8Unorm;
    pub const NORMAL_ROUGHNESS: euca_rhi::TextureFormat = euca_rhi::TextureFormat::Rgba16Float;
    pub const MATERIAL: euca_rhi::TextureFormat = euca_rhi::TextureFormat::Rgba8Unorm;
    pub const DEPTH: euca_rhi::TextureFormat = euca_rhi::TextureFormat::Depth32Float;
    pub const ALL_COLOR: [euca_rhi::TextureFormat; 3] =
        [Self::ALBEDO, Self::NORMAL_ROUGHNESS, Self::MATERIAL];
}

/// G-buffer render targets.
///
/// Generic over [`euca_rhi::RenderDevice`] — defaults to [`euca_rhi::wgpu_backend::WgpuDevice`]
/// for backward compatibility while the renderer is being generified.
pub struct GBuffer<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pub albedo_texture: D::Texture,
    pub albedo_view: D::TextureView,
    pub normal_roughness_texture: D::Texture,
    pub normal_roughness_view: D::TextureView,
    pub material_texture: D::Texture,
    pub material_view: D::TextureView,
    pub depth_texture: D::Texture,
    pub depth_view: D::TextureView,
    pub width: u32,
    pub height: u32,
}
impl<D: RenderDevice> GBuffer<D> {
    pub fn new(device: &D, width: u32, height: u32) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let mk = |label: &str, format: euca_rhi::TextureFormat| {
            let t = device.create_texture(&euca_rhi::TextureDesc {
                label: Some(label),
                size: euca_rhi::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: euca_rhi::TextureDimension::D2,
                format,
                usage: euca_rhi::TextureUsages::RENDER_ATTACHMENT
                    | euca_rhi::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let v = device.create_texture_view(&t, &euca_rhi::TextureViewDesc::default());
            (t, v)
        };
        let (at, av) = mk("G-Buffer RT0 Albedo", GBufferFormats::ALBEDO);
        let (nt, nv) = mk(
            "G-Buffer RT1 Normal+Roughness",
            GBufferFormats::NORMAL_ROUGHNESS,
        );
        let (mt, mv) = mk("G-Buffer RT2 Material", GBufferFormats::MATERIAL);
        let (dt, dv) = mk("G-Buffer Depth", GBufferFormats::DEPTH);
        Self {
            albedo_texture: at,
            albedo_view: av,
            normal_roughness_texture: nt,
            normal_roughness_view: nv,
            material_texture: mt,
            material_view: mv,
            depth_texture: dt,
            depth_view: dv,
            width: w,
            height: h,
        }
    }
    pub fn resize(&mut self, device: &D, width: u32, height: u32) {
        *self = Self::new(device, width, height);
    }
}

const GBUFFER_SHADER: &str = include_str!("../shaders/gbuffer.wgsl");
const DEFERRED_LIGHTING_SHADER: &str = include_str!("../shaders/deferred_lighting.wgsl");
const MAX_DEFERRED_POINT_LIGHTS: usize = 128;
const MAX_DEFERRED_SPOT_LIGHTS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct GpuDeferredPointLight {
    pub position: [f32; 4],
    pub color: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct GpuDeferredSpotLight {
    pub position: [f32; 4],
    pub direction: [f32; 4],
    pub color: [f32; 4],
    pub cone: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DeferredLightingUniforms {
    pub camera_pos: [f32; 4],
    pub light_direction: [f32; 4],
    pub light_color: [f32; 4],
    pub ambient_color: [f32; 4],
    pub inv_vp: [[f32; 4]; 4],
    pub screen_size: [f32; 4],
    pub point_lights: [GpuDeferredPointLight; MAX_DEFERRED_POINT_LIGHTS],
    pub spot_lights: [GpuDeferredSpotLight; MAX_DEFERRED_SPOT_LIGHTS],
    pub num_point_lights: [f32; 4],
    pub num_spot_lights: [f32; 4],
}

/// Deferred rendering pipeline: G-buffer geometry pass + fullscreen lighting pass.
///
/// Generic over [`euca_rhi::RenderDevice`] — defaults to [`euca_rhi::wgpu_backend::WgpuDevice`]
/// for backward compatibility while the renderer is being generified.
#[allow(dead_code)]
pub struct DeferredPipeline<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pub gbuffer: GBuffer<D>,
    gbuffer_pipeline: D::RenderPipeline,
    instance_bgl: D::BindGroupLayout,
    gbuffer_scene_bgl: D::BindGroupLayout,
    material_bgl: D::BindGroupLayout,
    instance_buffer: SmartBuffer<D>,
    instance_bind_group: D::BindGroup,
    gbuffer_scene_buffer: SmartBuffer<D>,
    gbuffer_scene_bind_group: D::BindGroup,
    lighting_pipeline: D::RenderPipeline,
    lighting_bgl: D::BindGroupLayout,
    lighting_bind_group: D::BindGroup,
    lighting_buffer: SmartBuffer<D>,
    gbuffer_sampler: D::Sampler,
    material_sampler: D::Sampler,
    hdr_format: euca_rhi::TextureFormat,
    /// Current capacity (in instances) of the deferred instance buffer.
    instance_capacity: usize,
    /// Whether the GPU uses unified memory (needed for buffer re-creation).
    unified_memory: bool,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GBufferSceneUniforms {
    camera_vp: [[f32; 4]; 4],
}
/// Initial deferred instance buffer capacity. Grows dynamically when exceeded.
const INITIAL_DEFERRED_INSTANCE_CAPACITY: usize = 16384;

impl<D: RenderDevice> DeferredPipeline<D> {
    pub fn new(device: &D, width: u32, height: u32, unified_memory: bool) -> Self {
        let gbuffer = GBuffer::new(device, width, height);
        let hdr_format = euca_rhi::TextureFormat::Rgba16Float;
        let instance_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("Deferred Instance BGL"),
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
        let ibsz =
            (INITIAL_DEFERRED_INSTANCE_CAPACITY * std::mem::size_of::<[[f32; 4]; 8]>()) as u64;
        let instance_buffer = SmartBuffer::new(
            device,
            ibsz,
            BufferKind::Storage,
            unified_memory,
            "Deferred Instance SSBO",
        );
        let instance_bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("Deferred Instance BG"),
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
        let gbuffer_scene_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("GBuffer Scene BGL"),
            entries: &[euca_rhi::BindGroupLayoutEntry {
                binding: 0,
                visibility: euca_rhi::ShaderStages::VERTEX,
                ty: euca_rhi::BindingType::Buffer {
                    ty: euca_rhi::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(std::mem::size_of::<GBufferSceneUniforms>() as u64),
                },
                count: None,
            }],
        });
        let gbuffer_scene_buffer = SmartBuffer::new(
            device,
            std::mem::size_of::<GBufferSceneUniforms>() as u64,
            BufferKind::Uniform,
            unified_memory,
            "GBuffer Scene UBO",
        );
        let gbuffer_scene_bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("GBuffer Scene BG"),
            layout: &gbuffer_scene_bgl,
            entries: &[euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                    buffer: gbuffer_scene_buffer.raw(),
                    offset: 0,
                    size: None,
                }),
            }],
        });
        let te = |b: u32| euca_rhi::BindGroupLayoutEntry {
            binding: b,
            visibility: euca_rhi::ShaderStages::FRAGMENT,
            ty: euca_rhi::BindingType::Texture {
                sample_type: euca_rhi::TextureSampleType::Float { filterable: true },
                view_dimension: euca_rhi::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let material_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("Deferred Material BGL"),
            entries: &[
                euca_rhi::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Buffer {
                        ty: euca_rhi::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                te(1),
                euca_rhi::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Sampler(euca_rhi::SamplerBindingType::Filtering),
                    count: None,
                },
                te(3),
                te(4),
                te(5),
                te(6),
            ],
        });
        let material_sampler = device.create_sampler(&euca_rhi::SamplerDesc {
            label: Some("Deferred Material Sampler"),
            address_mode_u: euca_rhi::AddressMode::Repeat,
            address_mode_v: euca_rhi::AddressMode::Repeat,
            address_mode_w: euca_rhi::AddressMode::Repeat,
            mag_filter: euca_rhi::FilterMode::Linear,
            min_filter: euca_rhi::FilterMode::Linear,
            mipmap_filter: euca_rhi::FilterMode::Linear,
            ..Default::default()
        });
        let gs = device.create_shader(&euca_rhi::ShaderDesc {
            label: Some("GBuffer Shader"),
            source: euca_rhi::ShaderSource::Wgsl(GBUFFER_SHADER.into()),
        });
        let gbuffer_pipeline = device.create_render_pipeline(&euca_rhi::RenderPipelineDesc {
            label: Some("GBuffer Pipeline"),
            layout: &[&instance_bgl, &gbuffer_scene_bgl, &material_bgl],
            vertex: euca_rhi::VertexState {
                module: &gs,
                entry_point: "vs_main",
                buffers: &[Vertex::RHI_LAYOUT],
            },
            fragment: Some(euca_rhi::FragmentState {
                module: &gs,
                entry_point: "fs_main",
                targets: &[
                    Some(euca_rhi::ColorTargetState {
                        format: GBufferFormats::ALBEDO,
                        blend: Some(euca_rhi::BlendState::REPLACE),
                        write_mask: euca_rhi::ColorWrites::ALL,
                    }),
                    Some(euca_rhi::ColorTargetState {
                        format: GBufferFormats::NORMAL_ROUGHNESS,
                        blend: Some(euca_rhi::BlendState::REPLACE),
                        write_mask: euca_rhi::ColorWrites::ALL,
                    }),
                    Some(euca_rhi::ColorTargetState {
                        format: GBufferFormats::MATERIAL,
                        blend: Some(euca_rhi::BlendState::REPLACE),
                        write_mask: euca_rhi::ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: euca_rhi::PrimitiveState {
                topology: euca_rhi::PrimitiveTopology::TriangleList,
                front_face: euca_rhi::FrontFace::Ccw,
                cull_mode: Some(euca_rhi::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(euca_rhi::DepthStencilState {
                format: GBufferFormats::DEPTH,
                depth_write_enabled: true,
                depth_compare: euca_rhi::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
        });
        let gbuffer_sampler = device.create_sampler(&euca_rhi::SamplerDesc {
            label: Some("GBuffer Read Sampler"),
            mag_filter: euca_rhi::FilterMode::Nearest,
            min_filter: euca_rhi::FilterMode::Nearest,
            ..Default::default()
        });
        let lighting_bgl = Self::create_lighting_bgl(device);
        let lighting_buffer = SmartBuffer::new(
            device,
            std::mem::size_of::<DeferredLightingUniforms>() as u64,
            BufferKind::Uniform,
            unified_memory,
            "Deferred Lighting UBO",
        );
        let lighting_bind_group = Self::create_lighting_bind_group(
            device,
            &lighting_bgl,
            &gbuffer,
            &gbuffer_sampler,
            &lighting_buffer,
        );
        let ls = device.create_shader(&euca_rhi::ShaderDesc {
            label: Some("Deferred Lighting Shader"),
            source: euca_rhi::ShaderSource::Wgsl(DEFERRED_LIGHTING_SHADER.into()),
        });
        let lighting_pipeline = device.create_render_pipeline(&euca_rhi::RenderPipelineDesc {
            label: Some("Deferred Lighting Pipeline"),
            layout: &[&lighting_bgl],
            vertex: euca_rhi::VertexState {
                module: &ls,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(euca_rhi::FragmentState {
                module: &ls,
                entry_point: "fs_main",
                targets: &[Some(euca_rhi::ColorTargetState {
                    format: hdr_format,
                    blend: Some(euca_rhi::BlendState::REPLACE),
                    write_mask: euca_rhi::ColorWrites::ALL,
                })],
            }),
            primitive: euca_rhi::PrimitiveState {
                topology: euca_rhi::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
        });
        Self {
            gbuffer,
            gbuffer_pipeline,
            instance_bgl,
            gbuffer_scene_bgl,
            material_bgl,
            instance_buffer,
            instance_bind_group,
            gbuffer_scene_buffer,
            gbuffer_scene_bind_group,
            lighting_pipeline,
            lighting_bgl,
            lighting_bind_group,
            lighting_buffer,
            gbuffer_sampler,
            material_sampler,
            hdr_format,
            instance_capacity: INITIAL_DEFERRED_INSTANCE_CAPACITY,
            unified_memory,
        }
    }
    pub fn resize(&mut self, device: &D, width: u32, height: u32) {
        self.gbuffer.resize(device, width, height);
        self.lighting_bind_group = Self::create_lighting_bind_group(
            device,
            &self.lighting_bgl,
            &self.gbuffer,
            &self.gbuffer_sampler,
            &self.lighting_buffer,
        );
    }
    pub fn material_bgl(&self) -> &D::BindGroupLayout {
        &self.material_bgl
    }
    pub fn material_sampler(&self) -> &D::Sampler {
        &self.material_sampler
    }
    pub fn instance_bgl(&self) -> &D::BindGroupLayout {
        &self.instance_bgl
    }
    pub fn hdr_format(&self) -> euca_rhi::TextureFormat {
        self.hdr_format
    }
    /// Grow the deferred instance buffer if `count` exceeds capacity.
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
            "Deferred Instance SSBO",
        );
        self.instance_bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("Deferred Instance BG"),
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
    pub fn write_instances<T: bytemuck::Pod>(&self, device: &D, data: &[T]) {
        self.instance_buffer.write(device, data);
    }
    pub fn write_gbuffer_scene(&self, device: &D, camera_vp: [[f32; 4]; 4]) {
        self.gbuffer_scene_buffer.write_bytes(
            device,
            bytemuck::bytes_of(&GBufferSceneUniforms { camera_vp }),
        );
    }
    pub fn write_lighting_uniforms(&self, device: &D, uniforms: &DeferredLightingUniforms) {
        self.lighting_buffer
            .write_bytes(device, bytemuck::bytes_of(uniforms));
    }
    pub fn encode_gbuffer_pass<'a, F>(
        &'a self,
        device: &'a D,
        encoder: &'a mut D::CommandEncoder,
        draw_fn: F,
    ) where
        F: FnOnce(&mut D::RenderPass<'a>),
    {
        let mut pass = device.begin_render_pass(
            encoder,
            &euca_rhi::RenderPassDesc {
                label: Some("Deferred G-Buffer Pass"),
                color_attachments: &[
                    Some(euca_rhi::RenderPassColorAttachment {
                        view: &self.gbuffer.albedo_view,
                        resolve_target: None,
                        ops: euca_rhi::Operations {
                            load: euca_rhi::LoadOp::Clear(euca_rhi::Color::BLACK),
                            store: euca_rhi::StoreOp::Store,
                        },
                    }),
                    Some(euca_rhi::RenderPassColorAttachment {
                        view: &self.gbuffer.normal_roughness_view,
                        resolve_target: None,
                        ops: euca_rhi::Operations {
                            load: euca_rhi::LoadOp::Clear(euca_rhi::Color::BLACK),
                            store: euca_rhi::StoreOp::Store,
                        },
                    }),
                    Some(euca_rhi::RenderPassColorAttachment {
                        view: &self.gbuffer.material_view,
                        resolve_target: None,
                        ops: euca_rhi::Operations {
                            load: euca_rhi::LoadOp::Clear(euca_rhi::Color::BLACK),
                            store: euca_rhi::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(euca_rhi::RenderPassDepthStencilAttachment {
                    view: &self.gbuffer.depth_view,
                    depth_ops: Some(euca_rhi::Operations {
                        load: euca_rhi::LoadOp::Clear(1.0),
                        store: euca_rhi::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
            },
        );
        pass.set_pipeline(&self.gbuffer_pipeline);
        pass.set_bind_group(0, &self.instance_bind_group, &[]);
        pass.set_bind_group(1, &self.gbuffer_scene_bind_group, &[]);
        draw_fn(&mut pass);
    }
    pub fn encode_lighting_pass(
        &self,
        device: &D,
        encoder: &mut D::CommandEncoder,
        hdr_view: &D::TextureView,
    ) {
        let mut pass = device.begin_render_pass(
            encoder,
            &euca_rhi::RenderPassDesc {
                label: Some("Deferred Lighting Pass"),
                color_attachments: &[Some(euca_rhi::RenderPassColorAttachment {
                    view: hdr_view,
                    resolve_target: None,
                    ops: euca_rhi::Operations {
                        load: euca_rhi::LoadOp::Clear(euca_rhi::Color::BLACK),
                        store: euca_rhi::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
            },
        );
        pass.set_pipeline(&self.lighting_pipeline);
        pass.set_bind_group(0, &self.lighting_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    fn create_lighting_bgl(device: &D) -> D::BindGroupLayout {
        let tex = |binding, sample_type| euca_rhi::BindGroupLayoutEntry {
            binding,
            visibility: euca_rhi::ShaderStages::FRAGMENT,
            ty: euca_rhi::BindingType::Texture {
                sample_type,
                view_dimension: euca_rhi::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let float_tex = |binding| {
            tex(
                binding,
                euca_rhi::TextureSampleType::Float { filterable: true },
            )
        };
        device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("Deferred Lighting BGL"),
            entries: &[
                float_tex(0),
                float_tex(1),
                float_tex(2),
                tex(3, euca_rhi::TextureSampleType::Depth),
                euca_rhi::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Sampler(euca_rhi::SamplerBindingType::Filtering),
                    count: None,
                },
                euca_rhi::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Buffer {
                        ty: euca_rhi::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
    fn create_lighting_bind_group(
        device: &D,
        layout: &D::BindGroupLayout,
        gbuffer: &GBuffer<D>,
        sampler: &D::Sampler,
        lb: &SmartBuffer<D>,
    ) -> D::BindGroup {
        device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("Deferred Lighting BG"),
            layout,
            entries: &[
                euca_rhi::BindGroupEntry {
                    binding: 0,
                    resource: euca_rhi::BindingResource::TextureView(&gbuffer.albedo_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 1,
                    resource: euca_rhi::BindingResource::TextureView(
                        &gbuffer.normal_roughness_view,
                    ),
                },
                euca_rhi::BindGroupEntry {
                    binding: 2,
                    resource: euca_rhi::BindingResource::TextureView(&gbuffer.material_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 3,
                    resource: euca_rhi::BindingResource::TextureView(&gbuffer.depth_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 4,
                    resource: euca_rhi::BindingResource::Sampler(sampler),
                },
                euca_rhi::BindGroupEntry {
                    binding: 5,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: lb.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gbuffer_format_validation() {
        assert_eq!(GBufferFormats::ALBEDO, euca_rhi::TextureFormat::Rgba8Unorm);
        assert_eq!(
            GBufferFormats::NORMAL_ROUGHNESS,
            euca_rhi::TextureFormat::Rgba16Float
        );
        assert_eq!(
            GBufferFormats::MATERIAL,
            euca_rhi::TextureFormat::Rgba8Unorm
        );
        assert_eq!(GBufferFormats::DEPTH, euca_rhi::TextureFormat::Depth32Float);
        assert_eq!(GBufferFormats::ALL_COLOR.len(), 3);
    }
    #[test]
    fn render_path_selection() {
        assert_eq!(RenderPath::default(), RenderPath::Forward);
        assert_ne!(RenderPath::Forward, RenderPath::Deferred);
        let p = RenderPath::Deferred;
        let c = p;
        assert_eq!(c, RenderPath::Deferred);
    }
    #[test]
    fn pipeline_structure_constants() {
        assert!(MAX_DEFERRED_POINT_LIGHTS >= 128);
        assert!(MAX_DEFERRED_SPOT_LIGHTS >= 32);
        let s = std::mem::size_of::<DeferredLightingUniforms>();
        assert!(s > 0);
        assert_eq!(s % 16, 0);
    }
    #[test]
    fn deferred_lighting_uniforms_pod_valid() {
        let u = DeferredLightingUniforms {
            camera_pos: [0.0; 4],
            light_direction: [0.0, -1.0, 0.0, 0.0],
            light_color: [1.0; 4],
            ambient_color: [0.1, 0.1, 0.1, 0.15],
            inv_vp: [[0.0; 4]; 4],
            screen_size: [1920.0, 1080.0, 1.0 / 1920.0, 1.0 / 1080.0],
            point_lights: [GpuDeferredPointLight::default(); MAX_DEFERRED_POINT_LIGHTS],
            spot_lights: [GpuDeferredSpotLight::default(); MAX_DEFERRED_SPOT_LIGHTS],
            num_point_lights: [0.0; 4],
            num_spot_lights: [0.0; 4],
        };
        assert_eq!(
            bytemuck::bytes_of(&u).len(),
            std::mem::size_of::<DeferredLightingUniforms>()
        );
    }
}
