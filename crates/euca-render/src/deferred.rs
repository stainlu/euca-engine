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
    pub const ALBEDO: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    pub const NORMAL_ROUGHNESS: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    pub const MATERIAL: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    pub const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    pub const ALL_COLOR: [wgpu::TextureFormat; 3] =
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
impl GBuffer {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let mk = |label: &str, format: wgpu::TextureFormat| {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let v = t.create_view(&wgpu::TextureViewDescriptor::default());
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
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
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
    hdr_format: wgpu::TextureFormat,
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

impl DeferredPipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, unified_memory: bool) -> Self {
        let gbuffer = GBuffer::new(device, width, height);
        let hdr_format = wgpu::TextureFormat::Rgba16Float;
        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Deferred Instance BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let ibsz =
            (INITIAL_DEFERRED_INSTANCE_CAPACITY * std::mem::size_of::<[[f32; 4]; 8]>()) as u64;
        let instance_buffer = SmartBuffer::from_wgpu(
            device,
            ibsz,
            BufferKind::Storage,
            unified_memory,
            "Deferred Instance SSBO",
        );
        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Deferred Instance BG"),
            layout: &instance_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buffer.raw().as_entire_binding(),
            }],
        });
        let gbuffer_scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GBuffer Scene BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<GBufferSceneUniforms>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let gbuffer_scene_buffer = SmartBuffer::from_wgpu(
            device,
            std::mem::size_of::<GBufferSceneUniforms>() as u64,
            BufferKind::Uniform,
            unified_memory,
            "GBuffer Scene UBO",
        );
        let gbuffer_scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GBuffer Scene BG"),
            layout: &gbuffer_scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gbuffer_scene_buffer.raw().as_entire_binding(),
            }],
        });
        let te = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Deferred Material BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                te(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                te(3),
                te(4),
                te(5),
                te(6),
            ],
        });
        let material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Deferred Material Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let gs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GBuffer Shader"),
            source: wgpu::ShaderSource::Wgsl(GBUFFER_SHADER.into()),
        });
        let gpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GBuffer Pipeline Layout"),
            bind_group_layouts: &[&instance_bgl, &gbuffer_scene_bgl, &material_bgl],
            push_constant_ranges: &[],
        });
        let gbuffer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GBuffer Pipeline"),
            layout: Some(&gpl),
            vertex: wgpu::VertexState {
                module: &gs,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gs,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: GBufferFormats::ALBEDO,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GBufferFormats::NORMAL_ROUGHNESS,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GBufferFormats::MATERIAL,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: GBufferFormats::DEPTH,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let gbuffer_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("GBuffer Read Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let lighting_bgl = Self::create_lighting_bgl(device);
        let lighting_buffer = SmartBuffer::from_wgpu(
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
        let ls = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Deferred Lighting Shader"),
            source: wgpu::ShaderSource::Wgsl(DEFERRED_LIGHTING_SHADER.into()),
        });
        let lpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Deferred Lighting Pipeline Layout"),
            bind_group_layouts: &[&lighting_bgl],
            push_constant_ranges: &[],
        });
        let lighting_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Deferred Lighting Pipeline"),
            layout: Some(&lpl),
            vertex: wgpu::VertexState {
                module: &ls,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ls,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: hdr_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
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
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.gbuffer.resize(device, width, height);
        self.lighting_bind_group = Self::create_lighting_bind_group(
            device,
            &self.lighting_bgl,
            &self.gbuffer,
            &self.gbuffer_sampler,
            &self.lighting_buffer,
        );
    }
    pub fn material_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.material_bgl
    }
    pub fn material_sampler(&self) -> &wgpu::Sampler {
        &self.material_sampler
    }
    pub fn instance_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.instance_bgl
    }
    pub fn hdr_format(&self) -> wgpu::TextureFormat {
        self.hdr_format
    }
    /// Grow the deferred instance buffer if `count` exceeds capacity.
    pub fn ensure_instance_capacity(&mut self, device: &wgpu::Device, count: usize) {
        if count <= self.instance_capacity {
            return;
        }
        self.instance_capacity = count.next_power_of_two();
        let size = (self.instance_capacity * std::mem::size_of::<[[f32; 4]; 8]>()) as u64;
        self.instance_buffer = SmartBuffer::from_wgpu(
            device,
            size,
            BufferKind::Storage,
            self.unified_memory,
            "Deferred Instance SSBO",
        );
        self.instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Deferred Instance BG"),
            layout: &self.instance_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.instance_buffer.raw().as_entire_binding(),
            }],
        });
    }
    pub fn write_instances<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &[T]) {
        self.instance_buffer.write_wgpu(queue, data);
    }
    pub fn write_gbuffer_scene(&self, queue: &wgpu::Queue, camera_vp: [[f32; 4]; 4]) {
        self.gbuffer_scene_buffer.write_bytes_wgpu(
            queue,
            bytemuck::bytes_of(&GBufferSceneUniforms { camera_vp }),
        );
    }
    pub fn write_lighting_uniforms(
        &self,
        queue: &wgpu::Queue,
        uniforms: &DeferredLightingUniforms,
    ) {
        self.lighting_buffer
            .write_bytes_wgpu(queue, bytemuck::bytes_of(uniforms));
    }
    pub fn encode_gbuffer_pass<'a, F>(&'a self, encoder: &'a mut wgpu::CommandEncoder, draw_fn: F)
    where
        F: FnOnce(&mut wgpu::RenderPass<'a>),
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Deferred G-Buffer Pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.gbuffer.albedo_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.gbuffer.normal_roughness_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.gbuffer.material_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.gbuffer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.gbuffer_pipeline);
        pass.set_bind_group(0, &self.instance_bind_group, &[]);
        pass.set_bind_group(1, &self.gbuffer_scene_bind_group, &[]);
        draw_fn(&mut pass);
    }
    pub fn encode_lighting_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hdr_view: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Deferred Lighting Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&self.lighting_pipeline);
        pass.set_bind_group(0, &self.lighting_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    fn create_lighting_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Deferred Lighting BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
    fn create_lighting_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        gbuffer: &GBuffer,
        sampler: &wgpu::Sampler,
        lb: &SmartBuffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Deferred Lighting BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.normal_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: lb.raw().as_entire_binding(),
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
        assert_eq!(GBufferFormats::ALBEDO, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(
            GBufferFormats::NORMAL_ROUGHNESS,
            wgpu::TextureFormat::Rgba16Float
        );
        assert_eq!(GBufferFormats::MATERIAL, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(GBufferFormats::DEPTH, wgpu::TextureFormat::Depth32Float);
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
