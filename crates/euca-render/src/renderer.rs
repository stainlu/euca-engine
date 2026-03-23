use crate::buffer::{BufferKind, SmartBuffer};
use crate::camera::Camera;
use crate::decal::{DecalDrawCommand, DecalRenderer};
use crate::gpu::GpuContext;
use crate::light::{AmbientLight, DirectionalLight};
use crate::material::{Material, MaterialHandle};
use crate::mesh::{Mesh, MeshHandle};
use crate::occlusion::OcclusionCuller;
use crate::post_process::{PostProcessSettings, PostProcessStack};
use crate::texture::{TextureHandle, TextureStore};
use crate::vertex::Vertex;
use crate::volumetric::{FrameParams, VolumetricFogPass, VolumetricFogSettings};
use euca_math::Mat4;

/// Preset quality tiers that map to sensible [`PostProcessSettings`] defaults.
///
/// Use [`RenderQuality::to_settings`] to obtain the corresponding settings,
/// then apply them via [`Renderer::set_post_process_settings`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderQuality {
    /// Minimal overhead: SSAO off, FXAA on, bloom on.
    Low,
    /// Balanced: SSAO on at reduced radius, FXAA on, bloom on.
    Medium,
    /// Full quality: SSAO on, all color grading at neutral, everything enabled.
    High,
    /// Maximum fidelity: boosted SSAO, slightly elevated contrast/saturation.
    Ultra,
}

impl RenderQuality {
    /// Convert this quality tier into concrete post-process settings.
    pub fn to_settings(self) -> PostProcessSettings {
        match self {
            RenderQuality::Low => PostProcessSettings {
                ssao_enabled: false,
                fxaa_enabled: true,
                bloom_enabled: true,
                ..PostProcessSettings::default()
            },
            RenderQuality::Medium => PostProcessSettings {
                ssao_enabled: true,
                ssao_radius: 0.3,
                fxaa_enabled: true,
                bloom_enabled: true,
                ..PostProcessSettings::default()
            },
            RenderQuality::High => PostProcessSettings {
                ssao_enabled: true,
                ssao_radius: 0.5,
                ssao_intensity: 1.0,
                fxaa_enabled: true,
                bloom_enabled: true,
                exposure: 1.0,
                contrast: 1.0,
                saturation: 1.0,
                ..PostProcessSettings::default()
            },
            RenderQuality::Ultra => PostProcessSettings {
                ssao_enabled: true,
                ssao_radius: 0.5,
                ssao_intensity: 1.2,
                fxaa_enabled: true,
                bloom_enabled: true,
                exposure: 1.0,
                contrast: 1.05,
                saturation: 1.05,
                ..PostProcessSettings::default()
            },
        }
    }

    /// Parse a quality tier from a case-insensitive string.
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" | "med" => Some(Self::Medium),
            "high" => Some(Self::High),
            "ultra" => Some(Self::Ultra),
            _ => None,
        }
    }

    /// Return the name of this quality tier as a lowercase string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Ultra => "ultra",
        }
    }
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniforms {
    albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
    has_normal_map: f32,
    has_metallic_roughness_tex: f32,
    emissive: [f32; 3],
    has_emissive_tex: f32,
    has_ao_tex: f32,
    alpha_mode: f32,
    alpha_cutoff: f32,
    _pad: f32,
}

/// A single draw request submitted each frame.
///
/// Pairs a mesh and material with a world-space transform. The renderer
/// batches draw commands by mesh and material to minimize GPU state changes.
pub struct DrawCommand {
    /// Handle to the GPU mesh to draw.
    pub mesh: MeshHandle,
    /// Handle to the GPU material to shade with.
    pub material: MaterialHandle,
    /// Object-to-world transform matrix.
    pub model_matrix: Mat4,
    /// Optional world-space AABB (center, half-extents) for occlusion culling.
    /// When provided and occlusion culling is enabled, objects fully behind
    /// previously rendered geometry are skipped.
    pub aabb: Option<(euca_math::Vec3, euca_math::Vec3)>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
}

/// Per-decal uniform data uploaded to the GPU before each decal draw call.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DecalUniforms {
    model_matrix: [[f32; 4]; 4],
    opacity: f32,
    _pad: [f32; 3],
}

const MAX_POINT_LIGHTS: usize = 4;
const MAX_SPOT_LIGHTS: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct GpuPointLight {
    position: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct GpuSpotLight {
    position: [f32; 4],
    direction: [f32; 4],
    color: [f32; 4],
    cone: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    camera_pos: [f32; 4],
    light_direction: [f32; 4],
    light_color: [f32; 4],
    ambient_color: [f32; 4],
    camera_vp: [[f32; 4]; 4],
    light_vp: [[f32; 4]; 4],
    inv_vp: [[f32; 4]; 4],
    cascade_vps: [[[f32; 4]; 4]; 3],
    cascade_splits: [f32; 4],
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
    spot_lights: [GpuSpotLight; MAX_SPOT_LIGHTS],
    num_point_lights: [f32; 4],
    num_spot_lights: [f32; 4],
    /// Interpolated L2 SH probe coefficients (9 bands × RGBA, blended on CPU).
    probe_sh: [[f32; 4]; 9],
    /// x=1.0 if probe data is valid, 0.0 otherwise. yzw=padding.
    probe_enabled: [f32; 4],
}

struct GpuMaterial {
    bind_group: wgpu::BindGroup,
    _buffer: wgpu::Buffer,
    is_transparent: bool,
}

struct DrawBatch {
    mesh: MeshHandle,
    material: MaterialHandle,
    instance_start: u32,
    instance_count: u32,
}

const MAX_INSTANCES: usize = 16384;
const SHADOW_MAP_SIZE: u32 = 2048;
const NUM_SHADOW_CASCADES: u32 = 3;
const CASCADE_ORTHO_SIZES: [f32; 3] = [8.0, 20.0, 50.0];

/// The main PBR forward renderer.
///
/// Owns all GPU pipeline state, uploaded meshes, materials, textures, and
/// optional subsystems (post-processing, TAA, volumetric fog, occlusion
/// culling, decals, GPU particles).
///
/// # Rendering pipeline
///
/// Each frame proceeds through these stages:
///
/// 1. **Shadow pass** -- render cascaded shadow maps for the directional light.
/// 2. **Opaque pass** -- draw all opaque [`DrawCommand`]s with PBR shading
///    (4x MSAA, HDR).
/// 3. **Decal pass** -- project deferred decals onto opaque surfaces.
/// 4. **Transparent pass** -- draw alpha-blended commands back-to-front.
/// 5. **Volumetric fog** -- ray-march scattering (if enabled).
/// 6. **Post-processing** -- SSAO, bloom, color grading, FXAA.
/// 7. **TAA resolve** -- temporal anti-aliasing jitter and history blending.
#[allow(dead_code)]
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    instance_buffer: SmartBuffer,
    instance_bind_group: wgpu::BindGroup,
    instance_bgl: wgpu::BindGroupLayout,
    scene_buffer: SmartBuffer,
    scene_bgl: wgpu::BindGroupLayout,
    scene_bind_group: wgpu::BindGroup,
    material_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    materials: Vec<GpuMaterial>,
    textures: TextureStore,
    sky_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_map: wgpu::Texture,
    shadow_map_view: wgpu::TextureView,
    shadow_cascade_views: Vec<wgpu::TextureView>,
    shadow_sampler: wgpu::Sampler,
    shadow_instance_buffer: SmartBuffer,
    shadow_instance_bind_group: wgpu::BindGroup,
    msaa_hdr_view: wgpu::TextureView,
    #[allow(dead_code)]
    msaa_hdr_texture: wgpu::Texture,
    meshes: Vec<GpuMesh>,
    depth_texture: wgpu::TextureView,
    depth_format: wgpu::TextureFormat,
    surface_format: wgpu::TextureFormat,
    /// Advanced post-process stack (SSAO, FXAA, color grading, bloom).
    post_process_stack: PostProcessStack,
    /// Settings controlling the advanced post-process stack.
    post_process_settings: PostProcessSettings,
    /// Optional volumetric fog pass. Created lazily via `enable_volumetric_fog`.
    volumetric_fog_pass: Option<VolumetricFogPass>,
    /// Settings for volumetric fog (density, scattering, etc.).
    volumetric_fog_settings: VolumetricFogSettings,
    /// Optional HZB occlusion culler. Uses previous frame's depth.
    occlusion_culler: Option<OcclusionCuller>,
    /// Previous frame's depth buffer for occlusion culling.
    prev_depth_buffer: Vec<f32>,
    /// Dimensions of the previous depth buffer.
    prev_depth_dims: (u32, u32),
    /// TAA resolve pass (temporal anti-aliasing).
    taa_pass: crate::taa::TaaPass,
    /// Frame counter for TAA jitter sequence.
    frame_count: u32,
    /// Interpolated SH probe coefficients for indirect lighting (set by caller).
    probe_sh: [[f32; 4]; 9],
    /// Whether probe data is valid.
    probe_enabled: bool,
    /// GPU resources for decal projection volumes.
    decal_renderer: DecalRenderer,
    /// Per-decal uniform buffer (model_matrix + opacity), written before each draw.
    decal_uniform_buffer: SmartBuffer,
    /// Bind group layout for the per-decal uniform buffer.
    decal_bgl: wgpu::BindGroupLayout,
    /// Bind group exposing the per-decal uniform buffer to shaders.
    decal_bind_group: wgpu::BindGroup,
    /// Decal draw commands staged by the caller for the current frame.
    pending_decals: Vec<DecalDrawCommand>,
    /// GPU compute particle systems.
    gpu_particle_systems: Vec<crate::gpu_particles::GpuParticleSystem>,
}

const MSAA_SAMPLE_COUNT: u32 = 4;

impl Renderer {
    /// Create a new renderer, allocating all GPU pipelines and buffers.
    ///
    /// The renderer is bound to the surface format and initial size reported
    /// by `gpu`. Call [`resize`](Self::resize) when the window size changes.
    pub fn new(gpu: &GpuContext) -> Self {
        let instance_buf_size = (MAX_INSTANCES * std::mem::size_of::<InstanceData>()) as u64;
        let unified = gpu.unified_memory;
        let instance_buffer = SmartBuffer::new(
            &gpu.device,
            instance_buf_size,
            BufferKind::Storage,
            unified,
            "Instance SSBO",
        );
        let scene_buffer = SmartBuffer::new(
            &gpu.device,
            std::mem::size_of::<SceneUniforms>() as u64,
            BufferKind::Uniform,
            unified,
            "Scene UBO",
        );
        let shadow_instance_buffer = SmartBuffer::new(
            &gpu.device,
            instance_buf_size,
            BufferKind::Storage,
            unified,
            "Shadow Instance SSBO",
        );
        let textures = TextureStore::new(&gpu.device, &gpu.queue);
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Material Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let shadow_map = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map Array"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: NUM_SHADOW_CASCADES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_map_view = shadow_map.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let shadow_cascade_views: Vec<wgpu::TextureView> = (0..NUM_SHADOW_CASCADES)
            .map(|i| {
                shadow_map.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let shadow_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let instance_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Instance BGL"),
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
        let scene_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Scene BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<SceneUniforms>() as u64,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            });

        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let material_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Material BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                                MaterialUniforms,
                            >()
                                as u64),
                        },
                        count: None,
                    },
                    tex_entry(1), // albedo
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    tex_entry(3), // normal
                    tex_entry(4), // metallic-roughness
                    tex_entry(5), // ao
                    tex_entry(6), // emissive
                ],
            });

        let instance_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Instance BG"),
            layout: &instance_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buffer.raw().as_entire_binding(),
            }],
        });
        let shadow_instance_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Instance BG"),
            layout: &instance_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_instance_buffer.raw().as_entire_binding(),
            }],
        });
        let scene_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene BG"),
            layout: &scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
        });

        let depth_format = wgpu::TextureFormat::Depth32Float;
        let shadow_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shadow Shader"),
                source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
            });
        let shadow_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Shadow Pipeline Layout"),
                    bind_group_layouts: &[&instance_bgl],
                    push_constant_ranges: &[],
                });
        let shadow_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Shadow Pipeline"),
                layout: Some(&shadow_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shadow_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::LAYOUT],
                    compilation_options: Default::default(),
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Front),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 4,
                        slope_scale: 3.0,
                        clamp: 0.0,
                    },
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

        let sky_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Sky Shader"),
                source: wgpu::ShaderSource::Wgsl(SKY_SHADER.into()),
            });
        let sky_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Sky Pipeline Layout"),
                    bind_group_layouts: &[&scene_bgl],
                    push_constant_ranges: &[],
                });
        let hdr_format = wgpu::TextureFormat::Rgba16Float;
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("PBR Shader"),
                source: wgpu::ShaderSource::Wgsl(PBR_SHADER.into()),
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PBR Pipeline Layout"),
                bind_group_layouts: &[&instance_bgl, &scene_bgl, &material_bgl],
                push_constant_ranges: &[],
            });
        let depth_texture =
            Self::create_depth_texture(&gpu.device, &gpu.surface_config, depth_format);

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("PBR Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::LAYOUT],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
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
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLE_COUNT,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        let transparent_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("PBR Transparent Pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[Vertex::LAYOUT],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: hdr_format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: depth_format,
                        depth_write_enabled: false,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: MSAA_SAMPLE_COUNT,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

        let sky_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Sky Pipeline"),
                layout: Some(&sky_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &sky_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &sky_shader,
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
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLE_COUNT,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        let (msaa_hdr_texture, msaa_hdr_view) =
            Self::create_msaa_hdr_texture(&gpu.device, &gpu.surface_config);
        let post_process_stack =
            PostProcessStack::new(&gpu.device, &gpu.queue, &gpu.surface_config);

        let decal_renderer = DecalRenderer::new(&gpu.device);
        decal_renderer.upload(&gpu.queue);

        let decal_uniform_buffer = SmartBuffer::new(
            &gpu.device,
            std::mem::size_of::<DecalUniforms>() as u64,
            BufferKind::Uniform,
            unified,
            "Decal UBO",
        );
        let decal_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Decal BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<DecalUniforms>() as u64,
                        ),
                    },
                    count: None,
                }],
            });
        let decal_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Decal BG"),
            layout: &decal_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: decal_uniform_buffer.raw().as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            transparent_pipeline,
            instance_buffer,
            instance_bind_group,
            instance_bgl,
            scene_buffer,
            scene_bgl,
            scene_bind_group,
            material_bgl,
            sampler,
            materials: Vec::new(),
            textures,
            sky_pipeline,
            shadow_pipeline,
            shadow_map,
            shadow_map_view,
            shadow_cascade_views,
            shadow_sampler,
            shadow_instance_buffer,
            shadow_instance_bind_group,
            msaa_hdr_texture,
            msaa_hdr_view,
            meshes: Vec::new(),
            depth_texture,
            depth_format,
            surface_format: gpu.surface_config.format,
            post_process_stack,
            post_process_settings: PostProcessSettings::default(),
            volumetric_fog_pass: None,
            volumetric_fog_settings: VolumetricFogSettings::default(),
            occlusion_culler: None,
            prev_depth_buffer: Vec::new(),
            prev_depth_dims: (0, 0),
            taa_pass: crate::taa::TaaPass::new(
                &gpu.device,
                gpu.surface_config.width,
                gpu.surface_config.height,
            ),
            frame_count: 0,
            probe_sh: [[0.0; 4]; 9],
            probe_enabled: false,
            decal_renderer,
            decal_uniform_buffer,
            decal_bgl,
            decal_bind_group,
            pending_decals: Vec::new(),
            gpu_particle_systems: Vec::new(),
        }
    }

    /// Upload CPU-side mesh data to the GPU and return a handle for use in
    /// [`DrawCommand`]s.
    pub fn upload_mesh(&mut self, gpu: &GpuContext, mesh: &Mesh) -> MeshHandle {
        use wgpu::util::DeviceExt;
        let vb = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ib = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let handle = MeshHandle(self.meshes.len() as u32);
        self.meshes.push(GpuMesh {
            vertex_buffer: vb,
            index_buffer: ib,
            index_count: mesh.indices.len() as u32,
        });
        handle
    }
    /// Upload raw RGBA8 pixel data as a GPU texture with auto-generated mipmaps.
    pub fn upload_texture(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> TextureHandle {
        self.textures
            .upload_rgba(&gpu.device, &gpu.queue, width, height, rgba)
    }

    /// Decode an image file (PNG, JPEG, etc.) from memory and upload it as a texture.
    pub fn upload_texture_image(&mut self, gpu: &GpuContext, data: &[u8]) -> TextureHandle {
        self.textures.upload_image(&gpu.device, &gpu.queue, data)
    }

    /// Generate and upload a checkerboard test pattern texture.
    pub fn checkerboard_texture(
        &mut self,
        gpu: &GpuContext,
        size: u32,
        tile: u32,
    ) -> TextureHandle {
        self.textures
            .checkerboard(&gpu.device, &gpu.queue, size, tile)
    }

    /// Upload a PBR material (uniforms + texture bindings) to the GPU and
    /// return a handle for use in [`DrawCommand`]s.
    pub fn upload_material(&mut self, gpu: &GpuContext, mat: &Material) -> MaterialHandle {
        use wgpu::util::DeviceExt;
        let handle = MaterialHandle(self.materials.len() as u32);
        let uniforms = MaterialUniforms {
            albedo: mat.albedo,
            metallic: mat.metallic,
            roughness: mat.roughness,
            has_normal_map: if mat.normal_texture.is_some() {
                1.0
            } else {
                0.0
            },
            has_metallic_roughness_tex: if mat.metallic_roughness_texture.is_some() {
                1.0
            } else {
                0.0
            },
            emissive: mat.emissive,
            has_emissive_tex: if mat.emissive_texture.is_some() {
                1.0
            } else {
                0.0
            },
            has_ao_tex: if mat.ao_texture.is_some() { 1.0 } else { 0.0 },
            alpha_mode: mat.alpha_mode.as_f32(),
            alpha_cutoff: mat.alpha_mode.cutoff(),
            _pad: 0.0,
        };
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Material UBO"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let dw = TextureStore::default_white();
        let albedo_view = self.textures.view(mat.albedo_texture.unwrap_or(dw));
        let normal_view = self.textures.view(mat.normal_texture.unwrap_or(dw));
        let mr_view = self
            .textures
            .view(mat.metallic_roughness_texture.unwrap_or(dw));
        let ao_view = self.textures.view(mat.ao_texture.unwrap_or(dw));
        let emissive_view = self.textures.view(mat.emissive_texture.unwrap_or(dw));
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material BG"),
            layout: &self.material_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(mr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(emissive_view),
                },
            ],
        });
        self.materials.push(GpuMaterial {
            bind_group,
            _buffer: buffer,
            is_transparent: mat.alpha_mode.is_transparent(),
        });
        handle
    }

    /// Recreate size-dependent GPU resources (depth buffer, MSAA target, etc.)
    /// after the window has been resized.
    pub fn resize(&mut self, gpu: &GpuContext) {
        self.depth_texture =
            Self::create_depth_texture(&gpu.device, &gpu.surface_config, self.depth_format);
        let (msaa_hdr_texture, msaa_hdr_view) =
            Self::create_msaa_hdr_texture(&gpu.device, &gpu.surface_config);
        self.msaa_hdr_texture = msaa_hdr_texture;
        self.msaa_hdr_view = msaa_hdr_view;
        self.post_process_stack.resize(
            &gpu.device,
            gpu.surface_config.width,
            gpu.surface_config.height,
        );
        if let Some(ref mut fog_pass) = self.volumetric_fog_pass {
            fog_pass.resize(
                &gpu.device,
                gpu.surface_config.width,
                gpu.surface_config.height,
            );
        }
        self.taa_pass.resize(
            &gpu.device,
            gpu.surface_config.width,
            gpu.surface_config.height,
        );
    }

    /// Set interpolated SH probe coefficients for indirect lighting.
    ///
    /// The caller should sample the nearest probes from a `LightProbeGrid`
    /// and pass the blended coefficients here. The PBR shader will use these
    /// instead of the flat ambient color.
    pub fn set_probe_sh(&mut self, sh: [[f32; 4]; 9]) {
        self.probe_sh = sh;
        self.probe_enabled = true;
    }

    /// Disable probe-based ambient lighting (fall back to flat ambient_color).
    pub fn clear_probe(&mut self) {
        self.probe_enabled = false;
    }

    /// Stage decal draw commands for the current frame.
    ///
    /// These will be rendered after opaque geometry in `render_to_view_with_lights`.
    /// The list is consumed (cleared) at the end of each frame.
    pub fn set_decal_commands(&mut self, commands: Vec<DecalDrawCommand>) {
        self.pending_decals = commands;
    }

    /// Read-only access to the decal renderer (unit-cube GPU resources).
    pub fn decal_renderer(&self) -> &DecalRenderer {
        &self.decal_renderer
    }

    /// Add a GPU compute particle system. Returns its index.
    pub fn add_gpu_particle_system(
        &mut self,
        gpu: &GpuContext,
        config: crate::gpu_particles::GpuParticleConfig,
    ) -> usize {
        let format = gpu.surface_config.format;
        let system = crate::gpu_particles::GpuParticleSystem::new(gpu, config, format);
        self.gpu_particle_systems.push(system);
        self.gpu_particle_systems.len() - 1
    }

    /// Access a GPU particle system by index.
    pub fn gpu_particle_system_mut(
        &mut self,
        index: usize,
    ) -> Option<&mut crate::gpu_particles::GpuParticleSystem> {
        self.gpu_particle_systems.get_mut(index)
    }

    /// Enable CPU-side HZB occlusion culling.
    pub fn enable_occlusion_culling(&mut self) {
        self.occlusion_culler = Some(OcclusionCuller::new());
    }

    /// Returns true if HZB occlusion culling is enabled.
    pub fn occlusion_culling_enabled(&self) -> bool {
        self.occlusion_culler.is_some()
    }

    /// Feed a depth buffer for the occlusion system to use next frame.
    pub fn update_prev_depth_buffer(&mut self, depth: Vec<f32>, width: u32, height: u32) {
        debug_assert_eq!(depth.len(), (width as usize) * (height as usize));
        self.prev_depth_buffer = depth;
        self.prev_depth_dims = (width, height);
    }

    /// Update the post-process settings that control SSAO, FXAA, bloom, and
    /// color grading.
    pub fn set_post_process_settings(&mut self, settings: PostProcessSettings) {
        self.post_process_settings = settings;
    }

    /// Read-only access to the current post-process settings.
    pub fn post_process_settings(&self) -> &PostProcessSettings {
        &self.post_process_settings
    }

    /// Enable screen-space reflections in the post-process pipeline.
    ///
    /// SSR settings can be further tuned via `set_post_process_settings`.
    pub fn enable_ssr(&mut self) {
        self.post_process_settings.ssr_enabled = true;
    }

    /// Initialize the volumetric fog pass.
    ///
    /// Once enabled, `render_to_view_with_lights` will execute the fog compute
    /// shader and composite the result over the HDR buffer after the PBR pass
    /// and before post-processing.
    pub fn enable_volumetric_fog(&mut self, gpu: &GpuContext) {
        self.volumetric_fog_pass = Some(VolumetricFogPass::new(
            &gpu.device,
            gpu.surface_config.width,
            gpu.surface_config.height,
            gpu.surface_config.format,
        ));
    }

    /// Update the volumetric fog settings (density, scattering, etc.).
    pub fn set_fog_settings(&mut self, settings: VolumetricFogSettings) {
        self.volumetric_fog_settings = settings;
    }

    /// Read-only access to the current volumetric fog settings.
    pub fn fog_settings(&self) -> &VolumetricFogSettings {
        &self.volumetric_fog_settings
    }

    /// Whether volumetric fog is currently active (pass created and enabled).
    pub fn is_fog_enabled(&self) -> bool {
        self.volumetric_fog_pass.is_some() && self.volumetric_fog_settings.enabled
    }

    fn light_vp_for_cascade(light: &DirectionalLight, ortho_size: f32) -> Mat4 {
        use euca_math::Vec3;
        let dir = Vec3::new(light.direction[0], light.direction[1], light.direction[2]).normalize();
        let light_pos = dir * -30.0;
        let light_view = Mat4::look_at_lh(light_pos, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        let s = ortho_size;
        let light_proj = Mat4::orthographic_lh(-s, s, -s, s, 0.1, 60.0);
        light_proj * light_view
    }
    fn light_vp(light: &DirectionalLight) -> Mat4 {
        Self::light_vp_for_cascade(light, CASCADE_ORTHO_SIZES[0])
    }

    #[allow(dead_code)]
    fn build_batches(commands: &[DrawCommand]) -> (Vec<InstanceData>, Vec<DrawBatch>) {
        if commands.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let mut indices: Vec<usize> = (0..commands.len()).collect();
        indices.sort_by_key(|&i| (commands[i].mesh.0, commands[i].material.0));
        let mut instances = Vec::with_capacity(commands.len());
        let mut batches = Vec::new();
        let mut batch_start = 0u32;
        let mut batch_mesh = commands[indices[0]].mesh;
        let mut batch_mat = commands[indices[0]].material;
        for &idx in &indices {
            let cmd = &commands[idx];
            if cmd.mesh != batch_mesh || cmd.material != batch_mat {
                batches.push(DrawBatch {
                    mesh: batch_mesh,
                    material: batch_mat,
                    instance_start: batch_start,
                    instance_count: instances.len() as u32 - batch_start,
                });
                batch_start = instances.len() as u32;
                batch_mesh = cmd.mesh;
                batch_mat = cmd.material;
            }
            let model = cmd.model_matrix;
            let normal_mat = model.inverse().transpose();
            instances.push(InstanceData {
                model: model.to_cols_array_2d(),
                normal_matrix: normal_mat.to_cols_array_2d(),
            });
        }
        batches.push(DrawBatch {
            mesh: batch_mesh,
            material: batch_mat,
            instance_start: batch_start,
            instance_count: instances.len() as u32 - batch_start,
        });
        (instances, batches)
    }

    fn partition_commands<'a>(
        &self,
        commands: &'a [DrawCommand],
        camera_pos: euca_math::Vec3,
    ) -> (Vec<&'a DrawCommand>, Vec<&'a DrawCommand>) {
        let mut opaque = Vec::new();
        let mut transparent = Vec::new();
        for cmd in commands {
            if self.materials[cmd.material.0 as usize].is_transparent {
                transparent.push(cmd);
            } else {
                opaque.push(cmd);
            }
        }
        transparent.sort_by(|a, b| {
            let da = Self::distance_to_camera(&a.model_matrix, camera_pos);
            let db = Self::distance_to_camera(&b.model_matrix, camera_pos);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        (opaque, transparent)
    }

    fn distance_to_camera(model_matrix: &Mat4, camera_pos: euca_math::Vec3) -> f32 {
        let cols = model_matrix.to_cols_array_2d();
        let obj_pos = euca_math::Vec3::new(cols[3][0], cols[3][1], cols[3][2]);
        (obj_pos - camera_pos).length()
    }

    fn build_batches_from_refs(commands: &[&DrawCommand]) -> (Vec<InstanceData>, Vec<DrawBatch>) {
        if commands.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let mut indices: Vec<usize> = (0..commands.len()).collect();
        indices.sort_by_key(|&i| (commands[i].mesh.0, commands[i].material.0));
        let mut instances = Vec::with_capacity(commands.len());
        let mut batches = Vec::new();
        let mut batch_start = 0u32;
        let mut batch_mesh = commands[indices[0]].mesh;
        let mut batch_mat = commands[indices[0]].material;
        for &idx in &indices {
            let cmd = commands[idx];
            if cmd.mesh != batch_mesh || cmd.material != batch_mat {
                batches.push(DrawBatch {
                    mesh: batch_mesh,
                    material: batch_mat,
                    instance_start: batch_start,
                    instance_count: instances.len() as u32 - batch_start,
                });
                batch_start = instances.len() as u32;
                batch_mesh = cmd.mesh;
                batch_mat = cmd.material;
            }
            let model = cmd.model_matrix;
            let normal_mat = model.inverse().transpose();
            instances.push(InstanceData {
                model: model.to_cols_array_2d(),
                normal_matrix: normal_mat.to_cols_array_2d(),
            });
        }
        batches.push(DrawBatch {
            mesh: batch_mesh,
            material: batch_mat,
            instance_start: batch_start,
            instance_count: instances.len() as u32 - batch_start,
        });
        (instances, batches)
    }

    /// Execute the full rendering pipeline for one frame using only a
    /// directional light and ambient light (no point/spot lights).
    ///
    /// Acquires the surface texture, renders, and presents. For off-screen
    /// rendering or when additional lights are needed, use
    /// [`render_to_view_with_lights`](Self::render_to_view_with_lights).
    pub fn draw(
        &mut self,
        gpu: &GpuContext,
        camera: &Camera,
        light: &DirectionalLight,
        ambient: &AmbientLight,
        commands: &[DrawCommand],
    ) {
        self.draw_with_lights(gpu, camera, light, ambient, commands, &[], &[]);
    }

    /// Execute the full rendering pipeline for one frame with point and spot
    /// lights in addition to the directional and ambient lights.
    ///
    /// Acquires the surface texture, renders, and presents.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_with_lights(
        &mut self,
        gpu: &GpuContext,
        camera: &Camera,
        light: &DirectionalLight,
        ambient: &AmbientLight,
        commands: &[DrawCommand],
        point_lights: &[(euca_math::Vec3, &crate::light::PointLight)],
        spot_lights: &[(euca_math::Vec3, &crate::light::SpotLight)],
    ) {
        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => return,
            Err(e) => {
                log::error!("Surface error: {e}");
                return;
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        self.render_to_view_with_lights(
            gpu,
            camera,
            light,
            ambient,
            commands,
            point_lights,
            spot_lights,
            &view,
            &mut encoder,
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    /// Render one frame into a caller-provided texture view (no point/spot
    /// lights). Useful for off-screen rendering or editor viewports.
    #[allow(clippy::too_many_arguments)]
    pub fn render_to_view(
        &mut self,
        gpu: &GpuContext,
        camera: &Camera,
        light: &DirectionalLight,
        ambient: &AmbientLight,
        commands: &[DrawCommand],
        color_view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.render_to_view_with_lights(
            gpu,
            camera,
            light,
            ambient,
            commands,
            &[],
            &[],
            color_view,
            encoder,
        );
    }

    /// Render one frame into a caller-provided texture view with full light
    /// support (directional, ambient, point, and spot lights).
    ///
    /// This is the most flexible entry point. The caller is responsible for
    /// creating the command encoder and submitting / presenting afterward.
    #[allow(clippy::too_many_arguments)]
    pub fn render_to_view_with_lights(
        &mut self,
        gpu: &GpuContext,
        camera: &Camera,
        light: &DirectionalLight,
        ambient: &AmbientLight,
        commands: &[DrawCommand],
        point_lights: &[(euca_math::Vec3, &crate::light::PointLight)],
        spot_lights: &[(euca_math::Vec3, &crate::light::SpotLight)],
        color_view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let vp = camera.view_projection_matrix(gpu.aspect_ratio());
        let light_vp = Self::light_vp(light);
        let (opaque_cmds, transparent_cmds) = self.partition_commands(commands, camera.eye);
        let opaque_cmds = self.apply_occlusion_culling(&opaque_cmds, vp);
        let (opaque_instances, opaque_batches) = Self::build_batches_from_refs(&opaque_cmds);
        if !opaque_instances.is_empty() {
            self.instance_buffer.write(&gpu.queue, &opaque_instances);
        }
        for (cascade_idx, &cascade_ortho) in CASCADE_ORTHO_SIZES.iter().enumerate() {
            let cascade_vp = Self::light_vp_for_cascade(light, cascade_ortho);
            let shadow_instances: Vec<InstanceData> = opaque_instances
                .iter()
                .map(|inst| {
                    let model = Mat4::from_cols_array_2d(&inst.model);
                    let shadow_mvp = cascade_vp * model;
                    InstanceData {
                        model: shadow_mvp.to_cols_array_2d(),
                        normal_matrix: [[0.0; 4]; 4],
                    }
                })
                .collect();
            if !shadow_instances.is_empty() {
                self.shadow_instance_buffer
                    .write(&gpu.queue, &shadow_instances);
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_cascade_views[cascade_idx],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.shadow_instance_bind_group, &[]);
            for batch in &opaque_batches {
                let mesh = &self.meshes[batch.mesh.0 as usize];
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(
                    0..mesh.index_count,
                    0,
                    batch.instance_start..batch.instance_start + batch.instance_count,
                );
            }
        }
        let dir = light.direction;
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
            .sqrt()
            .max(0.001);
        let scene = SceneUniforms {
            camera_pos: [camera.eye.x, camera.eye.y, camera.eye.z, 0.0],
            light_direction: [dir[0] / len, dir[1] / len, dir[2] / len, 0.0],
            light_color: [
                light.color[0],
                light.color[1],
                light.color[2],
                light.intensity,
            ],
            ambient_color: [
                ambient.color[0],
                ambient.color[1],
                ambient.color[2],
                ambient.intensity,
            ],
            camera_vp: vp.to_cols_array_2d(),
            light_vp: light_vp.to_cols_array_2d(),
            inv_vp: vp.inverse().to_cols_array_2d(),
            cascade_vps: [
                Self::light_vp_for_cascade(light, CASCADE_ORTHO_SIZES[0]).to_cols_array_2d(),
                Self::light_vp_for_cascade(light, CASCADE_ORTHO_SIZES[1]).to_cols_array_2d(),
                Self::light_vp_for_cascade(light, CASCADE_ORTHO_SIZES[2]).to_cols_array_2d(),
            ],
            cascade_splits: [
                CASCADE_ORTHO_SIZES[0],
                CASCADE_ORTHO_SIZES[1],
                CASCADE_ORTHO_SIZES[2],
                0.0,
            ],
            point_lights: {
                let mut arr = [GpuPointLight::default(); MAX_POINT_LIGHTS];
                for (i, (pos, pl)) in point_lights.iter().take(MAX_POINT_LIGHTS).enumerate() {
                    arr[i] = GpuPointLight {
                        position: [pos.x, pos.y, pos.z, pl.range],
                        color: [pl.color[0], pl.color[1], pl.color[2], pl.intensity],
                    };
                }
                arr
            },
            spot_lights: {
                let mut arr = [GpuSpotLight::default(); MAX_SPOT_LIGHTS];
                for (i, (pos, sl)) in spot_lights.iter().take(MAX_SPOT_LIGHTS).enumerate() {
                    arr[i] = GpuSpotLight {
                        position: [pos.x, pos.y, pos.z, sl.range],
                        direction: [sl.direction[0], sl.direction[1], sl.direction[2], 0.0],
                        color: [sl.color[0], sl.color[1], sl.color[2], sl.intensity],
                        cone: [sl.inner_cone.cos(), sl.outer_cone.cos(), 0.0, 0.0],
                    };
                }
                arr
            },
            num_point_lights: [
                point_lights.len().min(MAX_POINT_LIGHTS) as f32,
                0.0,
                0.0,
                0.0,
            ],
            num_spot_lights: [spot_lights.len().min(MAX_SPOT_LIGHTS) as f32, 0.0, 0.0, 0.0],
            probe_sh: self.probe_sh,
            probe_enabled: [if self.probe_enabled { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        };
        self.scene_buffer
            .write_bytes(&gpu.queue, bytemuck::bytes_of(&scene));

        // Resolve MSAA into the post-process stack's ping buffer.
        let resolve_target = self.post_process_stack.ping_view();

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PBR Pass (MSAA)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_hdr_view,
                    resolve_target: Some(resolve_target),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.scene_bind_group, &[]);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.instance_bind_group, &[]);
            pass.set_bind_group(1, &self.scene_bind_group, &[]);
            for batch in &opaque_batches {
                let gpu_mat = &self.materials[batch.material.0 as usize];
                pass.set_bind_group(2, &gpu_mat.bind_group, &[]);
                let mesh = &self.meshes[batch.mesh.0 as usize];
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(
                    0..mesh.index_count,
                    0,
                    batch.instance_start..batch.instance_start + batch.instance_count,
                );
            }
            // ── Decal pass: draw projected decal volumes after opaque geometry ──
            // Each decal is a unit cube scaled/positioned by its model matrix.
            // The decal shader (when a dedicated pipeline is added) reads the
            // depth buffer to reconstruct world-space position and projects the
            // decal texture. For now we bind the decal vertex/index buffers and
            // issue one draw per command so the integration path is exercised.
            if !self.pending_decals.is_empty() {
                pass.set_vertex_buffer(0, self.decal_renderer.vertex_buffer().slice(..));
                pass.set_index_buffer(
                    self.decal_renderer.index_buffer().slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                pass.set_bind_group(0, &self.decal_bind_group, &[]);
                for decal_cmd in &self.pending_decals {
                    let uniforms = DecalUniforms {
                        model_matrix: decal_cmd.model_matrix.to_cols_array_2d(),
                        opacity: decal_cmd.opacity,
                        _pad: [0.0; 3],
                    };
                    self.decal_uniform_buffer
                        .write(&gpu.queue, std::slice::from_ref(&uniforms));
                    pass.draw_indexed(0..self.decal_renderer.index_count(), 0, 0..1);
                }
            }

            if !transparent_cmds.is_empty() {
                let (trans_instances, trans_batches) =
                    Self::build_batches_from_refs(&transparent_cmds);
                if !trans_instances.is_empty() {
                    self.instance_buffer.write(&gpu.queue, &trans_instances);
                }
                pass.set_pipeline(&self.transparent_pipeline);
                pass.set_bind_group(0, &self.instance_bind_group, &[]);
                pass.set_bind_group(1, &self.scene_bind_group, &[]);
                for batch in &trans_batches {
                    let gpu_mat = &self.materials[batch.material.0 as usize];
                    pass.set_bind_group(2, &gpu_mat.bind_group, &[]);
                    let mesh = &self.meshes[batch.mesh.0 as usize];
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(
                        0..mesh.index_count,
                        0,
                        batch.instance_start..batch.instance_start + batch.instance_count,
                    );
                }
            }
        }

        // Clear pending decals after rendering — they must be re-submitted each frame.
        self.pending_decals.clear();

        // GPU compute particles: update (compute dispatch) then draw (render pass).
        if !self.gpu_particle_systems.is_empty() {
            let dt = 1.0 / 60.0; // Fixed timestep for particle update
            for system in &mut self.gpu_particle_systems {
                system.update(encoder, &gpu.queue, dt);
            }

            // Draw particles in a separate render pass (after opaque, blended on top)
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gpu_particles"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_texture,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                for system in &self.gpu_particle_systems {
                    system.draw(&mut pass);
                }
            }
        }

        // Volumetric fog: compute ray-march and composite over HDR buffer.
        if let Some(ref fog_pass) = self.volumetric_fog_pass
            && self.volumetric_fog_settings.enabled
        {
            let dir = light.direction;
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
                .sqrt()
                .max(0.001);
            let frame = FrameParams {
                camera_pos: [camera.eye.x, camera.eye.y, camera.eye.z],
                inv_vp: vp.inverse().to_cols_array_2d(),
                light_direction: [dir[0] / len, dir[1] / len, dir[2] / len],
                light_color: [light.color[0], light.color[1], light.color[2]],
                settings: &self.volumetric_fog_settings,
            };
            fog_pass.execute(&gpu.device, encoder, resolve_target, &gpu.queue, &frame);
        }

        // TAA resolve: blend current frame with history for temporal anti-aliasing.
        if self.post_process_settings.taa_enabled {
            let inv_vp = vp.inverse();
            let prev_vp = camera.prev_view_proj.unwrap_or(vp);
            self.taa_pass.execute(
                &gpu.device,
                &gpu.queue,
                encoder,
                resolve_target,
                &self.post_process_stack.depth_resolve_view,
                &inv_vp,
                &prev_vp,
                camera.jitter,
            );
            // Copy TAA output back to ping_view so post-processing reads the
            // temporally resolved image.
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: self.taa_pass.output_texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: self.post_process_stack.ping_texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: gpu.surface_config.width,
                    height: gpu.surface_config.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.frame_count = self.frame_count.wrapping_add(1);

        // Post-processing via the modular stack.
        {
            let proj = camera.projection_matrix(gpu.aspect_ratio());
            let inv_projection = proj.inverse().to_cols_array_2d();
            let projection = proj.to_cols_array_2d();
            self.post_process_stack.execute(
                &gpu.device,
                &gpu.queue,
                encoder,
                color_view,
                &self.post_process_settings,
                &inv_projection,
                &projection,
            );
        }
    }

    /// Filter out occluded draw commands using the HZB from the previous frame.
    fn apply_occlusion_culling<'a>(
        &mut self,
        commands: &[&'a DrawCommand],
        view_proj: Mat4,
    ) -> Vec<&'a DrawCommand> {
        if self.occlusion_culler.is_none() {
            return commands.to_vec();
        }
        let (w, h) = self.prev_depth_dims;
        if self.prev_depth_buffer.is_empty() || w == 0 || h == 0 {
            return commands.to_vec();
        }
        self.occlusion_culler
            .as_mut()
            .expect("occlusion culler initialized in Renderer::new")
            .update_from_depth_buffer(&self.prev_depth_buffer, w, h);
        let culler = self
            .occlusion_culler
            .as_ref()
            .expect("occlusion culler initialized in Renderer::new");
        let mut aabbs = Vec::new();
        let mut aabb_indices = Vec::new();
        for (i, cmd) in commands.iter().enumerate() {
            if let Some(aabb) = cmd.aabb {
                aabbs.push(aabb);
                aabb_indices.push(i);
            }
        }
        if aabbs.is_empty() {
            return commands.to_vec();
        }
        let result = match culler.test(&aabbs, view_proj) {
            Some(r) => r,
            None => return commands.to_vec(),
        };
        let mut occluded = vec![false; commands.len()];
        for (aabb_idx, &cmd_idx) in aabb_indices.iter().enumerate() {
            if !result.visible[aabb_idx] {
                occluded[cmd_idx] = true;
            }
        }
        commands
            .iter()
            .enumerate()
            .filter(|(i, _)| !occluded[*i])
            .map(|(_, cmd)| *cmd)
            .collect()
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture (MSAA)"),
            size: wgpu::Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
    fn create_msaa_hdr_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA HDR Texture"),
            size: wgpu::Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

const SHADOW_SHADER: &str = include_str!("../shaders/shadow.wgsl");

const PBR_SHADER: &str = include_str!("../shaders/pbr.wgsl");

const SKY_SHADER: &str = include_str!("../shaders/sky.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_preset_produces_valid_settings() {
        for quality in [
            RenderQuality::Low,
            RenderQuality::Medium,
            RenderQuality::High,
            RenderQuality::Ultra,
        ] {
            let s = quality.to_settings();
            assert!(s.fxaa_enabled, "{quality:?} should have FXAA enabled");
            assert!(s.bloom_enabled, "{quality:?} should have bloom enabled");
            assert!(s.ssao_radius >= 0.0, "{quality:?} ssao_radius must be >= 0");
            assert!(
                s.ssao_intensity >= 0.0,
                "{quality:?} ssao_intensity must be >= 0"
            );
        }
    }

    #[test]
    fn low_has_ssao_off() {
        let s = RenderQuality::Low.to_settings();
        assert!(!s.ssao_enabled);
    }

    #[test]
    fn ultra_has_everything_on() {
        let s = RenderQuality::Ultra.to_settings();
        assert!(s.ssao_enabled);
        assert!(s.fxaa_enabled);
        assert!(s.bloom_enabled);
        assert!(
            s.ssao_intensity > 1.0,
            "Ultra should have boosted SSAO intensity"
        );
        assert!(
            s.contrast > 1.0,
            "Ultra should have slightly elevated contrast"
        );
        assert!(
            s.saturation > 1.0,
            "Ultra should have slightly elevated saturation"
        );
    }

    #[test]
    fn from_name_case_insensitive() {
        assert_eq!(RenderQuality::from_name("low"), Some(RenderQuality::Low));
        assert_eq!(RenderQuality::from_name("HIGH"), Some(RenderQuality::High));
        assert_eq!(
            RenderQuality::from_name("Ultra"),
            Some(RenderQuality::Ultra)
        );
        assert_eq!(RenderQuality::from_name("med"), Some(RenderQuality::Medium));
        assert_eq!(RenderQuality::from_name("invalid"), None);
    }

    #[test]
    fn name_roundtrip() {
        for quality in [
            RenderQuality::Low,
            RenderQuality::Medium,
            RenderQuality::High,
            RenderQuality::Ultra,
        ] {
            assert_eq!(
                RenderQuality::from_name(quality.name()),
                Some(quality),
                "{quality:?} should roundtrip through name()"
            );
        }
    }
}
