use crate::camera::Camera;
use crate::gpu::GpuContext;
use crate::light::{AmbientLight, DirectionalLight};
use crate::material::{Material, MaterialHandle};
use crate::mesh::{Mesh, MeshHandle};
use crate::texture::{TextureHandle, TextureStore};
use crate::vertex::Vertex;
use euca_math::Mat4;

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

/// GPU-side material uniforms.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniforms {
    albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
    has_normal_map: f32, // 1.0 = yes, 0.0 = no (using f32 for alignment)
    _pad: f32,
}

/// A draw command: mesh + material + model transform.
pub struct DrawCommand {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub model_matrix: Mat4,
}

/// Per-instance data in storage buffer.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
}

const MAX_POINT_LIGHTS: usize = 4;
const MAX_SPOT_LIGHTS: usize = 2;

/// GPU-side point light data.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct GpuPointLight {
    position: [f32; 4], // xyz = position, w = range
    color: [f32; 4],    // rgb = color, a = intensity
}

/// GPU-side spot light data.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct GpuSpotLight {
    position: [f32; 4],  // xyz = position, w = range
    direction: [f32; 4], // xyz = direction, w = unused
    color: [f32; 4],     // rgb = color, a = intensity
    cone: [f32; 4],      // x = inner_cos, y = outer_cos, zw = unused
}

/// Per-frame scene uniforms.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    camera_pos: [f32; 4],
    light_direction: [f32; 4],
    light_color: [f32; 4],
    ambient_color: [f32; 4],
    camera_vp: [[f32; 4]; 4],
    light_vp: [[f32; 4]; 4], // Cascade 0 VP (kept for backward compat)
    inv_vp: [[f32; 4]; 4],
    cascade_vps: [[[f32; 4]; 4]; 3], // VP matrices for each cascade
    cascade_splits: [f32; 4],        // x,y,z = split distances, w = unused
    // Point + spot light arrays
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
    spot_lights: [GpuSpotLight; MAX_SPOT_LIGHTS],
    num_point_lights: [f32; 4], // x = count (using vec4 for alignment)
    num_spot_lights: [f32; 4],  // x = count
}

/// Per-material GPU resources.
struct GpuMaterial {
    bind_group: wgpu::BindGroup,
    _buffer: wgpu::Buffer,
}

/// A batch of instances sharing the same mesh + material.
struct DrawBatch {
    mesh: MeshHandle,
    material: MaterialHandle,
    instance_start: u32,
    instance_count: u32,
}

const MAX_INSTANCES: usize = 16384;
const SHADOW_MAP_SIZE: u32 = 2048;
const NUM_SHADOW_CASCADES: u32 = 3;
/// Ortho sizes for each cascade (near, mid, far).
const CASCADE_ORTHO_SIZES: [f32; 3] = [8.0, 20.0, 50.0];

#[allow(dead_code)]
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    // Group 0: per-instance storage buffer
    instance_buffer: wgpu::Buffer,
    instance_bind_group: wgpu::BindGroup,
    instance_bgl: wgpu::BindGroupLayout,
    // Group 1: scene + shadow
    scene_buffer: wgpu::Buffer,
    scene_bgl: wgpu::BindGroupLayout,
    scene_bind_group: wgpu::BindGroup,
    // Group 2: per-material
    material_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    materials: Vec<GpuMaterial>,
    textures: TextureStore,

    sky_pipeline: wgpu::RenderPipeline,

    shadow_pipeline: wgpu::RenderPipeline,
    shadow_map: wgpu::Texture,
    shadow_map_view: wgpu::TextureView, // Array view for sampling all cascades
    shadow_cascade_views: Vec<wgpu::TextureView>, // Per-layer views for rendering
    shadow_sampler: wgpu::Sampler,
    shadow_instance_buffer: wgpu::Buffer,
    shadow_instance_bind_group: wgpu::BindGroup,

    // Post-processing
    postprocess_pipeline: wgpu::RenderPipeline,
    postprocess_bgl: wgpu::BindGroupLayout,
    postprocess_sampler: wgpu::Sampler,
    hdr_texture: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    postprocess_bind_group: wgpu::BindGroup,

    meshes: Vec<GpuMesh>,
    depth_texture: wgpu::TextureView,
    depth_format: wgpu::TextureFormat,
    surface_format: wgpu::TextureFormat,
}

impl Renderer {
    pub fn new(gpu: &GpuContext) -> Self {
        let instance_buf_size = (MAX_INSTANCES * std::mem::size_of::<InstanceData>()) as u64;

        // ── Buffers ──
        let instance_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance SSBO"),
            size: instance_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene UBO"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_instance_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Instance SSBO"),
            size: instance_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        // ── Cascaded shadow map (2D array texture with NUM_SHADOW_CASCADES layers) ──
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
        // Array view for sampling all cascades in the PBR shader
        let shadow_map_view = shadow_map.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        // Per-cascade views for rendering
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

        // ── Bind group layouts ──
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Normal map texture (binding 3)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // ── Bind groups ──
        let instance_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Instance BG"),
            layout: &instance_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buffer.as_entire_binding(),
            }],
        });

        let shadow_instance_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Instance BG"),
            layout: &instance_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_instance_buffer.as_entire_binding(),
            }],
        });

        let scene_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene BG"),
            layout: &scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_buffer.as_entire_binding(),
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

        // ── Shadow pipeline ──
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
                    bind_group_layouts: &[&instance_bgl], // only instances, no scene (avoids shadow map conflict)
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
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

        // ── Sky pipeline ──
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

        // ── Main PBR pipeline (renders to HDR texture) ──
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
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

        // Sky pipeline also renders to HDR
        // (already created above — need to recreate with hdr_format)
        // We'll handle this by creating a separate sky pipeline for HDR
        // Actually sky_pipeline was created with surface format; let's recreate it
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
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

        // ── HDR offscreen texture ──
        let (hdr_texture, hdr_view) = Self::create_hdr_texture(&gpu.device, &gpu.surface_config);

        // ── Post-processing pipeline (HDR → surface) ──
        let postprocess_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Postprocess Shader"),
                source: wgpu::ShaderSource::Wgsl(POSTPROCESS_SHADER.into()),
            });

        let postprocess_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Postprocess Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let postprocess_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Postprocess BGL"),
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
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let postprocess_bind_group = Self::create_postprocess_bind_group(
            &gpu.device,
            &postprocess_bgl,
            &hdr_view,
            &postprocess_sampler,
        );

        let postprocess_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Postprocess Pipeline Layout"),
                    bind_group_layouts: &[&postprocess_bgl],
                    push_constant_ranges: &[],
                });

        let postprocess_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Postprocess Pipeline"),
                    layout: Some(&postprocess_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &postprocess_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &postprocess_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.surface_config.format,
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
            pipeline,
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
            postprocess_pipeline,
            postprocess_bgl,
            postprocess_sampler,
            hdr_texture,
            hdr_view,
            postprocess_bind_group,
            meshes: Vec::new(),
            depth_texture,
            depth_format,
            surface_format: gpu.surface_config.format,
        }
    }

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

    /// Upload a texture from raw RGBA8 pixel data.
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

    /// Upload a texture from an encoded image (PNG, JPEG, etc.).
    pub fn upload_texture_image(&mut self, gpu: &GpuContext, data: &[u8]) -> TextureHandle {
        self.textures.upload_image(&gpu.device, &gpu.queue, data)
    }

    /// Generate a checkerboard test texture.
    pub fn checkerboard_texture(
        &mut self,
        gpu: &GpuContext,
        size: u32,
        tile: u32,
    ) -> TextureHandle {
        self.textures
            .checkerboard(&gpu.device, &gpu.queue, size, tile)
    }

    /// Upload a material's PBR properties to the GPU.
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
            _pad: 0.0,
        };
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Material UBO"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let albedo_handle = mat
            .albedo_texture
            .unwrap_or_else(TextureStore::default_white);
        let albedo_view = self.textures.view(albedo_handle);

        // Normal map: use a default flat normal (0.5, 0.5, 1.0) if none provided
        let normal_handle = mat
            .normal_texture
            .unwrap_or_else(TextureStore::default_white);
        let normal_view = self.textures.view(normal_handle);

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
            ],
        });
        self.materials.push(GpuMaterial {
            bind_group,
            _buffer: buffer,
        });
        handle
    }

    pub fn resize(&mut self, gpu: &GpuContext) {
        self.depth_texture =
            Self::create_depth_texture(&gpu.device, &gpu.surface_config, self.depth_format);
        let (hdr_texture, hdr_view) = Self::create_hdr_texture(&gpu.device, &gpu.surface_config);
        self.hdr_texture = hdr_texture;
        self.hdr_view = hdr_view;
        self.postprocess_bind_group = Self::create_postprocess_bind_group(
            &gpu.device,
            &self.postprocess_bgl,
            &self.hdr_view,
            &self.postprocess_sampler,
        );
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

    /// Sort commands by (mesh, material) and build batches for instanced drawing.
    fn build_batches(commands: &[DrawCommand]) -> (Vec<InstanceData>, Vec<DrawBatch>) {
        if commands.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Sort indices by (mesh, material)
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
        // Final batch
        batches.push(DrawBatch {
            mesh: batch_mesh,
            material: batch_mat,
            instance_start: batch_start,
            instance_count: instances.len() as u32 - batch_start,
        });

        (instances, batches)
    }

    pub fn draw(
        &self,
        gpu: &GpuContext,
        camera: &Camera,
        light: &DirectionalLight,
        ambient: &AmbientLight,
        commands: &[DrawCommand],
    ) {
        self.draw_with_lights(gpu, camera, light, ambient, commands, &[], &[]);
    }

    /// Draw with full lighting: directional + point + spot lights.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_with_lights(
        &self,
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

    #[allow(clippy::too_many_arguments)]
    pub fn render_to_view(
        &self,
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

    #[allow(clippy::too_many_arguments)]
    pub fn render_to_view_with_lights(
        &self,
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
        let (instances, batches) = Self::build_batches(commands);

        // Upload instance data
        if !instances.is_empty() {
            gpu.queue
                .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
        }

        // ── Cascaded shadow passes (one per cascade layer) ──
        for (cascade_idx, &cascade_ortho) in CASCADE_ORTHO_SIZES.iter().enumerate() {
            let cascade_vp = Self::light_vp_for_cascade(light, cascade_ortho);

            let shadow_instances: Vec<InstanceData> = instances
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
                gpu.queue.write_buffer(
                    &self.shadow_instance_buffer,
                    0,
                    bytemuck::cast_slice(&shadow_instances),
                );
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

            for batch in &batches {
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

        // ── Write scene uniforms ──
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
        };
        gpu.queue
            .write_buffer(&self.scene_buffer, 0, bytemuck::bytes_of(&scene));

        // ── Main PBR pass (renders to HDR offscreen texture) ──
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PBR Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
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

            // Sky
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.scene_bind_group, &[]);
            pass.draw(0..3, 0..1);

            // PBR geometry (instanced batches)
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.instance_bind_group, &[]);
            pass.set_bind_group(1, &self.scene_bind_group, &[]);

            for batch in &batches {
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

        // ── Post-processing pass (HDR → output) ──
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Postprocess Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
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

            pass.set_pipeline(&self.postprocess_pipeline);
            pass.set_bind_group(0, &self.postprocess_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_hdr_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR Texture"),
            size: wgpu::Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_postprocess_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        hdr_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Postprocess BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }
}

// ════════════════════════════════════════════════════════════════════════
// WGSL Shaders
// ════════════════════════════════════════════════════════════════════════

/// Shadow pass: depth-only. Instance model already contains light_vp * model (baked on CPU).
const SHADOW_SHADER: &str = r#"
struct InstanceData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
};

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) iid: u32) -> @builtin(position) vec4<f32> {
    return instances[iid].model * vec4<f32>(in.position, 1.0);
}
"#;

/// PBR shader with instancing, textures, shadows.
const PBR_SHADER: &str = r#"
struct InstanceData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
};

struct PointLightData {
    position: vec4<f32>,  // xyz = position, w = range
    color: vec4<f32>,     // rgb = color, a = intensity
};

struct SpotLightData {
    position: vec4<f32>,   // xyz = position, w = range
    direction: vec4<f32>,  // xyz = direction
    color: vec4<f32>,      // rgb = color, a = intensity
    cone: vec4<f32>,       // x = inner_cos, y = outer_cos
};

struct SceneUniforms {
    camera_pos: vec4<f32>,
    light_direction: vec4<f32>,
    light_color: vec4<f32>,
    ambient_color: vec4<f32>,
    camera_vp: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    inv_vp: mat4x4<f32>,
    cascade_vps: array<mat4x4<f32>, 3>,
    cascade_splits: vec4<f32>,  // xyz = ortho sizes per cascade
    point_lights: array<PointLightData, 4>,
    spot_lights: array<SpotLightData, 2>,
    num_point_lights: vec4<f32>,
    num_spot_lights: vec4<f32>,
};

struct MaterialUniforms {
    albedo: vec4<f32>,
    metallic: f32,
    roughness: f32,
    has_normal_map: f32,
};

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var shadow_map: texture_depth_2d_array;
@group(1) @binding(2) var shadow_sampler: sampler_comparison;
@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var albedo_tex: texture_2d<f32>;
@group(2) @binding(2) var albedo_sampler: sampler;
@group(2) @binding(3) var normal_tex: texture_2d<f32>;
// Normal map uses the same sampler as albedo (binding 2)

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) iid: u32) -> VertexOutput {
    let model = instances[iid].model;
    let normal_mat = instances[iid].normal_matrix;
    var out: VertexOutput;
    let world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    out.clip_position = scene.camera_vp * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = normalize((normal_mat * vec4<f32>(in.normal, 0.0)).xyz);
    out.world_tangent = normalize((model * vec4<f32>(in.tangent, 0.0)).xyz);
    out.uv = in.uv;
    return out;
}

const PI: f32 = 3.14159265359;

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn shadow_factor(world_pos: vec3<f32>) -> f32 {
    // Select the tightest cascade that contains this fragment
    var cascade_idx = 0i;
    for (var ci = 0i; ci < 3; ci++) {
        let vp = scene.cascade_vps[ci];
        let clip = vp * vec4<f32>(world_pos, 1.0);
        let ndc = clip.xyz / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
        if uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 {
            cascade_idx = ci;
            break;
        }
    }

    let vp = scene.cascade_vps[cascade_idx];
    let light_clip = vp * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let current_depth = ndc.z;
    if shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 {
        return 1.0;
    }
    let texel_size = 1.0 / 2048.0;
    var shadow = 0.0;
    for (var x = -1i; x <= 1i; x++) {
        for (var y = -1i; y <= 1i; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow += textureSampleCompare(
                shadow_map, shadow_sampler,
                shadow_uv + offset, cascade_idx,
                current_depth
            );
        }
    }
    return shadow / 9.0;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(albedo_tex, albedo_sampler, in.uv);
    let albedo = material.albedo.rgb * tex_color.rgb;
    let metallic = material.metallic;
    let roughness = max(material.roughness, 0.04);

    // Normal: use normal map if available, otherwise vertex normal
    var N: vec3<f32>;
    if material.has_normal_map > 0.5 {
        // Sample normal map (tangent-space, [0,1] → [-1,1])
        let sampled = textureSample(normal_tex, albedo_sampler, in.uv).rgb;
        let tangent_normal = sampled * 2.0 - 1.0;
        // Construct TBN matrix
        let T = normalize(in.world_tangent);
        let N_vert = normalize(in.world_normal);
        let B = cross(N_vert, T);
        N = normalize(T * tangent_normal.x + B * tangent_normal.y + N_vert * tangent_normal.z);
    } else {
        N = normalize(in.world_normal);
    }
    let V = normalize(scene.camera_pos.xyz - in.world_pos);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    let L = normalize(-scene.light_direction.xyz);
    let H = normalize(V + L);
    let NdotL = max(dot(N, L), 0.0);

    let light_intensity = scene.light_color.w;
    let radiance = scene.light_color.rgb * light_intensity;

    let D = distribution_ggx(N, H, roughness);
    let G = geometry_smith(N, V, L, roughness);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);

    let numerator = D * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    let specular = numerator / denominator;

    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

    let shadow = shadow_factor(in.world_pos);
    var Lo = (kD * albedo / PI + specular) * radiance * NdotL * shadow;

    // ── Point lights ──
    let n_point = i32(scene.num_point_lights.x);
    for (var pi = 0; pi < n_point; pi++) {
        let pl = scene.point_lights[pi];
        let pl_pos = pl.position.xyz;
        let pl_range = pl.position.w;
        let pl_color = pl.color.rgb;
        let pl_intensity = pl.color.a;

        let pl_dir = pl_pos - in.world_pos;
        let pl_dist = length(pl_dir);
        if pl_dist > pl_range { continue; }
        let pl_L = pl_dir / pl_dist;
        let pl_NdotL = max(dot(N, pl_L), 0.0);
        let pl_attenuation = 1.0 / (pl_dist * pl_dist + 0.01);
        let pl_falloff = saturate(1.0 - pl_dist / pl_range);
        let pl_radiance = pl_color * pl_intensity * pl_attenuation * pl_falloff;
        Lo += (kD * albedo / PI) * pl_radiance * pl_NdotL;
    }

    // ── Spot lights ──
    let n_spot = i32(scene.num_spot_lights.x);
    for (var si = 0; si < n_spot; si++) {
        let sl = scene.spot_lights[si];
        let sl_pos = sl.position.xyz;
        let sl_range = sl.position.w;
        let sl_dir_norm = normalize(sl.direction.xyz);
        let sl_color = sl.color.rgb;
        let sl_intensity = sl.color.a;
        let sl_inner_cos = sl.cone.x;
        let sl_outer_cos = sl.cone.y;

        let sl_to_frag = in.world_pos - sl_pos;
        let sl_dist = length(sl_to_frag);
        if sl_dist > sl_range { continue; }
        let sl_L = -normalize(sl_to_frag);
        let sl_NdotL = max(dot(N, sl_L), 0.0);
        let sl_cos_theta = dot(normalize(sl_to_frag), sl_dir_norm);
        let sl_cone_atten = saturate((sl_cos_theta - sl_outer_cos) / (sl_inner_cos - sl_outer_cos));
        let sl_dist_atten = 1.0 / (sl_dist * sl_dist + 0.01);
        let sl_falloff = saturate(1.0 - sl_dist / sl_range);
        let sl_radiance = sl_color * sl_intensity * sl_dist_atten * sl_falloff * sl_cone_atten;
        Lo += (kD * albedo / PI) * sl_radiance * sl_NdotL;
    }

    let ambient_intensity = scene.ambient_color.w;
    let ambient = scene.ambient_color.rgb * ambient_intensity * albedo;

    let color = ambient + Lo;
    return vec4<f32>(color, 1.0); // linear HDR output — tone mapping done in post-processing
}
"#;

/// Procedural sky shader.
const SKY_SHADER: &str = r#"
struct SceneUniforms {
    camera_pos: vec4<f32>,
    light_direction: vec4<f32>,
    light_color: vec4<f32>,
    ambient_color: vec4<f32>,
    camera_vp: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    inv_vp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> scene: SceneUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    let x = f32(i32(id) / 2) * 4.0 - 1.0;
    let y = f32(i32(id) % 2) * 4.0 - 1.0;
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 1.0, 1.0);
    out.ndc = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let clip = vec4<f32>(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let world_h = scene.inv_vp * clip;
    let world_dir = normalize(world_h.xyz / world_h.w - scene.camera_pos.xyz);

    let up = max(world_dir.y, 0.0);
    let down = max(-world_dir.y, 0.0);

    let sky_zenith = vec3<f32>(0.15, 0.3, 0.65);
    let sky_horizon = vec3<f32>(0.55, 0.7, 0.9);
    let ground_color = vec3<f32>(0.15, 0.13, 0.12);

    var color: vec3<f32>;
    if world_dir.y >= 0.0 {
        let t = pow(up, 0.5);
        color = mix(sky_horizon, sky_zenith, t);
    } else {
        let t = pow(down, 0.8);
        color = mix(sky_horizon, ground_color, t);
    }

    let sun_dir = normalize(-scene.light_direction.xyz);
    let sun_dot = max(dot(world_dir, sun_dir), 0.0);

    let sun_disk = smoothstep(0.9995, 0.9999, sun_dot);
    let sun_color = vec3<f32>(1.0, 0.95, 0.85);
    color = mix(color, sun_color * 3.0, sun_disk);

    let glow = pow(sun_dot, 64.0) * 0.6;
    color += sun_color * glow;

    let horizon_glow = pow(sun_dot, 8.0) * (1.0 - up) * 0.3;
    color += vec3<f32>(1.0, 0.6, 0.3) * horizon_glow;

    return vec4<f32>(color, 1.0); // linear HDR — tone mapping in post-processing
}
"#;

/// Post-processing shader: bloom approximation + ACES tone mapping + vignette.
const POSTPROCESS_SHADER: &str = r#"
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    let x = f32(i32(id) / 2) * 4.0 - 1.0;
    let y = f32(i32(id) % 2) * 4.0 - 1.0;
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}

// ACES filmic tone mapping (fitted curve by Krzysztof Narkowicz)
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Simple bloom: 13-tap filter sampling at wide offsets for glow
fn bloom_sample(uv: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    var bloom = vec3<f32>(0.0);
    // Center
    let center = textureSample(hdr_tex, hdr_sampler, uv).rgb;

    // 12 samples in a circle at wide radius for glow approximation
    let offsets = array<vec2<f32>, 12>(
        vec2<f32>(-1.0,  0.0), vec2<f32>( 1.0,  0.0),
        vec2<f32>( 0.0, -1.0), vec2<f32>( 0.0,  1.0),
        vec2<f32>(-0.7, -0.7), vec2<f32>( 0.7, -0.7),
        vec2<f32>(-0.7,  0.7), vec2<f32>( 0.7,  0.7),
        vec2<f32>(-2.0,  0.0), vec2<f32>( 2.0,  0.0),
        vec2<f32>( 0.0, -2.0), vec2<f32>( 0.0,  2.0),
    );

    let radius = 4.0; // pixel radius for bloom
    for (var i = 0u; i < 12u; i++) {
        let sample_uv = uv + offsets[i] * texel * radius;
        let s = textureSample(hdr_tex, hdr_sampler, sample_uv).rgb;
        // Only bloom bright pixels (threshold at ~1.0 luminance)
        let luminance = dot(s, vec3<f32>(0.2126, 0.7152, 0.0722));
        let bright = max(luminance - 0.8, 0.0) / max(luminance, 0.001);
        bloom += s * bright;
    }

    return center + bloom * 0.08; // subtle bloom intensity
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(hdr_tex));
    let texel = 1.0 / dims;

    // Bloom
    let hdr = bloom_sample(in.uv, texel);

    // ACES tone mapping
    let mapped = aces_tonemap(hdr);

    // Gamma correction (linear → sRGB)
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));

    // Vignette (subtle darkening at edges)
    let center_dist = length(in.uv - 0.5) * 1.4;
    let vignette = 1.0 - center_dist * center_dist * 0.35;
    let final_color = gamma * vignette;

    return vec4<f32>(final_color, 1.0);
}
"#;
