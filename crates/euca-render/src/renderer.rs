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
    _pad: [f32; 2],
}

/// A draw command: mesh + material + model transform.
pub struct DrawCommand {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub model_matrix: Mat4,
}

/// Per-object transform uniforms (used in both shadow and main pass).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectUniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
}

/// Per-frame scene uniforms (camera + lighting + shadow).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    camera_pos: [f32; 4],
    light_direction: [f32; 4],
    light_color: [f32; 4],
    ambient_color: [f32; 4],
    light_vp: [[f32; 4]; 4], // light view-projection for shadow mapping
}

/// Per-material GPU resources.
struct GpuMaterial {
    bind_group: wgpu::BindGroup,
    _buffer: wgpu::Buffer,
}

const MAX_DRAW_CALLS: usize = 1024;
const SHADOW_MAP_SIZE: u32 = 2048;
const SHADOW_ORTHO_SIZE: f32 = 20.0; // world units covered by shadow map

#[allow(dead_code)]
pub struct Renderer {
    // Main PBR pipeline
    pipeline: wgpu::RenderPipeline,
    // Group 0: per-object transforms (dynamic)
    object_buffer: wgpu::Buffer,
    object_bind_group: wgpu::BindGroup,
    object_aligned_size: u64,
    // Group 1: per-frame scene data + shadow map
    scene_buffer: wgpu::Buffer,
    scene_bgl: wgpu::BindGroupLayout,
    scene_bind_group: wgpu::BindGroup,
    // Group 2: per-material (individual bind groups)
    material_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    materials: Vec<GpuMaterial>,
    textures: TextureStore,

    // Shadow mapping
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_map: wgpu::Texture,
    shadow_map_view: wgpu::TextureView,
    shadow_sampler: wgpu::Sampler,
    shadow_object_buffer: wgpu::Buffer,
    shadow_object_bind_group: wgpu::BindGroup,
    shadow_object_aligned_size: u64,

    meshes: Vec<GpuMesh>,
    depth_texture: wgpu::TextureView,
    depth_format: wgpu::TextureFormat,
}

impl Renderer {
    pub fn new(gpu: &GpuContext) -> Self {
        let min_align = gpu.device.limits().min_uniform_buffer_offset_alignment as u64;
        let object_size = std::mem::size_of::<ObjectUniforms>() as u64;
        let object_aligned = object_size.div_ceil(min_align) * min_align;

        // ── Shared resources ──
        let object_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Object UBO"),
            size: object_aligned * MAX_DRAW_CALLS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene UBO"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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

        // ── Shadow map ──
        let shadow_map = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_map_view = shadow_map.create_view(&wgpu::TextureViewDescriptor::default());

        let shadow_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ── Bind group layouts ──
        let object_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Object BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(object_size),
                    },
                    count: None,
                }],
            });

        // Scene BGL: uniforms + shadow map + shadow sampler
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
                            view_dimension: wgpu::TextureViewDimension::D2,
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
                ],
            });

        // ── Bind groups ──
        let object_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Object BG"),
            layout: &object_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &object_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(object_size),
                }),
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

        // ── Shadow pipeline (depth-only, uses only group 0) ──
        let shadow_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shadow Shader"),
                source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
            });

        // Shadow pass uses its own object buffer (light-space MVPs)
        let shadow_object_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Object UBO"),
            size: object_aligned * MAX_DRAW_CALLS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_object_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Object BG"),
            layout: &object_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &shadow_object_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(object_size),
                }),
            }],
        });

        let shadow_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Shadow Pipeline Layout"),
                    bind_group_layouts: &[&object_bgl],
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
                fragment: None, // depth-only
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Front), // front-face culling reduces shadow acne
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
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

        // ── Main PBR pipeline ──
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
                bind_group_layouts: &[&object_bgl, &scene_bgl, &material_bgl],
                push_constant_ranges: &[],
            });

        let depth_format = wgpu::TextureFormat::Depth32Float;
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
                        format: gpu.surface_config.format,
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

        Self {
            pipeline,
            object_buffer,
            object_bind_group,
            object_aligned_size: object_aligned,
            scene_buffer,
            scene_bgl,
            scene_bind_group,
            material_bgl,
            sampler,
            materials: Vec::new(),
            textures,
            shadow_pipeline,
            shadow_map,
            shadow_map_view,
            shadow_sampler,
            shadow_object_buffer,
            shadow_object_bind_group,
            shadow_object_aligned_size: object_aligned,
            meshes: Vec::new(),
            depth_texture,
            depth_format,
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

    /// Upload a material's PBR properties to the GPU, returning a handle.
    pub fn upload_material(&mut self, gpu: &GpuContext, mat: &Material) -> MaterialHandle {
        use wgpu::util::DeviceExt;

        let handle = MaterialHandle(self.materials.len() as u32);

        let uniforms = MaterialUniforms {
            albedo: mat.albedo,
            metallic: mat.metallic,
            roughness: mat.roughness,
            _pad: [0.0; 2],
        };

        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Material UBO"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let tex_handle = mat
            .albedo_texture
            .unwrap_or_else(TextureStore::default_white);
        let tex_view = self.textures.view(tex_handle);

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
                    resource: wgpu::BindingResource::TextureView(tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
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
    }

    /// Compute light view-projection matrix for directional shadow mapping.
    fn light_vp(light: &DirectionalLight) -> Mat4 {
        use euca_math::Vec3;
        let dir = Vec3::new(light.direction[0], light.direction[1], light.direction[2]).normalize();
        // Place light camera far back along the light direction
        let light_pos = dir * -30.0;
        let light_view = Mat4::look_at_lh(light_pos, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        let s = SHADOW_ORTHO_SIZE;
        let light_proj = Mat4::orthographic_lh(-s, s, -s, s, 0.1, 60.0);
        light_proj * light_view
    }

    /// Draw all commands with PBR lighting + shadows.
    pub fn draw(
        &self,
        gpu: &GpuContext,
        camera: &Camera,
        light: &DirectionalLight,
        ambient: &AmbientLight,
        commands: &[DrawCommand],
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

        self.render_to_view(gpu, camera, light, ambient, commands, &view, &mut encoder);

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    /// Render to a given texture view (shadow pass + main pass).
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
        let light_vp = Self::light_vp(light);

        // ── Shadow pass: render depth from light's perspective ──
        {
            let aligned = self.shadow_object_aligned_size as usize;
            let mut shadow_data = vec![0u8; aligned * commands.len()];
            for (i, cmd) in commands.iter().enumerate() {
                let model = cmd.model_matrix;
                let mvp = light_vp * model;
                let obj = ObjectUniforms {
                    mvp: mvp.to_cols_array_2d(),
                    model: model.to_cols_array_2d(),
                    normal_matrix: [[0.0; 4]; 4], // unused in shadow pass
                };
                let offset = i * aligned;
                shadow_data[offset..offset + std::mem::size_of::<ObjectUniforms>()]
                    .copy_from_slice(bytemuck::bytes_of(&obj));
            }
            if !commands.is_empty() {
                gpu.queue
                    .write_buffer(&self.shadow_object_buffer, 0, &shadow_data);
            }

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_map_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.shadow_pipeline);

            for (i, cmd) in commands.iter().enumerate() {
                let obj_offset = (i as u64 * self.shadow_object_aligned_size) as u32;
                pass.set_bind_group(0, &self.shadow_object_bind_group, &[obj_offset]);

                let mesh = &self.meshes[cmd.mesh.0 as usize];
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        // ── Main PBR pass ──
        let vp = camera.view_projection_matrix(gpu.aspect_ratio());

        // Write scene uniforms (with light VP for shadow sampling)
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
            light_vp: light_vp.to_cols_array_2d(),
        };
        gpu.queue
            .write_buffer(&self.scene_buffer, 0, bytemuck::bytes_of(&scene));

        // Recreate scene bind group each frame (shadow map view doesn't change, but
        // this ensures the shadow map rendered above is available for sampling)
        // Note: since shadow_map_view is persistent, the bind group created at init works fine.

        // Write object transforms for main pass
        let aligned = self.object_aligned_size as usize;
        let mut obj_data = vec![0u8; aligned * commands.len()];
        for (i, cmd) in commands.iter().enumerate() {
            let model = cmd.model_matrix;
            let mvp = vp * model;
            let normal_mat = model.inverse().transpose();
            let obj = ObjectUniforms {
                mvp: mvp.to_cols_array_2d(),
                model: model.to_cols_array_2d(),
                normal_matrix: normal_mat.to_cols_array_2d(),
            };
            let offset = i * aligned;
            obj_data[offset..offset + std::mem::size_of::<ObjectUniforms>()]
                .copy_from_slice(bytemuck::bytes_of(&obj));
        }
        if !commands.is_empty() {
            gpu.queue.write_buffer(&self.object_buffer, 0, &obj_data);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PBR Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
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

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(1, &self.scene_bind_group, &[]);

            for (i, cmd) in commands.iter().enumerate() {
                let obj_offset = (i as u64 * self.object_aligned_size) as u32;
                pass.set_bind_group(0, &self.object_bind_group, &[obj_offset]);

                let gpu_mat = &self.materials[cmd.material.0 as usize];
                pass.set_bind_group(2, &gpu_mat.bind_group, &[]);

                let mesh = &self.meshes[cmd.mesh.0 as usize];
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
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
}

/// Shadow pass vertex-only shader (depth output from light perspective).
const SHADOW_SHADER: &str = r#"
struct ObjectUniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> object: ObjectUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return object.mvp * vec4<f32>(in.position, 1.0);
}
"#;

/// PBR shader with Cook-Torrance BRDF, texture support, and shadow mapping.
const PBR_SHADER: &str = r#"
// ── Bind Group 0: Per-object transforms ──
struct ObjectUniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> object: ObjectUniforms;

// ── Bind Group 1: Per-frame scene data + shadow map ──
struct SceneUniforms {
    camera_pos: vec4<f32>,
    light_direction: vec4<f32>,
    light_color: vec4<f32>,
    ambient_color: vec4<f32>,
    light_vp: mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var shadow_map: texture_depth_2d;
@group(1) @binding(2) var shadow_sampler: sampler_comparison;

// ── Bind Group 2: Per-material data ──
struct MaterialUniforms {
    albedo: vec4<f32>,
    metallic: f32,
    roughness: f32,
};
@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var albedo_tex: texture_2d<f32>;
@group(2) @binding(2) var albedo_sampler: sampler;

// ── Vertex ──
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = (object.model * vec4<f32>(in.position, 1.0)).xyz;
    out.clip_position = object.mvp * vec4<f32>(in.position, 1.0);
    out.world_pos = world_pos;
    out.world_normal = normalize((object.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.uv;
    return out;
}

// ── PBR math ──
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

// ── Shadow sampling with 3×3 PCF ──
fn shadow_factor(world_pos: vec3<f32>) -> f32 {
    let light_clip = scene.light_vp * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;

    // NDC to shadow UV: x [-1,1] → [0,1], y [-1,1] → [1,0] (flip Y)
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let current_depth = ndc.z;

    // Outside shadow map → fully lit
    if shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 {
        return 1.0;
    }

    // 3×3 PCF (Percentage Closer Filtering) for soft shadows
    let texel_size = 1.0 / 2048.0; // SHADOW_MAP_SIZE
    var shadow = 0.0;
    for (var x = -1i; x <= 1i; x++) {
        for (var y = -1i; y <= 1i; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow += textureSampleCompare(
                shadow_map, shadow_sampler,
                shadow_uv + offset, current_depth
            );
        }
    }
    return shadow / 9.0;
}

// ── Fragment ──
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(albedo_tex, albedo_sampler, in.uv);
    let albedo = material.albedo.rgb * tex_color.rgb;
    let metallic = material.metallic;
    let roughness = max(material.roughness, 0.04);

    let N = normalize(in.world_normal);
    let V = normalize(scene.camera_pos.xyz - in.world_pos);

    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    let L = normalize(-scene.light_direction.xyz);
    let H = normalize(V + L);
    let NdotL = max(dot(N, L), 0.0);

    let light_intensity = scene.light_color.w;
    let radiance = scene.light_color.rgb * light_intensity;

    // Cook-Torrance BRDF
    let D = distribution_ggx(N, H, roughness);
    let G = geometry_smith(N, V, L, roughness);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);

    let numerator = D * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    let specular = numerator / denominator;

    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

    // Shadow
    let shadow = shadow_factor(in.world_pos);

    // Outgoing radiance (attenuated by shadow)
    let Lo = (kD * albedo / PI + specular) * radiance * NdotL * shadow;

    // Ambient (not affected by shadow)
    let ambient_intensity = scene.ambient_color.w;
    let ambient = scene.ambient_color.rgb * ambient_intensity * albedo;

    let color = ambient + Lo;

    // Reinhard tone mapping
    let mapped = color / (color + vec3<f32>(1.0));

    // Gamma correction
    let gamma_corrected = pow(mapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, 1.0);
}
"#;
