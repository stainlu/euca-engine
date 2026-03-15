use crate::camera::Camera;
use crate::gpu::GpuContext;
use crate::light::{AmbientLight, DirectionalLight};
use crate::material::{Material, MaterialHandle};
use crate::mesh::{Mesh, MeshHandle};
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

/// Per-object transform uniforms.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectUniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4], // transpose(inverse(model)) — using mat4 for alignment
}

/// Per-frame scene uniforms (camera + lighting).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    camera_pos: [f32; 4],      // xyz + pad
    light_direction: [f32; 4], // xyz + pad
    light_color: [f32; 4],     // rgb + intensity
    ambient_color: [f32; 4],   // rgb + intensity
}

const MAX_DRAW_CALLS: usize = 1024;
const MAX_MATERIALS: usize = 256;

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    // Group 0: per-object transforms (dynamic)
    object_buffer: wgpu::Buffer,
    object_bind_group: wgpu::BindGroup,
    object_aligned_size: u64,
    // Group 1: per-frame scene data
    scene_buffer: wgpu::Buffer,
    scene_bind_group: wgpu::BindGroup,
    // Group 2: per-material data (dynamic)
    material_buffer: wgpu::Buffer,
    material_bind_group: wgpu::BindGroup,
    material_aligned_size: u64,

    meshes: Vec<GpuMesh>,
    material_count: u32,

    depth_texture: wgpu::TextureView,
    depth_format: wgpu::TextureFormat,
}

impl Renderer {
    pub fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("PBR Shader"),
                source: wgpu::ShaderSource::Wgsl(PBR_SHADER.into()),
            });

        let min_align = gpu.device.limits().min_uniform_buffer_offset_alignment as u64;
        let object_size = std::mem::size_of::<ObjectUniforms>() as u64;
        let object_aligned = object_size.div_ceil(min_align) * min_align;
        let material_size = std::mem::size_of::<MaterialUniforms>() as u64;
        let material_aligned = material_size.div_ceil(min_align) * min_align;

        // Object transform buffer (dynamic)
        let object_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Object UBO"),
            size: object_aligned * MAX_DRAW_CALLS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Scene buffer (per-frame)
        let scene_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene UBO"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Material buffer (dynamic)
        let material_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material UBO"),
            size: material_aligned * MAX_MATERIALS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layouts
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

        let scene_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Scene BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<SceneUniforms>() as u64,
                        ),
                    },
                    count: None,
                }],
            });

        let material_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Material BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(material_size),
                    },
                    count: None,
                }],
            });

        // Bind groups
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
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buffer.as_entire_binding(),
            }],
        });

        let material_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material BG"),
            layout: &material_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &material_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(material_size),
                }),
            }],
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
            scene_bind_group,
            material_buffer,
            material_bind_group,
            material_aligned_size: material_aligned,
            meshes: Vec::new(),
            material_count: 0,
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

    /// Upload a material's PBR properties to the GPU, returning a handle.
    pub fn upload_material(&mut self, gpu: &GpuContext, mat: &Material) -> MaterialHandle {
        let handle = MaterialHandle(self.material_count);
        let uniforms = MaterialUniforms {
            albedo: mat.albedo,
            metallic: mat.metallic,
            roughness: mat.roughness,
            _pad: [0.0; 2],
        };
        let offset = handle.0 as u64 * self.material_aligned_size;
        gpu.queue
            .write_buffer(&self.material_buffer, offset, bytemuck::bytes_of(&uniforms));
        self.material_count += 1;
        handle
    }

    pub fn resize(&mut self, gpu: &GpuContext) {
        self.depth_texture =
            Self::create_depth_texture(&gpu.device, &gpu.surface_config, self.depth_format);
    }

    /// Draw all commands with PBR lighting (convenience: gets surface, renders, presents).
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

    /// Render to a given texture view. Does NOT present — caller manages the surface.
    /// Use this when compositing with egui or other overlays.
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
        let vp = camera.view_projection_matrix(gpu.aspect_ratio());

        // Write scene uniforms
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
        };
        gpu.queue
            .write_buffer(&self.scene_buffer, 0, bytemuck::bytes_of(&scene));

        // Write object transforms
        let aligned = self.object_aligned_size as usize;
        let mut obj_data = vec![0u8; aligned * commands.len()];
        for (i, cmd) in commands.iter().enumerate() {
            let model = cmd.model_matrix;
            let mvp = vp * model;
            let normal_mat = model.0.inverse().transpose();
            let obj = ObjectUniforms {
                mvp: mvp.0.to_cols_array_2d(),
                model: model.0.to_cols_array_2d(),
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
                let mat_offset = (cmd.material.0 as u64 * self.material_aligned_size) as u32;

                pass.set_bind_group(0, &self.object_bind_group, &[obj_offset]);
                pass.set_bind_group(2, &self.material_bind_group, &[mat_offset]);

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

/// PBR shader implementing Cook-Torrance BRDF with a single directional light.
const PBR_SHADER: &str = r#"
// ── Bind Group 0: Per-object transforms ──
struct ObjectUniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> object: ObjectUniforms;

// ── Bind Group 1: Per-frame scene data ──
struct SceneUniforms {
    camera_pos: vec4<f32>,
    light_direction: vec4<f32>,
    light_color: vec4<f32>,      // rgb + intensity
    ambient_color: vec4<f32>,    // rgb + intensity
};
@group(1) @binding(0) var<uniform> scene: SceneUniforms;

// ── Bind Group 2: Per-material data ──
struct MaterialUniforms {
    albedo: vec4<f32>,
    metallic: f32,
    roughness: f32,
};
@group(2) @binding(0) var<uniform> material: MaterialUniforms;

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

// GGX/Trowbridge-Reitz Normal Distribution Function
fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX Geometry Function
fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith's method combining view and light geometry
fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

// Fresnel-Schlick approximation
fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ── Fragment ──
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = material.albedo.rgb;
    let metallic = material.metallic;
    let roughness = max(material.roughness, 0.04); // clamp to avoid division by zero

    let N = normalize(in.world_normal);
    let V = normalize(scene.camera_pos.xyz - in.world_pos);

    // F0: reflectance at normal incidence (0.04 for dielectrics, albedo for metals)
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Directional light
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

    // Energy conservation
    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

    // Outgoing radiance
    let Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    // Ambient
    let ambient_intensity = scene.ambient_color.w;
    let ambient = scene.ambient_color.rgb * ambient_intensity * albedo;

    let color = ambient + Lo;

    // Simple Reinhard tone mapping
    let mapped = color / (color + vec3<f32>(1.0));

    // Gamma correction (linear → sRGB)
    let gamma_corrected = pow(mapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, 1.0);
}
"#;
