//! GPU compute particle system — emit and update millions of particles on the GPU.
//!
//! Uses two compute passes (emit + update) and an instanced draw for billboard rendering.

use euca_rhi::RenderDevice;
use euca_rhi::pass::{ComputePassOps, RenderPassOps};
use euca_rhi::wgpu_backend::WgpuDevice;

/// GPU-side particle struct layout (must match WGSL `Particle`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParticle {
    position: [f32; 3],
    age: f32,
    velocity: [f32; 3],
    lifetime: f32,
    size: f32,
    color: [f32; 4],
    _pad: [f32; 3],
}

/// Emit parameters uniform (must match WGSL `EmitParams`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EmitParamsGpu {
    emitter_position: [f32; 3],
    emit_count: u32,
    speed_min: f32,
    speed_max: f32,
    size_min: f32,
    size_max: f32,
    lifetime_min: f32,
    lifetime_max: f32,
    _pad0: [f32; 2],
    gravity: [f32; 3],
    dt: f32,
    time: f32,
    max_particles: u32,
    _pad1: [f32; 2],
    color_start: [f32; 4],
    color_end: [f32; 4],
    emit_direction: [f32; 3],
    cone_half_angle: f32,
}

/// Camera uniforms for the particle render pass (must match WGSL `CameraUniforms`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleCameraGpu {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    camera_right: [f32; 3],
    _pad0: f32,
    camera_up: [f32; 3],
    _pad1: f32,
}

/// Configuration for a GPU particle emitter.
#[derive(Clone, Debug)]
pub struct GpuParticleConfig {
    pub max_particles: u32,
    pub emit_rate: f32,
    pub speed_range: [f32; 2],
    pub size_range: [f32; 2],
    pub lifetime_range: [f32; 2],
    pub gravity: [f32; 3],
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
    pub emit_direction: [f32; 3],
    pub cone_half_angle: f32,
}

impl Default for GpuParticleConfig {
    fn default() -> Self {
        Self {
            max_particles: 100_000,
            emit_rate: 1000.0,
            speed_range: [1.0, 5.0],
            size_range: [0.05, 0.2],
            lifetime_range: [1.0, 3.0],
            gravity: [0.0, -9.81, 0.0],
            color_start: [1.0, 0.8, 0.2, 1.0],
            color_end: [1.0, 0.1, 0.0, 0.0],
            emit_direction: [0.0, 1.0, 0.0],
            cone_half_angle: 0.5,
        }
    }
}

/// A GPU-driven particle system.
#[allow(dead_code)]
pub struct GpuParticleSystem<D: RenderDevice = WgpuDevice> {
    config: GpuParticleConfig,
    position: [f32; 3],

    // Compute resources
    emit_pipeline: D::ComputePipeline,
    update_pipeline: D::ComputePipeline,
    particle_buffer: D::Buffer,
    counter_buffer: D::Buffer,
    params_buffer: D::Buffer,
    compute_bind_group: D::BindGroup,

    // Render resources
    render_pipeline: D::RenderPipeline,
    camera_buffer: D::Buffer,
    render_bind_group: D::BindGroup,

    // Emission accumulator
    emit_accumulator: f32,
    elapsed: f32,
}

impl<D: RenderDevice> GpuParticleSystem<D> {
    /// Create a new GPU particle system.
    pub fn new(
        device: &D,
        config: GpuParticleConfig,
        surface_format: euca_rhi::TextureFormat,
    ) -> Self {
        // ── Compute shader ──
        let compute_shader = device.create_shader(&euca_rhi::ShaderDesc {
            label: Some("particle_compute"),
            source: euca_rhi::ShaderSource::Wgsl(
                include_str!("../shaders/particle_compute.wgsl").into(),
            ),
        });

        // ── Compute bind group layout ──
        let compute_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("particle_compute_bgl"),
            entries: &[
                // particles storage (read_write)
                euca_rhi::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: euca_rhi::ShaderStages::COMPUTE,
                    ty: euca_rhi::BindingType::Buffer {
                        ty: euca_rhi::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // counters storage (read_write)
                euca_rhi::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: euca_rhi::ShaderStages::COMPUTE,
                    ty: euca_rhi::BindingType::Buffer {
                        ty: euca_rhi::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // params uniform
                euca_rhi::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: euca_rhi::ShaderStages::COMPUTE,
                    ty: euca_rhi::BindingType::Buffer {
                        ty: euca_rhi::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── Compute pipelines ──
        let emit_pipeline = device.create_compute_pipeline(&euca_rhi::ComputePipelineDesc {
            label: Some("particle_emit"),
            layout: &[&compute_bgl],
            module: &compute_shader,
            entry_point: "emit",
        });
        let update_pipeline = device.create_compute_pipeline(&euca_rhi::ComputePipelineDesc {
            label: Some("particle_update"),
            layout: &[&compute_bgl],
            module: &compute_shader,
            entry_point: "update",
        });

        // ── Buffers ──
        let particle_size = std::mem::size_of::<GpuParticle>() as u64;
        let particle_buffer = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some("particles"),
            size: particle_size * config.max_particles as u64,
            usage: euca_rhi::BufferUsages::STORAGE
                | euca_rhi::BufferUsages::COPY_DST
                | euca_rhi::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Counter buffer: one atomic u32 for alive_count
        let counter_buffer = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some("counters"),
            size: std::mem::size_of::<u32>() as u64,
            usage: euca_rhi::BufferUsages::STORAGE
                | euca_rhi::BufferUsages::COPY_DST
                | euca_rhi::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        device.write_buffer(&counter_buffer, 0, bytemuck::bytes_of(&0u32));

        let initial_params = EmitParamsGpu {
            emitter_position: [0.0; 3],
            emit_count: 0,
            speed_min: config.speed_range[0],
            speed_max: config.speed_range[1],
            size_min: config.size_range[0],
            size_max: config.size_range[1],
            lifetime_min: config.lifetime_range[0],
            lifetime_max: config.lifetime_range[1],
            _pad0: [0.0; 2],
            gravity: config.gravity,
            dt: 0.0,
            time: 0.0,
            max_particles: config.max_particles,
            _pad1: [0.0; 2],
            color_start: config.color_start,
            color_end: config.color_end,
            emit_direction: config.emit_direction,
            cone_half_angle: config.cone_half_angle,
        };
        let params_buffer = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some("emit_params"),
            size: std::mem::size_of::<EmitParamsGpu>() as u64,
            usage: euca_rhi::BufferUsages::UNIFORM | euca_rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&initial_params));

        // Compute bind group (shared between emit and update — same layout)
        let compute_bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("particle_compute_bg"),
            layout: &compute_bgl,
            entries: &[
                euca_rhi::BindGroupEntry {
                    binding: 0,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: &particle_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 1,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: &counter_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 2,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // ── Render pipeline ──
        let render_shader = device.create_shader(&euca_rhi::ShaderDesc {
            label: Some("particle_render"),
            source: euca_rhi::ShaderSource::Wgsl(
                include_str!("../shaders/particle_render.wgsl").into(),
            ),
        });

        let initial_cam = ParticleCameraGpu {
            view: [[0.0; 4]; 4],
            proj: [[0.0; 4]; 4],
            view_proj: [[0.0; 4]; 4],
            camera_right: [1.0, 0.0, 0.0],
            _pad0: 0.0,
            camera_up: [0.0, 1.0, 0.0],
            _pad1: 0.0,
        };
        let camera_buffer = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some("particle_camera"),
            size: std::mem::size_of::<ParticleCameraGpu>() as u64,
            usage: euca_rhi::BufferUsages::UNIFORM | euca_rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.write_buffer(&camera_buffer, 0, bytemuck::bytes_of(&initial_cam));

        let render_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("particle_render_bgl"),
            entries: &[
                // particles storage (read-only)
                euca_rhi::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: euca_rhi::ShaderStages::VERTEX,
                    ty: euca_rhi::BindingType::Buffer {
                        ty: euca_rhi::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // counters storage (read-only)
                euca_rhi::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: euca_rhi::ShaderStages::VERTEX,
                    ty: euca_rhi::BindingType::Buffer {
                        ty: euca_rhi::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // camera uniform
                euca_rhi::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: euca_rhi::ShaderStages::VERTEX,
                    ty: euca_rhi::BindingType::Buffer {
                        ty: euca_rhi::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let render_bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("particle_render_bg"),
            layout: &render_bgl,
            entries: &[
                euca_rhi::BindGroupEntry {
                    binding: 0,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: &particle_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 1,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: &counter_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 2,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: &camera_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let render_pipeline = device.create_render_pipeline(&euca_rhi::RenderPipelineDesc {
            label: Some("particle_render"),
            layout: &[&render_bgl],
            vertex: euca_rhi::VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[], // No vertex buffers — all data from storage
            },
            fragment: Some(euca_rhi::FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                targets: &[Some(euca_rhi::ColorTargetState {
                    format: surface_format,
                    blend: Some(euca_rhi::BlendState::ALPHA_BLENDING),
                    write_mask: euca_rhi::ColorWrites::ALL,
                })],
            }),
            primitive: euca_rhi::PrimitiveState::default(),
            depth_stencil: Some(euca_rhi::DepthStencilState {
                format: euca_rhi::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Particles don't write depth
                depth_compare: euca_rhi::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
        });

        Self {
            config,
            position: [0.0; 3],
            emit_pipeline,
            update_pipeline,
            particle_buffer,
            counter_buffer,
            params_buffer,
            compute_bind_group,
            render_pipeline,
            camera_buffer,
            render_bind_group,
            emit_accumulator: 0.0,
            elapsed: 0.0,
        }
    }

    /// Set the emitter world position.
    pub fn set_position(&mut self, pos: [f32; 3]) {
        self.position = pos;
    }

    /// Run the compute emit + update passes.
    pub fn update(&mut self, device: &D, encoder: &mut D::CommandEncoder, dt: f32) {
        self.elapsed += dt;
        self.emit_accumulator += self.config.emit_rate * dt;
        let emit_count = self.emit_accumulator as u32;
        self.emit_accumulator -= emit_count as f32;

        // Update params uniform
        let params = EmitParamsGpu {
            emitter_position: self.position,
            emit_count,
            speed_min: self.config.speed_range[0],
            speed_max: self.config.speed_range[1],
            size_min: self.config.size_range[0],
            size_max: self.config.size_range[1],
            lifetime_min: self.config.lifetime_range[0],
            lifetime_max: self.config.lifetime_range[1],
            _pad0: [0.0; 2],
            gravity: self.config.gravity,
            dt,
            time: self.elapsed,
            max_particles: self.config.max_particles,
            _pad1: [0.0; 2],
            color_start: self.config.color_start,
            color_end: self.config.color_end,
            emit_direction: self.config.emit_direction,
            cone_half_angle: self.config.cone_half_angle,
        };
        device.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Emit pass
        if emit_count > 0 {
            let workgroups = emit_count.div_ceil(64);
            let mut pass = device.begin_compute_pass(encoder, Some("particle_emit"));
            pass.set_pipeline(&self.emit_pipeline);
            pass.set_bind_group(0, &self.compute_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Update pass
        let max_workgroups = self.config.max_particles.div_ceil(64);
        {
            let mut pass = device.begin_compute_pass(encoder, Some("particle_update"));
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.compute_bind_group, &[]);
            pass.dispatch_workgroups(max_workgroups, 1, 1);
        }
    }

    /// Update camera uniforms for billboard rendering.
    pub fn set_camera(
        &self,
        device: &D,
        view: [[f32; 4]; 4],
        proj: [[f32; 4]; 4],
        view_proj: [[f32; 4]; 4],
        camera_right: [f32; 3],
        camera_up: [f32; 3],
    ) {
        let cam = ParticleCameraGpu {
            view,
            proj,
            view_proj,
            camera_right,
            _pad0: 0.0,
            camera_up,
            _pad1: 0.0,
        };
        device.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam));
    }

    /// Record the particle render pass into a render pass.
    pub fn draw(&self, render_pass: &mut impl RenderPassOps<D>) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.render_bind_group, &[]);
        // 6 vertices per quad, max_particles instances
        render_pass.draw(0..6, 0..self.config.max_particles);
    }

    /// Get the emitter configuration.
    pub fn config(&self) -> &GpuParticleConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_particle_struct_sizes() {
        assert_eq!(std::mem::size_of::<GpuParticle>(), 64);
        // EmitParams must be 16-byte aligned for uniform buffers
        assert_eq!(std::mem::size_of::<EmitParamsGpu>() % 16, 0);
    }

    #[test]
    fn default_config() {
        let config = GpuParticleConfig::default();
        assert_eq!(config.max_particles, 100_000);
        assert!(config.emit_rate > 0.0);
    }
}
