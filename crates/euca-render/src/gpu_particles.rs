//! GPU compute particle system — emit and update millions of particles on the GPU.
//!
//! Uses two compute passes (emit + update) and an instanced draw for billboard rendering.

use crate::compute::{ComputePipeline, ComputePipelineDesc, GpuBuffer, dispatch_compute};
use crate::gpu::GpuContext;

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
pub struct GpuParticleSystem<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    config: GpuParticleConfig,
    position: [f32; 3],

    // Compute resources
    emit_pipeline: ComputePipeline,
    update_pipeline: ComputePipeline,
    particle_buffer: GpuBuffer,
    counter_buffer: GpuBuffer,
    params_buffer: GpuBuffer,
    compute_bind_group: D::BindGroup,

    // Render resources
    render_pipeline: D::RenderPipeline,
    camera_buffer: GpuBuffer,
    render_bind_group: D::BindGroup,

    // Emission accumulator
    emit_accumulator: f32,
    elapsed: f32,
}

impl GpuParticleSystem {
    /// Create a new GPU particle system.
    pub fn new(
        gpu: &GpuContext,
        config: GpuParticleConfig,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let device = &gpu.device;

        // ── Compute pipelines ──
        let emit_pipeline = ComputePipeline::new(
            device,
            &ComputePipelineDesc {
                label: "particle_emit",
                shader_source: include_str!("../shaders/particle_compute.wgsl"),
                entry_point: "emit",
            },
        );
        let update_pipeline = ComputePipeline::new(
            device,
            &ComputePipelineDesc {
                label: "particle_update",
                shader_source: include_str!("../shaders/particle_compute.wgsl"),
                entry_point: "update",
            },
        );

        // ── Buffers ──
        let particle_size = std::mem::size_of::<GpuParticle>() as u64;
        let particle_buffer = GpuBuffer::new_storage(
            device,
            particle_size * config.max_particles as u64,
            "particles",
        );

        // Counter buffer: one atomic u32 for alive_count
        let counter_buffer = GpuBuffer::new_storage_with_data(device, &[0u32], "counters");

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
        let params_buffer =
            GpuBuffer::new_uniform_with_data(device, &initial_params, "emit_params");

        // Compute bind group (shared between emit and update — same layout)
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_compute_bg"),
            layout: emit_pipeline.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counter_buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.raw().as_entire_binding(),
                },
            ],
        });

        // ── Render pipeline ──
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_render"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/particle_render.wgsl").into(),
            ),
        });

        let camera_buffer = GpuBuffer::new_uniform_with_data(
            device,
            &ParticleCameraGpu {
                view: [[0.0; 4]; 4],
                proj: [[0.0; 4]; 4],
                view_proj: [[0.0; 4]; 4],
                camera_right: [1.0, 0.0, 0.0],
                _pad0: 0.0,
                camera_up: [0.0, 1.0, 0.0],
                _pad1: 0.0,
            },
            "particle_camera",
        );

        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_render_bgl"),
            entries: &[
                // particles storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // counters storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // camera uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_render_bg"),
            layout: &render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counter_buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buffer.raw().as_entire_binding(),
                },
            ],
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle_render_layout"),
                bind_group_layouts: &[&render_bgl],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle_render"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffers — all data from storage
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Particles don't write depth
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
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
    pub fn update(&mut self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, dt: f32) {
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
        queue.write_buffer(self.params_buffer.raw(), 0, bytemuck::bytes_of(&params));

        // Emit pass
        if emit_count > 0 {
            let workgroups = emit_count.div_ceil(64);
            dispatch_compute(
                encoder,
                &self.emit_pipeline,
                &[&self.compute_bind_group],
                [workgroups, 1, 1],
                None,
            );
        }

        // Update pass
        let max_workgroups = self.config.max_particles.div_ceil(64);
        dispatch_compute(
            encoder,
            &self.update_pipeline,
            &[&self.compute_bind_group],
            [max_workgroups, 1, 1],
            None,
        );
    }

    /// Update camera uniforms for billboard rendering.
    pub fn set_camera(
        &self,
        queue: &wgpu::Queue,
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
        queue.write_buffer(self.camera_buffer.raw(), 0, bytemuck::bytes_of(&cam));
    }

    /// Record the particle render pass into a render pass.
    pub fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
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
