//! Temporal Anti-Aliasing (TAA) resolve pass.
//!
//! Blends the current jittered frame with accumulated history using
//! neighborhood clamping to prevent ghosting. Runs as a compute shader
//! between the main PBR pass and post-processing.

use euca_math::Mat4;

const TAA_SHADER: &str = include_str!("../shaders/taa_resolve.wgsl");

/// GPU-side TAA parameters (must match shader struct layout).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaParamsGpu {
    inv_vp: [[f32; 4]; 4],
    prev_vp: [[f32; 4]; 4],
    jitter: [f32; 2],
    resolution: [f32; 2],
    blend_factor: f32,
    _pad: [f32; 3],
}

/// TAA resolve pass — manages history textures and dispatches the resolve shader.
pub struct TaaPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Ping-pong history textures (Rgba16Float, full resolution).
    history: [wgpu::Texture; 2],
    history_views: [wgpu::TextureView; 2],
    /// Which history texture was written to last frame (read from this, write to other).
    current_read: usize,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Current dimensions (recreate textures on resize).
    width: u32,
    height: u32,
    /// Output texture view (the resolved TAA result for post-processing to read).
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
}

impl TaaPass {
    /// Create a new TAA pass. Call once at renderer init.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa_resolve"),
            source: wgpu::ShaderSource::Wgsl(TAA_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa_bind_group_layout"),
            entries: &[
                // 0: current frame (texture_2d<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 1: history frame (texture_2d<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: output storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 5: linear sampler (for bilinear history sampling)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("taa_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa_uniforms"),
            size: std::mem::size_of::<TaaParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("taa_linear_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (history, history_views) = Self::create_history_textures(device, width, height);
        let (output_texture, output_view) = Self::create_output_texture(device, width, height);

        Self {
            pipeline,
            bind_group_layout,
            history,
            history_views,
            current_read: 0,
            uniform_buffer,
            sampler,
            width,
            height,
            output_texture,
            output_view,
        }
    }

    fn create_history_textures(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> ([wgpu::Texture; 2], [wgpu::TextureView; 2]) {
        let desc = wgpu::TextureDescriptor {
            label: Some("taa_history"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let t0 = device.create_texture(&desc);
        let t1 = device.create_texture(&desc);
        let v0 = t0.create_view(&Default::default());
        let v1 = t1.create_view(&Default::default());
        ([t0, t1], [v0, v1])
    }

    fn create_output_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("taa_output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    /// Resize history textures when the window changes size.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        let (history, views) = Self::create_history_textures(device, width, height);
        self.history = history;
        self.history_views = views;
        let (out_tex, out_view) = Self::create_output_texture(device, width, height);
        self.output_texture = out_tex;
        self.output_view = out_view;
        self.current_read = 0;
    }

    /// Returns the output texture view (resolved TAA result).
    /// Post-processing reads from this instead of the raw PBR output.
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// Execute the TAA resolve pass.
    ///
    /// Reads the current HDR frame and depth, reprojects from history,
    /// blends with neighborhood clamping, writes to output.
    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        current_frame_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        inv_vp: &Mat4,
        prev_vp: &Mat4,
        jitter: [f32; 2],
    ) {
        // Upload uniforms
        let params = TaaParamsGpu {
            inv_vp: inv_vp.to_cols_array_2d(),
            prev_vp: prev_vp.to_cols_array_2d(),
            jitter,
            resolution: [self.width as f32, self.height as f32],
            blend_factor: 0.1,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));

        // read_idx = current_read (history to sample from)
        // write_idx = 1 - current_read (history to write to = output)
        let read_idx = self.current_read;
        let write_idx = 1 - self.current_read;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(current_frame_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.history_views[read_idx]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("taa_resolve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // Copy output to history[write_idx] for next frame
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.history[write_idx],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        // Swap: next frame reads from write_idx
        self.current_read = write_idx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taa_params_gpu_size() {
        // Ensure the GPU struct is the expected size (must match shader layout)
        assert_eq!(
            std::mem::size_of::<TaaParamsGpu>(),
            // 2 mat4x4 (64 each) + vec2 + vec2 + f32 + vec3 pad = 128 + 8 + 8 + 4 + 12 = 160
            160,
        );
    }
}
