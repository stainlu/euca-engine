//! Enhanced Temporal Anti-Aliasing (TAA) resolve pass.
//!
//! Blends the current jittered frame with accumulated history using
//! velocity-buffer reprojection, variance-based neighborhood clamping in
//! YCoCg space, and disocclusion detection. Runs as a compute shader
//! between the main PBR pass and post-processing.

use euca_math::Mat4;
use euca_rhi::{ComputePassOps, RenderDevice};

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
    variance_gamma: f32,
    depth_threshold: f32,
    _pad: f32,
}

/// TAA resolve pass — manages history textures and dispatches the resolve shader.
///
/// Fully generic over [`euca_rhi::RenderDevice`] — defaults to
/// [`euca_rhi::wgpu_backend::WgpuDevice`] for backward compatibility.
pub struct TaaPass<D: RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pipeline: D::ComputePipeline,
    bind_group_layout: D::BindGroupLayout,
    /// Ping-pong history textures (Rgba16Float, full resolution).
    history: [D::Texture; 2],
    history_views: [D::TextureView; 2],
    /// Which history texture was written to last frame (read from this, write to other).
    current_read: usize,
    uniform_buffer: D::Buffer,
    sampler: D::Sampler,
    /// Current dimensions (recreate textures on resize).
    width: u32,
    height: u32,
    /// Output texture view (the resolved TAA result for post-processing to read).
    output_texture: D::Texture,
    output_view: D::TextureView,
}

impl<D: RenderDevice> TaaPass<D> {
    /// Create a new TAA pass. Call once at renderer init.
    pub fn new(device: &D, width: u32, height: u32) -> Self {
        let shader_module = device.create_shader(&euca_rhi::ShaderDesc {
            label: Some("taa_resolve"),
            source: euca_rhi::ShaderSource::Wgsl(TAA_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("taa_bind_group_layout"),
            entries: &[
                // 0: current frame (texture_2d<f32>)
                euca_rhi::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: euca_rhi::ShaderStages::COMPUTE,
                    ty: euca_rhi::BindingType::Texture {
                        sample_type: euca_rhi::TextureSampleType::Float { filterable: true },
                        view_dimension: euca_rhi::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 1: history frame (texture_2d<f32>)
                euca_rhi::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: euca_rhi::ShaderStages::COMPUTE,
                    ty: euca_rhi::BindingType::Texture {
                        sample_type: euca_rhi::TextureSampleType::Float { filterable: true },
                        view_dimension: euca_rhi::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: depth texture
                euca_rhi::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: euca_rhi::ShaderStages::COMPUTE,
                    ty: euca_rhi::BindingType::Texture {
                        sample_type: euca_rhi::TextureSampleType::Depth,
                        view_dimension: euca_rhi::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: uniform buffer
                euca_rhi::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: euca_rhi::ShaderStages::COMPUTE,
                    ty: euca_rhi::BindingType::Buffer {
                        ty: euca_rhi::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: output storage texture
                euca_rhi::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: euca_rhi::ShaderStages::COMPUTE,
                    ty: euca_rhi::BindingType::StorageTexture {
                        access: euca_rhi::StorageTextureAccess::WriteOnly,
                        format: euca_rhi::TextureFormat::Rgba16Float,
                        view_dimension: euca_rhi::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 5: linear sampler (for bilinear history sampling)
                euca_rhi::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: euca_rhi::ShaderStages::COMPUTE,
                    ty: euca_rhi::BindingType::Sampler(euca_rhi::SamplerBindingType::Filtering),
                    count: None,
                },
                // 6: velocity buffer (Rg16Float motion vectors)
                euca_rhi::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: euca_rhi::ShaderStages::COMPUTE,
                    ty: euca_rhi::BindingType::Texture {
                        sample_type: euca_rhi::TextureSampleType::Float { filterable: false },
                        view_dimension: euca_rhi::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline = device.create_compute_pipeline(&euca_rhi::ComputePipelineDesc {
            label: Some("taa_resolve_pipeline"),
            layout: &[&bind_group_layout],
            module: &shader_module,
            entry_point: "main",
        });

        let uniform_buffer = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some("taa_uniforms"),
            size: std::mem::size_of::<TaaParamsGpu>() as u64,
            usage: euca_rhi::BufferUsages::UNIFORM | euca_rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&euca_rhi::SamplerDesc {
            label: Some("taa_linear_sampler"),
            mag_filter: euca_rhi::FilterMode::Linear,
            min_filter: euca_rhi::FilterMode::Linear,
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
        device: &D,
        width: u32,
        height: u32,
    ) -> ([D::Texture; 2], [D::TextureView; 2]) {
        let desc = euca_rhi::TextureDesc {
            label: Some("taa_history"),
            size: euca_rhi::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: euca_rhi::TextureDimension::D2,
            format: euca_rhi::TextureFormat::Rgba16Float,
            usage: euca_rhi::TextureUsages::TEXTURE_BINDING
                | euca_rhi::TextureUsages::STORAGE_BINDING
                | euca_rhi::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let t0 = device.create_texture(&desc);
        let t1 = device.create_texture(&desc);
        let v0 = device.create_texture_view(&t0, &Default::default());
        let v1 = device.create_texture_view(&t1, &Default::default());
        ([t0, t1], [v0, v1])
    }

    fn create_output_texture(device: &D, width: u32, height: u32) -> (D::Texture, D::TextureView) {
        let tex = device.create_texture(&euca_rhi::TextureDesc {
            label: Some("taa_output"),
            size: euca_rhi::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: euca_rhi::TextureDimension::D2,
            format: euca_rhi::TextureFormat::Rgba16Float,
            usage: euca_rhi::TextureUsages::TEXTURE_BINDING
                | euca_rhi::TextureUsages::STORAGE_BINDING
                | euca_rhi::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = device.create_texture_view(&tex, &Default::default());
        (tex, view)
    }

    /// Resize history textures when the window changes size.
    pub fn resize(&mut self, device: &D, width: u32, height: u32) {
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
    pub fn output_view(&self) -> &D::TextureView {
        &self.output_view
    }

    /// Returns a reference to the output texture (for copy operations).
    pub fn output_texture(&self) -> &D::Texture {
        &self.output_texture
    }

    /// Execute the enhanced TAA resolve pass.
    ///
    /// Uses velocity-buffer reprojection instead of depth-based reprojection
    /// for accurate per-pixel motion tracking. Applies variance-based
    /// neighborhood clamping in YCoCg space and detects disocclusion events.
    // clippy::too_many_arguments — TAA resolve requires the current frame,
    // depth, velocity, two view-projection matrices, and jitter; all are
    // distinct GPU resources or per-frame parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        device: &D,
        encoder: &mut D::CommandEncoder,
        current_frame_view: &D::TextureView,
        depth_view: &D::TextureView,
        velocity_view: &D::TextureView,
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
            variance_gamma: 1.0,
            depth_threshold: 0.01,
            _pad: 0.0,
        };
        device.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));

        // read_idx = current_read (history to sample from)
        // write_idx = 1 - current_read (history to write to = output)
        let read_idx = self.current_read;
        let write_idx = 1 - self.current_read;

        let bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("taa_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                euca_rhi::BindGroupEntry {
                    binding: 0,
                    resource: euca_rhi::BindingResource::TextureView(current_frame_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 1,
                    resource: euca_rhi::BindingResource::TextureView(&self.history_views[read_idx]),
                },
                euca_rhi::BindGroupEntry {
                    binding: 2,
                    resource: euca_rhi::BindingResource::TextureView(depth_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 3,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 4,
                    resource: euca_rhi::BindingResource::TextureView(&self.output_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 5,
                    resource: euca_rhi::BindingResource::Sampler(&self.sampler),
                },
                euca_rhi::BindGroupEntry {
                    binding: 6,
                    resource: euca_rhi::BindingResource::TextureView(velocity_view),
                },
            ],
        });

        {
            let mut pass = device.begin_compute_pass(encoder, Some("taa_resolve"));
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // Copy output to history[write_idx] for next frame
        let copy_size = euca_rhi::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };
        device.copy_texture_to_texture(
            encoder,
            &euca_rhi::TexelCopyTextureInfo {
                texture: &self.output_texture,
                mip_level: 0,
                origin: euca_rhi::Origin3d::default(),
                aspect: euca_rhi::TextureAspect::All,
            },
            &euca_rhi::TexelCopyTextureInfo {
                texture: &self.history[write_idx],
                mip_level: 0,
                origin: euca_rhi::Origin3d::default(),
                aspect: euca_rhi::TextureAspect::All,
            },
            copy_size,
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
        // Ensure the GPU struct is the expected size (must match shader layout).
        // 2 mat4x4 (64 each) + vec2 + vec2 + f32 + f32 + f32 + f32 pad = 128 + 8 + 8 + 16 = 160
        assert_eq!(std::mem::size_of::<TaaParamsGpu>(), 160);
    }

    #[test]
    fn taa_params_gpu_alignment() {
        assert_eq!(
            std::mem::size_of::<TaaParamsGpu>() % 16,
            0,
            "TaaParamsGpu must be 16-byte aligned for uniform buffers"
        );
    }

    #[test]
    fn taa_shader_source_valid() {
        assert!(!TAA_SHADER.is_empty());
        assert!(TAA_SHADER.contains("@compute"));
        assert!(TAA_SHADER.contains("@workgroup_size(8, 8)"));
        assert!(TAA_SHADER.contains("fn main"));
        assert!(
            TAA_SHADER.contains("velocity_tex"),
            "Shader must reference velocity texture"
        );
        assert!(
            TAA_SHADER.contains("variance_gamma"),
            "Shader must use variance-based clamping"
        );
        assert!(
            TAA_SHADER.contains("depth_threshold"),
            "Shader must have disocclusion detection"
        );
    }

    #[test]
    fn taa_params_default_values() {
        // Verify the default uniform values are reasonable.
        let params = TaaParamsGpu {
            inv_vp: [[0.0; 4]; 4],
            prev_vp: [[0.0; 4]; 4],
            jitter: [0.0; 2],
            resolution: [1920.0, 1080.0],
            blend_factor: 0.1,
            variance_gamma: 1.0,
            depth_threshold: 0.01,
            _pad: 0.0,
        };

        assert!(
            params.blend_factor > 0.0 && params.blend_factor <= 0.2,
            "Blend factor should be small (mostly history)"
        );
        assert!(
            params.variance_gamma >= 0.5 && params.variance_gamma <= 3.0,
            "Variance gamma should be in a reasonable range"
        );
        assert!(
            params.depth_threshold > 0.0 && params.depth_threshold < 1.0,
            "Depth threshold should be a small positive value"
        );
    }
}
