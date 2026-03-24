//! Screen-Space Global Illumination (SSGI) compute pass.
//!
//! # Overview
//! Approximates indirect diffuse lighting by ray-marching the depth buffer in
//! screen space. For each half-resolution pixel, a configurable number of rays
//! are cast in a cosine-weighted hemisphere around the surface normal. On hit,
//! the previous frame's HDR color is sampled and accumulated. Results are
//! temporally blended with the previous frame's GI output to reduce noise.
//!
//! # Architecture
//! - Runs as a compute shader at **half resolution** (width/2 x height/2).
//! - Uses ping-pong history textures for temporal accumulation (same pattern
//!   as [`crate::taa::TaaPass`]).
//! - Inputs: depth buffer, normal buffer (from prepass), previous frame HDR
//!   color, previous frame depth, camera matrices (current inverse VP +
//!   previous VP).
//! - Output: Rgba16Float half-res GI texture for compositing into the scene.
//!
//! # Usage
//! 1. Store [`SsgiSettings`] in your post-process settings.
//! 2. Create a [`SsgiPass`] once during renderer initialization.
//! 3. Each frame, call [`SsgiPass::execute`] with the required inputs.
//!    It returns a `&wgpu::TextureView` of the half-res GI result.

use euca_math::Mat4;

const SSGI_SHADER: &str = include_str!("../shaders/ssgi.wgsl");

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Runtime configuration for screen-space global illumination.
///
/// Intended to be embedded in [`crate::post_process::PostProcessSettings`] or
/// stored as a standalone ECS resource.
#[derive(Clone, Debug)]
pub struct SsgiSettings {
    /// Master toggle for the SSGI pass.
    pub enabled: bool,
    /// Number of rays cast per half-res pixel (4-8 recommended).
    pub ray_count: u32,
    /// Maximum world-space distance a ray can travel before giving up.
    pub max_distance: f32,
    /// Multiplier applied to the accumulated indirect radiance.
    pub intensity: f32,
    /// Blend factor for temporal accumulation (0.0 = no history, 1.0 = all
    /// history). Higher values reduce noise but increase ghosting.
    pub temporal_blend: f32,
}

impl Default for SsgiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            ray_count: 4,
            max_distance: 10.0,
            intensity: 1.0,
            temporal_blend: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// GPU uniform (must match SsgiParams in ssgi.wgsl)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsgiUniforms {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    screen_size: [f32; 2],
    ray_count: u32,
    max_steps: u32,
    max_distance: f32,
    intensity: f32,
    temporal_blend: f32,
    frame_index: u32,
}

// ---------------------------------------------------------------------------
// SsgiPass
// ---------------------------------------------------------------------------

/// Manages the compute pipeline, history textures, and output for SSGI.
pub struct SsgiPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    /// Ping-pong history textures for temporal accumulation (half resolution).
    history: [wgpu::Texture; 2],
    history_views: [wgpu::TextureView; 2],
    /// Which history texture to read from this frame.
    current_read: usize,
    sampler: wgpu::Sampler,
    #[allow(dead_code)]
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
    frame_index: u32,
    /// Half-res width.
    width: u32,
    /// Half-res height.
    height: u32,
}

/// Parameters for a single SSGI execution.
///
/// Bundles the per-frame GPU resources and matrices so
/// [`SsgiPass::execute`] has a clean signature.
pub struct SsgiExecuteParams<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Full-resolution depth buffer from the current frame.
    pub depth_view: &'a wgpu::TextureView,
    /// Full-resolution normal buffer from the prepass (encoded N*0.5+0.5).
    pub normal_view: &'a wgpu::TextureView,
    /// Previous frame HDR color (full resolution).
    pub prev_color_view: &'a wgpu::TextureView,
    /// Previous frame depth buffer (full resolution).
    pub prev_depth_view: &'a wgpu::TextureView,
    /// Current frame inverse view-projection matrix.
    pub inv_view_proj: &'a Mat4,
    /// Previous frame view-projection matrix.
    pub prev_view_proj: &'a Mat4,
    /// SSGI settings (ray count, intensity, etc.).
    pub settings: &'a SsgiSettings,
}

/// Number of ray-march steps per ray. Kept as a constant to balance quality
/// and performance; adjusting `max_distance` already controls effective
/// step size.
const SSGI_MAX_STEPS: u32 = 12;

impl SsgiPass {
    /// Create a new SSGI pass. `width` and `height` are **full resolution**;
    /// internal textures are allocated at half resolution.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssgi_compute"),
            source: wgpu::ShaderSource::Wgsl(SSGI_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssgi_bgl"),
            entries: &[
                // 0: depth texture (full-res, current frame)
                bgl_texture_entry(0),
                // 1: normal texture (full-res, from prepass)
                bgl_texture_entry(1),
                // 2: previous frame HDR color (filterable for bilinear sampling)
                bgl_filterable_texture_entry(2),
                // 3: previous frame depth (full-res)
                bgl_texture_entry(3),
                // 4: history GI texture (half-res, filterable)
                bgl_filterable_texture_entry(4),
                // 5: uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: output storage texture (half-res)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 7: linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssgi_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssgi_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssgi_uniforms"),
            size: std::mem::size_of::<SsgiUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssgi_linear_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (history, history_views) = Self::create_history_textures(device, half_w, half_h);
        let (output_texture, output_view) = Self::create_output_texture(device, half_w, half_h);

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            history,
            history_views,
            current_read: 0,
            sampler,
            output_texture,
            output_view,
            frame_index: 0,
            width: half_w,
            height: half_h,
        }
    }

    /// Resize internal textures when the window changes size.
    /// `width` and `height` are **full resolution**.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);

        if self.width == half_w && self.height == half_h {
            return;
        }
        self.width = half_w;
        self.height = half_h;

        let (history, history_views) = Self::create_history_textures(device, half_w, half_h);
        self.history = history;
        self.history_views = history_views;

        let (output_texture, output_view) = Self::create_output_texture(device, half_w, half_h);
        self.output_texture = output_texture;
        self.output_view = output_view;

        self.current_read = 0;
    }

    /// Execute the SSGI compute pass.
    ///
    /// Ray-marches the depth buffer, samples indirect color from the previous
    /// frame, accumulates temporally with history, and writes to the output
    /// texture.
    ///
    /// Returns the half-resolution GI texture view for compositing.
    pub fn execute(&mut self, params: SsgiExecuteParams<'_>) -> &wgpu::TextureView {
        // Upload uniforms.
        let uniforms = SsgiUniforms {
            inv_view_proj: params.inv_view_proj.to_cols_array_2d(),
            prev_view_proj: params.prev_view_proj.to_cols_array_2d(),
            screen_size: [self.width as f32, self.height as f32],
            ray_count: params.settings.ray_count,
            max_steps: SSGI_MAX_STEPS,
            max_distance: params.settings.max_distance,
            intensity: params.settings.intensity,
            temporal_blend: params.settings.temporal_blend,
            frame_index: self.frame_index,
        };
        params
            .queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let read_idx = self.current_read;
        let write_idx = 1 - self.current_read;

        let bind_group = params.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssgi_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(params.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(params.normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(params.prev_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(params.prev_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.history_views[read_idx]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        {
            let mut pass = params
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ssgi_compute"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // Copy output to history[write_idx] for next frame's temporal blend.
        params.encoder.copy_texture_to_texture(
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

        // Swap: next frame reads from write_idx.
        self.current_read = write_idx;
        self.frame_index = self.frame_index.wrapping_add(1);

        &self.output_view
    }

    /// Returns the output texture view (half-res GI result).
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// Half-resolution width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Half-resolution height.
    pub fn height(&self) -> u32 {
        self.height
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn create_history_textures(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> ([wgpu::Texture; 2], [wgpu::TextureView; 2]) {
        let desc = wgpu::TextureDescriptor {
            label: Some("ssgi_history"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
            label: Some("ssgi_output"),
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
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }
}

// -----------------------------------------------------------------------
// Bind group layout helpers
// -----------------------------------------------------------------------

fn bgl_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_filterable_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

// -----------------------------------------------------------------------
// Pure helper: compute effective step size from settings
// -----------------------------------------------------------------------

/// Compute the world-space distance per ray-march step given the current
/// SSGI settings. Useful for diagnostics and performance tuning.
pub fn step_size(settings: &SsgiSettings) -> f32 {
    if SSGI_MAX_STEPS == 0 {
        return 0.0;
    }
    settings.max_distance / SSGI_MAX_STEPS as f32
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_values() {
        let s = SsgiSettings::default();
        assert!(!s.enabled, "SSGI should be disabled by default");
        assert_eq!(s.ray_count, 4);
        assert!((s.max_distance - 10.0).abs() < f32::EPSILON);
        assert!((s.intensity - 1.0).abs() < f32::EPSILON);
        assert!((s.temporal_blend - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn settings_clone_is_independent() {
        let original = SsgiSettings {
            ray_count: 8,
            intensity: 2.0,
            ..SsgiSettings::default()
        };
        let mut cloned = original.clone();
        cloned.ray_count = 2;
        cloned.intensity = 0.5;

        assert_eq!(original.ray_count, 8, "Original should be unchanged");
        assert!(
            (original.intensity - 2.0).abs() < f32::EPSILON,
            "Original intensity should be unchanged"
        );
        assert_eq!(cloned.ray_count, 2);
    }

    #[test]
    fn uniforms_size_is_gpu_aligned() {
        let size = std::mem::size_of::<SsgiUniforms>();
        assert_eq!(
            size % 16,
            0,
            "SsgiUniforms size ({size}) must be 16-byte aligned"
        );
    }

    #[test]
    fn uniforms_layout_matches_expected_size() {
        // 2x mat4x4 (128 bytes) + vec2f (8) + u32 (4) + u32 (4) + f32 (4)
        // + f32 (4) + f32 (4) + u32 (4) = 128 + 32 = 160 bytes
        let size = std::mem::size_of::<SsgiUniforms>();
        assert_eq!(size, 160);
    }

    #[test]
    fn step_size_calculation() {
        let settings = SsgiSettings::default(); // max_distance = 10.0
        let s = step_size(&settings);
        let expected = 10.0 / SSGI_MAX_STEPS as f32;
        assert!(
            (s - expected).abs() < f32::EPSILON,
            "Step size should be max_distance / max_steps"
        );
    }

    #[test]
    fn step_size_with_custom_distance() {
        let settings = SsgiSettings {
            max_distance: 24.0,
            ..SsgiSettings::default()
        };
        let s = step_size(&settings);
        let expected = 24.0 / SSGI_MAX_STEPS as f32;
        assert!((s - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn shader_source_is_valid() {
        assert!(!SSGI_SHADER.is_empty());
        assert!(SSGI_SHADER.contains("@compute"));
        assert!(SSGI_SHADER.contains("@workgroup_size(8, 8)"));
        assert!(SSGI_SHADER.contains("fn main"));
        assert!(SSGI_SHADER.contains("SsgiParams"));
    }

    #[test]
    fn settings_enabled_toggle() {
        let mut s = SsgiSettings::default();
        assert!(!s.enabled, "Default should be disabled");
        s.enabled = true;
        assert!(s.enabled, "Should be toggleable to enabled");
        s.enabled = false;
        assert!(!s.enabled, "Should be toggleable back to disabled");
    }

    #[test]
    fn half_res_dimensions() {
        // Verify the half-res logic: for 1920x1080, half is 960x540.
        let half_w = (1920u32 / 2).max(1);
        let half_h = (1080u32 / 2).max(1);
        assert_eq!(half_w, 960);
        assert_eq!(half_h, 540);

        // For small sizes, ensure at least 1x1.
        let tiny_w = (1u32 / 2).max(1);
        let tiny_h = (1u32 / 2).max(1);
        assert_eq!(tiny_w, 1);
        assert_eq!(tiny_h, 1);
    }
}
