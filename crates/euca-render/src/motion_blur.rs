//! Per-pixel motion blur post-process pass.
//!
//! # Architecture
//! Uses a two-pass compute approach for efficient velocity-based motion blur:
//!
//! 1. **Tile pass**: Divides the screen into 16x16 tiles and computes the
//!    maximum velocity magnitude per tile. This enables early-out for static
//!    regions — the majority of pixels in a typical scene.
//!
//! 2. **Blur pass**: For each pixel, if its tile has significant motion,
//!    samples along the pixel's velocity vector with distance-based weighting.
//!    The number of taps is configurable (4–16).
//!
//! # Integration
//! Run after TAA resolve and before tonemapping. The velocity buffer from
//! `VelocityPipeline` (WU4) provides per-pixel Rg16Float motion vectors.

use euca_rhi::pass::ComputePassOps;
use euca_rhi::{
    BindGroupLayoutDesc, BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDesc,
    BufferUsages, ComputePipelineDesc, Extent3d, ShaderDesc, ShaderSource, ShaderStages,
    StorageTextureAccess, TextureDesc, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDesc, TextureViewDimension,
};

const MOTION_BLUR_SHADER: &str = include_str!("../shaders/motion_blur.wgsl");

/// Tile size for the velocity tile-max pass.
const TILE_SIZE: u32 = 16;

// ---------------------------------------------------------------------------
// Settings (ECS resource)
// ---------------------------------------------------------------------------

/// Configuration for per-pixel motion blur.
///
/// Intended to be stored as an ECS resource and read each frame.
#[derive(Clone, Debug)]
pub struct MotionBlurSettings {
    /// Master switch.
    pub enabled: bool,
    /// Velocity multiplier — scales the blur strength (default 1.0).
    pub intensity: f32,
    /// Number of samples along the velocity vector (default 8, range 4–16).
    pub sample_count: u32,
    /// Maximum blur length in pixels (default 40.0).
    pub max_velocity: f32,
}

impl Default for MotionBlurSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 1.0,
            sample_count: 8,
            max_velocity: 40.0,
        }
    }
}

// ---------------------------------------------------------------------------
// GPU uniform (must match MotionBlurParams in the WGSL shader)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionBlurParamsGpu {
    resolution: [f32; 2],
    inv_resolution: [f32; 2],
    intensity: f32,
    max_velocity: f32,
    sample_count: u32,
    tile_size: u32,
    tile_count: [u32; 2],
    _pad: [u32; 2],
}

// ---------------------------------------------------------------------------
// MotionBlurPass
// ---------------------------------------------------------------------------

/// Manages the compute pipelines and textures for velocity-based motion blur.
///
/// Generic over [`euca_rhi::RenderDevice`] — defaults to [`WgpuDevice`] for
/// backward compatibility.
pub struct MotionBlurPass<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    tile_pipeline: D::ComputePipeline,
    blur_pipeline: D::ComputePipeline,
    bind_group_layout: D::BindGroupLayout,
    uniform_buffer: D::Buffer,
    /// Per-tile maximum velocity (Rg16Float, ceil(width/16) x ceil(height/16)).
    tile_max_texture: D::Texture,
    tile_max_view: D::TextureView,
    /// Output blurred color (Rgba16Float, full resolution).
    output_texture: D::Texture,
    output_view: D::TextureView,
    width: u32,
    height: u32,
}

impl<D: euca_rhi::RenderDevice> MotionBlurPass<D> {
    /// Create the motion blur pass for the given screen dimensions.
    pub fn new(device: &D, width: u32, height: u32) -> Self {
        let shader_module = device.create_shader(&ShaderDesc {
            label: Some("motion_blur"),
            source: ShaderSource::Wgsl(MOTION_BLUR_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("motion_blur_bgl"),
            entries: &[
                // 0: uniform buffer
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: color input texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: velocity texture (Rg16Float)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: tile max velocity (storage read_write)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: TextureFormat::Rg16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 4: output color (storage write)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let tile_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            label: Some("motion_blur_tile"),
            layout: &[&bind_group_layout],
            module: &shader_module,
            entry_point: "tile_max",
        });

        let blur_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            label: Some("motion_blur_blur"),
            layout: &[&bind_group_layout],
            module: &shader_module,
            entry_point: "blur",
        });

        let uniform_buffer = device.create_buffer(&BufferDesc {
            label: Some("motion_blur_params"),
            size: std::mem::size_of::<MotionBlurParamsGpu>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (tile_max_texture, tile_max_view) = Self::create_tile_texture(device, width, height);
        let (output_texture, output_view) = Self::create_output_texture(device, width, height);

        Self {
            tile_pipeline,
            blur_pipeline,
            bind_group_layout,
            uniform_buffer,
            tile_max_texture,
            tile_max_view,
            output_texture,
            output_view,
            width,
            height,
        }
    }

    /// Recreate textures after a window resize.
    pub fn resize(&mut self, device: &D, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;

        let (tile_tex, tile_view) = Self::create_tile_texture(device, width, height);
        self.tile_max_texture = tile_tex;
        self.tile_max_view = tile_view;

        let (out_tex, out_view) = Self::create_output_texture(device, width, height);
        self.output_texture = out_tex;
        self.output_view = out_view;
    }

    /// Returns a reference to the output texture.
    pub fn output_texture(&self) -> &D::Texture {
        &self.output_texture
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn create_tile_texture(device: &D, width: u32, height: u32) -> (D::Texture, D::TextureView) {
        let tile_w = width.div_ceil(TILE_SIZE);
        let tile_h = height.div_ceil(TILE_SIZE);

        let tex = device.create_texture(&TextureDesc {
            label: Some("motion_blur_tile_max"),
            size: Extent3d {
                width: tile_w,
                height: tile_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = device.create_texture_view(&tex, &TextureViewDesc::default());
        (tex, view)
    }

    fn create_output_texture(device: &D, width: u32, height: u32) -> (D::Texture, D::TextureView) {
        let tex = device.create_texture(&TextureDesc {
            label: Some("motion_blur_output"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = device.create_texture_view(&tex, &TextureViewDesc::default());
        (tex, view)
    }

    /// Returns the output texture view (motion-blurred result).
    pub fn output_view(&self) -> &D::TextureView {
        &self.output_view
    }

    /// Execute the motion blur pass (tile + blur).
    pub fn execute(
        &self,
        device: &D,
        encoder: &mut D::CommandEncoder,
        color_view: &D::TextureView,
        velocity_view: &D::TextureView,
        settings: &MotionBlurSettings,
    ) {
        let tile_w = self.width.div_ceil(TILE_SIZE);
        let tile_h = self.height.div_ceil(TILE_SIZE);

        let params = MotionBlurParamsGpu {
            resolution: [self.width as f32, self.height as f32],
            inv_resolution: [1.0 / self.width as f32, 1.0 / self.height as f32],
            intensity: settings.intensity,
            max_velocity: settings.max_velocity,
            sample_count: settings.sample_count.clamp(4, 16),
            tile_size: TILE_SIZE,
            tile_count: [tile_w, tile_h],
            _pad: [0; 2],
        };
        device.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("motion_blur_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                euca_rhi::BindGroupEntry {
                    binding: 0,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 1,
                    resource: euca_rhi::BindingResource::TextureView(color_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 2,
                    resource: euca_rhi::BindingResource::TextureView(velocity_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 3,
                    resource: euca_rhi::BindingResource::TextureView(&self.tile_max_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 4,
                    resource: euca_rhi::BindingResource::TextureView(&self.output_view),
                },
            ],
        });

        // Pass 1: Tile max velocity
        {
            let mut pass = device.begin_compute_pass(encoder, Some("motion_blur_tile"));
            pass.set_pipeline(&self.tile_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Each workgroup is 16x16, processing one tile
            pass.dispatch_workgroups(tile_w, tile_h, 1);
        }

        // Pass 2: Directional blur
        {
            let mut pass = device.begin_compute_pass(encoder, Some("motion_blur_blur"));
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions (testable without GPU)
// ---------------------------------------------------------------------------

/// Compute the number of tiles needed for the given screen dimension and tile size.
pub fn tile_count(screen_dim: u32, tile_size: u32) -> u32 {
    screen_dim.div_ceil(tile_size)
}

/// Clamp a velocity magnitude to the maximum allowed blur length.
pub fn clamp_velocity(velocity: [f32; 2], max_magnitude: f32) -> [f32; 2] {
    let mag_sq = velocity[0] * velocity[0] + velocity[1] * velocity[1];
    if mag_sq <= max_magnitude * max_magnitude {
        return velocity;
    }
    let mag = mag_sq.sqrt();
    let scale = max_magnitude / mag;
    [velocity[0] * scale, velocity[1] * scale]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_defaults() {
        let s = MotionBlurSettings::default();
        assert!(!s.enabled, "Motion blur should be disabled by default");
        assert!(
            (s.intensity - 1.0).abs() < 1e-6,
            "Default intensity should be 1.0"
        );
        assert_eq!(s.sample_count, 8, "Default sample count should be 8");
        assert!(
            (s.max_velocity - 40.0).abs() < 1e-6,
            "Default max velocity should be 40.0 pixels"
        );
    }

    #[test]
    fn settings_clone_is_independent() {
        let original = MotionBlurSettings {
            intensity: 0.5,
            ..MotionBlurSettings::default()
        };
        let mut cloned = original.clone();
        cloned.intensity = 2.0;

        assert!(
            (original.intensity - 0.5).abs() < 1e-8,
            "Original should be unchanged"
        );
        assert!(
            (cloned.intensity - 2.0).abs() < 1e-8,
            "Clone should be updated"
        );
    }

    #[test]
    fn tile_count_exact_division() {
        assert_eq!(tile_count(1920, 16), 120);
        assert_eq!(tile_count(1080, 16), 68); // 1080 / 16 = 67.5 -> 68
    }

    #[test]
    fn tile_count_with_remainder() {
        assert_eq!(tile_count(100, 16), 7); // 100 / 16 = 6.25 -> 7
        assert_eq!(tile_count(1, 16), 1);
    }

    #[test]
    fn clamp_velocity_within_bounds() {
        let v = [10.0_f32, 0.0];
        let clamped = clamp_velocity(v, 40.0);
        assert!((clamped[0] - 10.0).abs() < 1e-6);
        assert!((clamped[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn clamp_velocity_exceeds_bounds() {
        let v = [30.0_f32, 40.0]; // magnitude = 50
        let clamped = clamp_velocity(v, 25.0);
        let mag = (clamped[0] * clamped[0] + clamped[1] * clamped[1]).sqrt();
        assert!(
            (mag - 25.0).abs() < 1e-4,
            "Clamped magnitude should be max_velocity"
        );
    }

    #[test]
    fn clamp_velocity_zero() {
        let v = [0.0_f32, 0.0];
        let clamped = clamp_velocity(v, 40.0);
        assert_eq!(clamped, [0.0, 0.0]);
    }

    #[test]
    fn motion_blur_params_gpu_alignment() {
        let size = std::mem::size_of::<MotionBlurParamsGpu>();
        assert_eq!(
            size % 16,
            0,
            "MotionBlurParamsGpu ({size} bytes) must be 16-byte aligned"
        );
    }

    #[test]
    fn shader_source_valid() {
        assert!(!MOTION_BLUR_SHADER.is_empty());
        assert!(MOTION_BLUR_SHADER.contains("@compute"));
        assert!(MOTION_BLUR_SHADER.contains("fn tile_max"));
        assert!(MOTION_BLUR_SHADER.contains("fn blur"));
        assert!(MOTION_BLUR_SHADER.contains("MotionBlurParams"));
    }

    #[test]
    fn sample_count_clamped_in_range() {
        // Verify the clamping logic used in execute()
        assert_eq!(3_u32.clamp(4, 16), 4);
        assert_eq!(8_u32.clamp(4, 16), 8);
        assert_eq!(20_u32.clamp(4, 16), 16);
    }
}
