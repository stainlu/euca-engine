//! Depth of Field (DoF) post-process pass.
//!
//! # Architecture
//! Implements a physically-based thin-lens DoF model using a two-pass compute
//! approach:
//!
//! 1. **CoC pass**: Computes the circle-of-confusion (CoC) for each pixel from
//!    depth, focus distance, aperture, and focal length. The thin-lens formula
//!    `coc = |1/focus - 1/depth| * aperture * focal_length` maps to pixel-space
//!    blur radii. Positive CoC = background blur, negative = foreground blur.
//!
//! 2. **Gather pass**: Variable-radius disk blur using a 16-tap Poisson-disk
//!    pattern scaled by the CoC. Near-field and far-field are handled separately
//!    to prevent background bleeding onto sharp foreground objects.
//!
//! # Integration
//! Run after motion blur (if enabled) and before tonemapping. Reads the depth
//! buffer and HDR color texture.

use euca_rhi::{
    BindGroupLayoutDesc, BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDesc,
    BufferUsages, ComputePipelineDesc, Extent3d, ShaderDesc, ShaderSource, ShaderStages,
    StorageTextureAccess, TextureDesc, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDesc, TextureViewDimension,
};

const DOF_SHADER: &str = include_str!("../shaders/dof.wgsl");

// ---------------------------------------------------------------------------
// Settings (ECS resource)
// ---------------------------------------------------------------------------

/// Configuration for depth of field.
///
/// Intended to be stored as an ECS resource and read each frame.
#[derive(Clone, Debug)]
pub struct DofSettings {
    /// Master switch.
    pub enabled: bool,
    /// Distance to the focal plane in world units (default 10.0).
    pub focus_distance: f32,
    /// Aperture diameter — larger values produce shallower DoF (default 0.05).
    pub aperture: f32,
    /// Focal length in world units (e.g. 0.05 for a 50mm lens at 1:1 scale).
    pub focal_length: f32,
    /// Maximum blur radius in pixels (default 20.0).
    pub max_blur_radius: f32,
    /// Near clip plane distance (must match camera).
    pub near_plane: f32,
    /// Far clip plane distance (must match camera).
    pub far_plane: f32,
}

impl Default for DofSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            focus_distance: 10.0,
            aperture: 0.05,
            focal_length: 0.05,
            max_blur_radius: 20.0,
            near_plane: 0.1,
            far_plane: 1000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// GPU uniform (must match DofParams in the WGSL shader)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DofParamsGpu {
    resolution: [f32; 2],
    inv_resolution: [f32; 2],
    focus_distance: f32,
    aperture: f32,
    focal_length: f32,
    max_blur_radius: f32,
    near_far: [f32; 2],
    _pad: [f32; 2],
}

// ---------------------------------------------------------------------------
// DofPass
// ---------------------------------------------------------------------------

/// Manages the compute pipelines and textures for depth-of-field rendering.
///
/// Generic over [`euca_rhi::RenderDevice`] — defaults to [`WgpuDevice`] for
/// backward compatibility.
pub struct DofPass<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    coc_pipeline: D::ComputePipeline,
    gather_pipeline: D::ComputePipeline,
    bind_group_layout: D::BindGroupLayout,
    uniform_buffer: D::Buffer,
    /// Per-pixel circle-of-confusion (R16Float).
    coc_texture: D::Texture,
    coc_view: D::TextureView,
    /// Output blurred color (Rgba16Float, full resolution).
    output_texture: D::Texture,
    output_view: D::TextureView,
    width: u32,
    height: u32,
}

impl<D: euca_rhi::RenderDevice> DofPass<D> {
    /// Create the DoF pass for the given screen dimensions.
    pub fn new(device: &D, width: u32, height: u32) -> Self {
        let shader_module = device.create_shader(&ShaderDesc {
            label: Some("dof"),
            source: ShaderSource::Wgsl(DOF_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("dof_bgl"),
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
                // 2: depth texture
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: CoC texture (storage read_write, r16float)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: TextureFormat::R16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 4: output color (storage write, rgba16float)
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

        let coc_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            label: Some("dof_coc"),
            layout: &[&bind_group_layout],
            module: &shader_module,
            entry_point: "coc_pass",
        });

        let gather_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            label: Some("dof_gather"),
            layout: &[&bind_group_layout],
            module: &shader_module,
            entry_point: "gather_pass",
        });

        let uniform_buffer = device.create_buffer(&BufferDesc {
            label: Some("dof_params"),
            size: std::mem::size_of::<DofParamsGpu>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (coc_texture, coc_view) = Self::create_coc_texture(device, width, height);
        let (output_texture, output_view) = Self::create_output_texture(device, width, height);

        Self {
            coc_pipeline,
            gather_pipeline,
            bind_group_layout,
            uniform_buffer,
            coc_texture,
            coc_view,
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

        let (coc_tex, coc_v) = Self::create_coc_texture(device, width, height);
        self.coc_texture = coc_tex;
        self.coc_view = coc_v;

        let (out_tex, out_v) = Self::create_output_texture(device, width, height);
        self.output_texture = out_tex;
        self.output_view = out_v;
    }

    /// Returns a reference to the output texture.
    pub fn output_texture(&self) -> &D::Texture {
        &self.output_texture
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn create_coc_texture(device: &D, width: u32, height: u32) -> (D::Texture, D::TextureView) {
        let tex = device.create_texture(&TextureDesc {
            label: Some("dof_coc"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = device.create_texture_view(&tex, &TextureViewDesc::default());
        (tex, view)
    }

    fn create_output_texture(device: &D, width: u32, height: u32) -> (D::Texture, D::TextureView) {
        let tex = device.create_texture(&TextureDesc {
            label: Some("dof_output"),
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
}

// Keep the wgpu-specific execute/accessor methods in a separate impl block
// since the execute params and command encoding are not yet abstracted.
impl DofPass {
    /// Returns the output texture view (DoF-blurred result).
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// Returns the CoC texture view (for debug visualization).
    pub fn coc_view(&self) -> &wgpu::TextureView {
        &self.coc_view
    }

    /// Execute the DoF pass (CoC computation + gather blur).
    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        settings: &DofSettings,
    ) {
        let params = DofParamsGpu {
            resolution: [self.width as f32, self.height as f32],
            inv_resolution: [1.0 / self.width as f32, 1.0 / self.height as f32],
            focus_distance: settings.focus_distance,
            aperture: settings.aperture,
            focal_length: settings.focal_length,
            max_blur_radius: settings.max_blur_radius,
            near_far: [settings.near_plane, settings.far_plane],
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.coc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
            ],
        });

        // Pass 1: CoC computation
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dof_coc"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.coc_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // Pass 2: Gather blur
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dof_gather"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gather_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions (testable without GPU)
// ---------------------------------------------------------------------------

/// Compute the thin-lens circle-of-confusion in world-space units.
///
/// Returns a signed value: positive for background blur, negative for foreground blur.
pub fn compute_coc(focus_distance: f32, pixel_depth: f32, aperture: f32, focal_length: f32) -> f32 {
    let inv_focus = 1.0 / focus_distance.max(0.001);
    let inv_depth = 1.0 / pixel_depth.max(0.001);
    (inv_focus - inv_depth) * aperture * focal_length
}

/// Linearise a reverse-Z depth value given near and far planes.
///
/// Assumes reverse-Z: near maps to 1.0, far maps to 0.0.
pub fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    near * far / (near + depth * (far - near))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_defaults() {
        let s = DofSettings::default();
        assert!(!s.enabled, "DoF should be disabled by default");
        assert!(
            (s.focus_distance - 10.0).abs() < 1e-6,
            "Default focus distance should be 10.0"
        );
        assert!(
            (s.aperture - 0.05).abs() < 1e-6,
            "Default aperture should be 0.05"
        );
        assert!(
            (s.focal_length - 0.05).abs() < 1e-6,
            "Default focal length should be 0.05"
        );
        assert!(
            (s.max_blur_radius - 20.0).abs() < 1e-6,
            "Default max blur radius should be 20.0"
        );
        assert!(
            (s.near_plane - 0.1).abs() < 1e-6,
            "Default near plane should be 0.1"
        );
        assert!(
            (s.far_plane - 1000.0).abs() < 1e-6,
            "Default far plane should be 1000.0"
        );
    }

    #[test]
    fn settings_clone_is_independent() {
        let original = DofSettings {
            focus_distance: 5.0,
            aperture: 0.1,
            ..DofSettings::default()
        };
        let mut cloned = original.clone();
        cloned.focus_distance = 20.0;
        cloned.aperture = 0.02;

        assert!(
            (original.focus_distance - 5.0).abs() < 1e-8,
            "Original should be unchanged"
        );
        assert!(
            (original.aperture - 0.1).abs() < 1e-8,
            "Original aperture should be unchanged"
        );
    }

    #[test]
    fn coc_at_focus_is_zero() {
        let coc = compute_coc(10.0, 10.0, 0.05, 0.05);
        assert!(
            coc.abs() < 1e-6,
            "CoC at the focus distance should be zero, got {coc}"
        );
    }

    #[test]
    fn coc_background_is_positive() {
        // Object behind focus plane produces positive CoC (background blur).
        let coc = compute_coc(10.0, 50.0, 0.05, 0.05);
        assert!(
            coc > 0.0,
            "Background blur CoC should be positive, got {coc}"
        );
    }

    #[test]
    fn coc_foreground_is_negative() {
        // Object in front of focus plane produces negative CoC (foreground blur).
        let coc = compute_coc(10.0, 3.0, 0.05, 0.05);
        assert!(
            coc < 0.0,
            "Foreground blur CoC should be negative, got {coc}"
        );
    }

    #[test]
    fn coc_increases_with_distance_from_focus() {
        let aperture = 0.05;
        let focal = 0.05;
        let focus = 10.0;

        let coc_near = compute_coc(focus, 20.0, aperture, focal).abs();
        let coc_far = compute_coc(focus, 100.0, aperture, focal).abs();
        assert!(
            coc_far > coc_near,
            "More distant objects should have larger CoC"
        );
    }

    #[test]
    fn coc_increases_with_aperture() {
        let coc_small = compute_coc(10.0, 50.0, 0.01, 0.05).abs();
        let coc_large = compute_coc(10.0, 50.0, 0.1, 0.05).abs();
        assert!(
            coc_large > coc_small,
            "Larger aperture should produce larger CoC"
        );
    }

    #[test]
    fn linearize_depth_at_near() {
        // At reverse-Z depth=1.0, the result should equal near.
        let result = linearize_depth(1.0, 0.1, 1000.0);
        assert!(
            (result - 0.1).abs() < 1e-4,
            "Depth 1.0 should linearize to near plane, got {result}"
        );
    }

    #[test]
    fn linearize_depth_at_far() {
        // At reverse-Z depth=0.0, the result should equal far.
        let result = linearize_depth(0.0, 0.1, 1000.0);
        assert!(
            (result - 1000.0).abs() < 1e-2,
            "Depth 0.0 should linearize to far plane, got {result}"
        );
    }

    #[test]
    fn linearize_depth_monotonic() {
        let near = 0.1_f32;
        let far = 1000.0_f32;
        // In reverse-Z, smaller depth values correspond to farther distances.
        let d_near = linearize_depth(0.9, near, far);
        let d_mid = linearize_depth(0.5, near, far);
        let d_far = linearize_depth(0.1, near, far);
        assert!(
            d_near < d_mid && d_mid < d_far,
            "Linearized depth should be monotonically increasing as raw depth decreases"
        );
    }

    #[test]
    fn dof_params_gpu_alignment() {
        let size = std::mem::size_of::<DofParamsGpu>();
        assert_eq!(
            size % 16,
            0,
            "DofParamsGpu ({size} bytes) must be 16-byte aligned"
        );
    }

    #[test]
    fn shader_source_valid() {
        assert!(!DOF_SHADER.is_empty());
        assert!(DOF_SHADER.contains("@compute"));
        assert!(DOF_SHADER.contains("fn coc_pass"));
        assert!(DOF_SHADER.contains("fn gather_pass"));
        assert!(DOF_SHADER.contains("DofParams"));
        assert!(DOF_SHADER.contains("POISSON_DISK"));
    }
}
