//! Volumetric fog and god-ray rendering.
//!
//! # Architecture
//! A compute shader ray-marches through an exponential height-fog volume for
//! every screen pixel, accumulating extinction and in-scattering from the
//! primary directional light. The result is an Rgba16Float texture (rgb = fog
//! colour, a = opacity) that is composited over the scene colour with alpha
//! blending.
//!
//! # Usage
//! 1. Store a [`VolumetricFogSettings`] as an ECS resource.
//! 2. Create a [`VolumetricFogPass`] once during renderer initialisation.
//! 3. Each frame, call [`VolumetricFogPass::execute`] with the current camera
//!    and light state. It returns a `&wgpu::TextureView` for compositing.

use euca_rhi::RenderDevice;
use euca_rhi::wgpu_backend::WgpuDevice;

use crate::compute::{ComputePipeline, ComputePipelineDesc};

// ---------------------------------------------------------------------------
// Settings (ECS resource)
// ---------------------------------------------------------------------------

/// Configuration for volumetric fog / god rays.
///
/// Intended to be stored as a shared ECS resource and read each frame.
#[derive(Clone, Debug)]
pub struct VolumetricFogSettings {
    /// Master switch.
    pub enabled: bool,
    /// Base fog density (higher = thicker fog).
    pub density: f32,
    /// Scattering coefficient — how much light is redirected toward the camera.
    pub scattering: f32,
    /// Absorption coefficient — how much light is absorbed (removed) by the fog.
    pub absorption: f32,
    /// Rate of density decrease with height (exponential falloff).
    pub height_falloff: f32,
    /// Maximum ray-march distance from the camera.
    pub max_distance: f32,
    /// Fog tint colour (linear RGB).
    pub color: [f32; 3],
    /// God-ray strength (scales directional light contribution).
    pub light_contribution: f32,
}

impl Default for VolumetricFogSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            density: 0.02,
            scattering: 0.5,
            absorption: 0.1,
            height_falloff: 0.1,
            max_distance: 100.0,
            color: [1.0, 1.0, 1.0],
            light_contribution: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// GPU uniform (must match FogParams in the WGSL shader)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FogParams {
    camera_pos: [f32; 4],
    inv_vp: [[f32; 4]; 4],
    light_direction: [f32; 4],
    light_color: [f32; 4],
    fog_color: [f32; 4],   // xyz = tint, w = light_contribution
    fog_params: [f32; 4],  // x = density, y = scattering, z = absorption, w = height_falloff
    fog_params2: [f32; 4], // x = max_distance, y = screen_width, z = screen_height, w = 0
}

/// Shader source embedded at compile time.
pub const VOLUMETRIC_FOG_SHADER: &str = include_str!("../shaders/volumetric_fog.wgsl");

/// Compositing fragment shader: alpha-blends fog texture over the scene.
const COMPOSITE_SHADER: &str = r"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    let x = f32(i32(id) / 2) * 4.0 - 1.0;
    let y = f32(i32(id) % 2) * 4.0 - 1.0;
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}

@group(0) @binding(0) var fog_tex: texture_2d<f32>;
@group(0) @binding(1) var fog_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let fog = textureSample(fog_tex, fog_sampler, in.uv);
    // Pre-multiplied alpha output -- blended via pipeline blend state.
    return fog;
}
";

// ---------------------------------------------------------------------------
// Frame parameters
// ---------------------------------------------------------------------------

/// Parameters for a single frame's fog execution.
pub struct FrameParams<'a> {
    pub camera_pos: [f32; 3],
    pub inv_vp: [[f32; 4]; 4],
    pub light_direction: [f32; 3],
    pub light_color: [f32; 3],
    pub settings: &'a VolumetricFogSettings,
}

// ---------------------------------------------------------------------------
// VolumetricFogPass
// ---------------------------------------------------------------------------

/// Manages the compute pipeline, output texture, and compositing pipeline for
/// volumetric fog rendering.
///
/// Generic over [`RenderDevice`] — defaults to [`WgpuDevice`] for
/// backward compatibility.
pub struct VolumetricFogPass<D: RenderDevice = WgpuDevice> {
    compute_pipeline: ComputePipeline,
    #[allow(dead_code)]
    fog_texture: D::Texture,
    fog_texture_view: D::TextureView,
    /// A second view with `Filterable` access for sampling in the composite pass.
    fog_texture_sample_view: D::TextureView,
    uniform_buffer: D::Buffer,
    bind_group: D::BindGroup,
    composite_pipeline: D::RenderPipeline,
    composite_bgl: D::BindGroupLayout,
    composite_sampler: D::Sampler,
    width: u32,
    height: u32,
}

impl VolumetricFogPass {
    /// Create the pass for the given screen dimensions.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        // --- Compute pipeline --------------------------------------------------
        let compute_pipeline = ComputePipeline::new(
            device,
            &ComputePipelineDesc {
                label: "volumetric_fog_compute",
                shader_source: VOLUMETRIC_FOG_SHADER,
                entry_point: "main",
            },
        );

        // --- Fog output texture ------------------------------------------------
        let (fog_texture, fog_texture_view, fog_texture_sample_view) =
            Self::create_fog_texture(device, width, height);

        // --- Uniform buffer ----------------------------------------------------
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volumetric_fog_params"),
            size: std::mem::size_of::<FogParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Bind group (group 0) for compute ----------------------------------
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_fog_bg"),
            layout: compute_pipeline.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&fog_texture_view),
                },
            ],
        });

        // --- Composite render pipeline -----------------------------------------
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_fog_composite"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
        });

        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetric_fog_composite_bgl"),
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

        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetric_fog_composite_layout"),
            bind_group_layouts: &[&composite_bgl],
            push_constant_ranges: &[],
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volumetric_fog_composite"),
            layout: Some(&composite_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volumetric_fog_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            compute_pipeline,
            fog_texture,
            fog_texture_view,
            fog_texture_sample_view,
            uniform_buffer,
            bind_group,
            composite_pipeline,
            composite_bgl,
            composite_sampler,
            width,
            height,
        }
    }

    /// Recreate the fog texture and bind group after a window resize.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;

        let (tex, view, sample_view) = Self::create_fog_texture(device, width, height);
        self.fog_texture = tex;
        self.fog_texture_view = view;
        self.fog_texture_sample_view = sample_view;

        // Re-create the compute bind group with the new texture view.
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_fog_bg"),
            layout: self.compute_pipeline.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.fog_texture_view),
                },
            ],
        });
    }

    /// Dispatch the volumetric fog compute shader and composite it over the
    /// scene colour target.
    ///
    /// Returns the fog `TextureView` so callers can also use it for other
    /// compositing strategies if desired.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        queue: &wgpu::Queue,
        frame: &FrameParams<'_>,
    ) -> &wgpu::TextureView {
        // Upload uniforms.
        let uniforms = FogParams {
            camera_pos: [
                frame.camera_pos[0],
                frame.camera_pos[1],
                frame.camera_pos[2],
                0.0,
            ],
            inv_vp: frame.inv_vp,
            light_direction: [
                frame.light_direction[0],
                frame.light_direction[1],
                frame.light_direction[2],
                0.0,
            ],
            light_color: [
                frame.light_color[0],
                frame.light_color[1],
                frame.light_color[2],
                0.0,
            ],
            fog_color: [
                frame.settings.color[0],
                frame.settings.color[1],
                frame.settings.color[2],
                frame.settings.light_contribution,
            ],
            fog_params: [
                frame.settings.density,
                frame.settings.scattering,
                frame.settings.absorption,
                frame.settings.height_falloff,
            ],
            fog_params2: [
                frame.settings.max_distance,
                self.width as f32,
                self.height as f32,
                0.0,
            ],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // --- Compute dispatch --------------------------------------------------
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("volumetric_fog_compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.compute_pipeline.raw());
            pass.set_bind_group(0, Some(&self.bind_group), &[]);
            let wg_x = self.width.div_ceil(8);
            let wg_y = self.height.div_ceil(8);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // --- Composite pass (alpha-blend fog over scene) ----------------------
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric_fog_composite_bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.fog_texture_sample_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volumetric_fog_composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, Some(&composite_bg), &[]);
            pass.draw(0..3, 0..1);
        }

        &self.fog_texture_sample_view
    }

    /// The fog texture view for external compositing (sampling/filtering).
    ///
    /// Returns `fog_texture_sample_view` (the filterable view) rather than
    /// the identically-named field `fog_texture_view` (the storage/render-target
    /// view), because external consumers always need the sampling view.
    #[allow(clippy::misnamed_getters)]
    pub fn fog_texture_view(&self) -> &wgpu::TextureView {
        &self.fog_texture_sample_view
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn create_fog_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volumetric_fog_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Storage view for the compute shader (write).
        let storage_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Sampling view for the composite fragment shader (read).
        let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("volumetric_fog_sample_view"),
            ..Default::default()
        });

        (texture, storage_view, sample_view)
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions (testable without GPU)
// ---------------------------------------------------------------------------

/// Compute exponential height-based fog density at a given height.
///
/// `base_density * exp(-height_falloff * max(height, 0.0))`
pub fn fog_density_at_height(base_density: f32, height_falloff: f32, height: f32) -> f32 {
    base_density * (-height_falloff * height.max(0.0)).exp()
}

/// Compute the Henyey-Greenstein phase function value.
///
/// `g` controls the scattering asymmetry: 0 = isotropic, >0 = forward scattering.
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    (1.0 - g2) / (4.0 * std::f32::consts::PI * denom.powf(1.5))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_defaults() {
        let s = VolumetricFogSettings::default();
        assert!(s.enabled);
        assert!((s.density - 0.02).abs() < 1e-6);
        assert!((s.scattering - 0.5).abs() < 1e-6);
        assert!((s.absorption - 0.1).abs() < 1e-6);
        assert!((s.height_falloff - 0.1).abs() < 1e-6);
        assert!((s.max_distance - 100.0).abs() < 1e-6);
        assert_eq!(s.color, [1.0, 1.0, 1.0]);
        assert!((s.light_contribution - 1.0).abs() < 1e-6);
    }

    #[test]
    fn density_at_height_calculation() {
        let base = 0.02;
        let falloff = 0.1;

        // At height 0 the density equals the base density.
        let d0 = fog_density_at_height(base, falloff, 0.0);
        assert!((d0 - base).abs() < 1e-8);

        // Density decreases with height.
        let d10 = fog_density_at_height(base, falloff, 10.0);
        assert!(d10 < d0);
        let expected = base * (-falloff * 10.0_f32).exp();
        assert!((d10 - expected).abs() < 1e-8);

        // Negative heights are clamped to 0 (underwater stays at base density).
        let d_neg = fog_density_at_height(base, falloff, -5.0);
        assert!((d_neg - base).abs() < 1e-8);
    }

    #[test]
    fn height_falloff_steeper_means_faster_decay() {
        let base = 1.0;
        let height = 5.0;

        let slow = fog_density_at_height(base, 0.1, height);
        let fast = fog_density_at_height(base, 1.0, height);
        assert!(
            fast < slow,
            "Steeper falloff should produce lower density at the same height"
        );
    }

    #[test]
    fn henyey_greenstein_isotropic() {
        // g = 0 -> isotropic: phase function is constant = 1 / (4 * PI).
        let iso = 1.0 / (4.0 * std::f32::consts::PI);
        let val_fwd = henyey_greenstein(1.0, 0.0);
        let val_side = henyey_greenstein(0.0, 0.0);
        let val_back = henyey_greenstein(-1.0, 0.0);
        assert!((val_fwd - iso).abs() < 1e-6);
        assert!((val_side - iso).abs() < 1e-6);
        assert!((val_back - iso).abs() < 1e-6);
    }

    #[test]
    fn henyey_greenstein_forward_bias() {
        // For g > 0 the forward direction (cos_theta = 1) should be brightest.
        let g = 0.7;
        let fwd = henyey_greenstein(1.0, g);
        let side = henyey_greenstein(0.0, g);
        let back = henyey_greenstein(-1.0, g);
        assert!(fwd > side);
        assert!(side > back);
    }

    #[test]
    fn fog_params_uniform_size() {
        // Ensure the GPU struct size matches our expectations (must be 16-byte aligned).
        let size = std::mem::size_of::<FogParams>();
        assert_eq!(
            size % 16,
            0,
            "FogParams must be 16-byte aligned for uniform buffers"
        );
    }

    #[test]
    fn shader_source_is_valid() {
        assert!(!VOLUMETRIC_FOG_SHADER.is_empty());
        assert!(VOLUMETRIC_FOG_SHADER.contains("@compute"));
        assert!(VOLUMETRIC_FOG_SHADER.contains("@workgroup_size(8, 8)"));
        assert!(VOLUMETRIC_FOG_SHADER.contains("fn main"));
        assert!(VOLUMETRIC_FOG_SHADER.contains("FogParams"));
    }

    #[test]
    fn settings_enabled_toggle() {
        let mut s = VolumetricFogSettings::default();
        assert!(s.enabled, "Default settings should have fog enabled");

        s.enabled = false;
        assert!(!s.enabled, "Should be able to disable fog");

        s.enabled = true;
        assert!(s.enabled, "Should be able to re-enable fog");
    }

    #[test]
    fn settings_clone_is_independent() {
        let original = VolumetricFogSettings {
            density: 0.05,
            scattering: 0.8,
            height_falloff: 0.3,
            ..VolumetricFogSettings::default()
        };

        let mut cloned = original.clone();
        cloned.density = 0.1;
        cloned.scattering = 0.2;

        // Original must remain unchanged after mutating the clone.
        assert!(
            (original.density - 0.05).abs() < 1e-8,
            "Original density should be unchanged"
        );
        assert!(
            (original.scattering - 0.8).abs() < 1e-8,
            "Original scattering should be unchanged"
        );
        assert!(
            (cloned.density - 0.1).abs() < 1e-8,
            "Cloned density should be updated"
        );
    }
}
