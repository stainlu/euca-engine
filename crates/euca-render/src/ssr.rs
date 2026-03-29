//! Screen-Space Reflections (SSR) post-process pass.
//!
//! # Overview
//! For each pixel, if the surface is metallic and roughness falls below a
//! configurable threshold, the pass reflects the view ray using the surface
//! normal and ray-marches in screen space to find an intersection with the
//! depth buffer. On hit, it samples the color buffer at the hit point; on
//! miss, it falls back to a subtle sky/ambient tint.
//!
//! # Inputs
//! - Depth buffer (R32Float or equivalent resolved depth)
//! - Normal + material G-buffer (RT1: octahedral normal xy, metallic z, roughness w)
//! - Color buffer (HDR scene after lighting, before tonemapping)
//!
//! # Output
//! - Reflection overlay texture (Rgba16Float) with premultiplied alpha.
//!   The caller composites this over the scene using alpha blending.

use euca_rhi::pass::RenderPassOps;
use euca_rhi::{
    BindGroupLayoutDesc, BindGroupLayoutEntry, BindingType, BlendState, BufferBindingType,
    BufferDesc, BufferUsages, ColorTargetState, ColorWrites, Extent3d, FilterMode, FragmentState,
    PrimitiveState, RenderPipelineDesc, SamplerBindingType, SamplerDesc, ShaderDesc, ShaderSource,
    ShaderStages, TextureDesc, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureViewDesc, TextureViewDimension, VertexState,
};

/// Runtime settings for screen-space reflections.
#[derive(Clone, Debug)]
pub struct SsrSettings {
    /// Master toggle for the SSR pass.
    pub enabled: bool,
    /// Maximum number of ray-march steps per pixel (higher = more accurate, slower).
    pub max_steps: u32,
    /// Distance to advance per step in view space.
    pub step_size: f32,
    /// Maximum ray-march distance in view-space units before giving up.
    pub max_distance: f32,
    /// Depth tolerance for intersection detection (view-space units).
    pub thickness: f32,
    /// Surfaces with roughness >= this value are excluded from SSR.
    pub roughness_threshold: f32,
}

impl Default for SsrSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_steps: 64,
            step_size: 0.1,
            max_distance: 50.0,
            thickness: 0.5,
            roughness_threshold: 0.5,
        }
    }
}

/// GPU-side uniform layout matching `SsrUniforms` in `ssr.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SsrUniforms {
    /// Inverse projection matrix for reconstructing view-space position from depth.
    pub inv_projection: [[f32; 4]; 4],
    /// Projection matrix for projecting view-space positions back to screen.
    pub projection: [[f32; 4]; 4],
    /// x = max_steps, y = step_size, z = max_distance, w = thickness
    pub params0: [f32; 4],
    /// x = roughness_threshold, y = screen_width, z = screen_height, w = unused
    pub params1: [f32; 4],
}

impl SsrUniforms {
    pub fn new(
        settings: &SsrSettings,
        inv_projection: &[[f32; 4]; 4],
        projection: &[[f32; 4]; 4],
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            inv_projection: *inv_projection,
            projection: *projection,
            params0: [
                settings.max_steps as f32,
                settings.step_size,
                settings.max_distance,
                settings.thickness,
            ],
            params1: [
                settings.roughness_threshold,
                width as f32,
                height as f32,
                0.0,
            ],
        }
    }
}

/// Parameters for a single SSR pass execution.
///
/// Groups the input textures, matrices, and settings that `SsrPass::execute`
/// needs, keeping the function signature clean and extensible.
pub struct SsrExecuteParams<'a, D: euca_rhi::RenderDevice> {
    pub rhi: &'a D,
    pub encoder: &'a mut D::CommandEncoder,
    /// Resolved single-sample depth (R32Float).
    pub depth_view: &'a D::TextureView,
    /// G-buffer RT1 (octahedral normal + metallic + roughness).
    pub normal_material_view: &'a D::TextureView,
    /// HDR scene color before tonemapping.
    pub color_view: &'a D::TextureView,
    pub settings: &'a SsrSettings,
    /// Inverse of the camera projection matrix (4x4 column-major).
    pub inv_projection: &'a [[f32; 4]; 4],
    /// Camera projection matrix (4x4 column-major).
    pub projection: &'a [[f32; 4]; 4],
}

/// Manages the GPU pipeline and resources for the SSR post-process pass.
pub struct SsrPass<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pipeline: D::RenderPipeline,
    bind_group_layout: D::BindGroupLayout,
    uniform_buffer: D::Buffer,
    sampler: D::Sampler,
    #[allow(dead_code)]
    output_texture: D::Texture,
    output_view: D::TextureView,
    width: u32,
    height: u32,
}

const SSR_OUTPUT_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

impl<D: euca_rhi::RenderDevice> SsrPass<D> {
    /// Create a new SSR pass for the given surface dimensions.
    pub fn new(device: &D, width: u32, height: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);

        let sampler = device.create_sampler(&SamplerDesc {
            label: Some("SSR Sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer(&BufferDesc {
            label: Some("SSR Uniforms"),
            size: std::mem::size_of::<SsrUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("SSR BGL"),
            entries: &[
                // binding 0: depth texture
                bgl_texture_entry(0),
                // binding 1: normal+material G-buffer
                bgl_texture_entry(1),
                // binding 2: color texture
                bgl_texture_entry(2),
                // binding 3: sampler
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 4: uniforms
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::mem::size_of::<SsrUniforms>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let pipeline = create_ssr_pipeline(device, &bind_group_layout);
        let (output_texture, output_view) = create_ssr_target(device, width, height);

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            sampler,
            output_texture,
            output_view,
            width,
            height,
        }
    }

    /// Recreate resolution-dependent resources after a window resize.
    pub fn resize(&mut self, device: &D, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.width = width;
        self.height = height;
        let (output_texture, output_view) = create_ssr_target(device, width, height);
        self.output_texture = output_texture;
        self.output_view = output_view;
    }

    /// Current width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Current height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Execute the SSR pass: dispatch a fullscreen triangle that ray-marches
    /// reflections and writes the result into the internal overlay texture.
    pub fn execute(&self, params: SsrExecuteParams<'_, D>) {
        if !params.settings.enabled {
            return;
        }

        let uniforms = SsrUniforms::new(
            params.settings,
            params.inv_projection,
            params.projection,
            self.width,
            self.height,
        );
        params
            .rhi
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let bind_group = params.rhi.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("SSR BG"),
            layout: &self.bind_group_layout,
            entries: &[
                euca_rhi::BindGroupEntry {
                    binding: 0,
                    resource: euca_rhi::BindingResource::TextureView(params.depth_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 1,
                    resource: euca_rhi::BindingResource::TextureView(params.normal_material_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 2,
                    resource: euca_rhi::BindingResource::TextureView(params.color_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 3,
                    resource: euca_rhi::BindingResource::Sampler(&self.sampler),
                },
                euca_rhi::BindGroupEntry {
                    binding: 4,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let mut pass = params.rhi.begin_render_pass(
            params.encoder,
            &euca_rhi::RenderPassDesc {
                label: Some("SSR Pass"),
                color_attachments: &[Some(euca_rhi::RenderPassColorAttachment {
                    view: &self.output_view,
                    resolve_target: None,
                    ops: euca_rhi::Operations {
                        load: euca_rhi::LoadOp::Clear(euca_rhi::Color::TRANSPARENT),
                        store: euca_rhi::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
            },
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// The texture view containing the SSR reflection overlay.
    /// Composite this over the scene with alpha blending.
    pub fn output_view(&self) -> &D::TextureView {
        &self.output_view
    }
}

// ────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────

fn bgl_texture_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn create_ssr_target<D: euca_rhi::RenderDevice>(
    device: &D,
    width: u32,
    height: u32,
) -> (D::Texture, D::TextureView) {
    let texture = device.create_texture(&TextureDesc {
        label: Some("SSR Output"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: SSR_OUTPUT_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = device.create_texture_view(&texture, &TextureViewDesc::default());
    (texture, view)
}

fn create_ssr_pipeline<D: euca_rhi::RenderDevice>(
    device: &D,
    bgl: &D::BindGroupLayout,
) -> D::RenderPipeline {
    let shader_source = include_str!("../shaders/ssr.wgsl");
    let shader = device.create_shader(&ShaderDesc {
        label: Some("SSR Shader"),
        source: ShaderSource::Wgsl(shader_source.into()),
    });
    device.create_render_pipeline(&RenderPipelineDesc {
        label: Some("SSR Pipeline"),
        layout: &[bgl],
        vertex: VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(ColorTargetState {
                format: SSR_OUTPUT_FORMAT,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
    })
}

// ════════════════════════════════════════════════════════════════════════
// Public helpers: CPU-side utilities for SSR configuration
// ════════════════════════════════════════════════════════════════════════

/// Compute the number of steps needed to cover the full march distance
/// at the configured step size. Useful for performance budgeting.
pub fn compute_step_count(settings: &SsrSettings) -> u32 {
    if settings.step_size <= 0.0 {
        return 0;
    }
    let needed = (settings.max_distance / settings.step_size).ceil() as u32;
    needed.min(settings.max_steps)
}

/// Returns `true` if a surface with the given roughness and metallic values
/// would be processed by the SSR pass (i.e., not filtered out).
pub fn passes_roughness_filter(roughness: f32, metallic: f32, settings: &SsrSettings) -> bool {
    metallic >= 0.01 && roughness < settings.roughness_threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_values() {
        let s = SsrSettings::default();
        assert!(s.enabled);
        assert_eq!(s.max_steps, 64);
        assert!((s.step_size - 0.1).abs() < f32::EPSILON);
        assert!((s.max_distance - 50.0).abs() < f32::EPSILON);
        assert!((s.thickness - 0.5).abs() < f32::EPSILON);
        assert!((s.roughness_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn step_count_calculation() {
        let settings = SsrSettings::default(); // 50.0 / 0.1 = 500, clamped to 64
        assert_eq!(compute_step_count(&settings), 64);

        let short = SsrSettings {
            max_distance: 5.0,
            step_size: 0.1,
            max_steps: 100,
            ..Default::default()
        };
        assert_eq!(compute_step_count(&short), 50); // 5.0 / 0.1 = 50 < 100

        let zero_step = SsrSettings {
            step_size: 0.0,
            ..Default::default()
        };
        assert_eq!(compute_step_count(&zero_step), 0);

        let negative_step = SsrSettings {
            step_size: -1.0,
            ..Default::default()
        };
        assert_eq!(compute_step_count(&negative_step), 0);
    }

    #[test]
    fn roughness_filter() {
        let settings = SsrSettings::default(); // threshold = 0.5

        // Smooth metal: should pass
        assert!(passes_roughness_filter(0.1, 1.0, &settings));
        // Rough metal: should not pass
        assert!(!passes_roughness_filter(0.5, 1.0, &settings));
        assert!(!passes_roughness_filter(0.8, 1.0, &settings));
        // Smooth but not metallic: should not pass
        assert!(!passes_roughness_filter(0.1, 0.0, &settings));
        assert!(!passes_roughness_filter(0.1, 0.005, &settings));
        // Barely metallic, smooth: should pass
        assert!(passes_roughness_filter(0.0, 0.01, &settings));
        // Edge case: roughness exactly at threshold
        assert!(!passes_roughness_filter(0.5, 1.0, &settings));
        // Just below threshold
        assert!(passes_roughness_filter(0.499, 1.0, &settings));
    }

    #[test]
    fn uniforms_encode_correctly() {
        let settings = SsrSettings {
            enabled: true,
            max_steps: 128,
            step_size: 0.05,
            max_distance: 100.0,
            thickness: 1.0,
            roughness_threshold: 0.3,
        };
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let u = SsrUniforms::new(&settings, &identity, &identity, 1920, 1080);

        assert_eq!(u.params0[0], 128.0);
        assert!((u.params0[1] - 0.05).abs() < f32::EPSILON);
        assert!((u.params0[2] - 100.0).abs() < f32::EPSILON);
        assert!((u.params0[3] - 1.0).abs() < f32::EPSILON);
        assert!((u.params1[0] - 0.3).abs() < f32::EPSILON);
        assert_eq!(u.params1[1], 1920.0);
        assert_eq!(u.params1[2], 1080.0);
        assert_eq!(u.params1[3], 0.0);
    }

    #[test]
    fn uniforms_size_is_gpu_aligned() {
        // Must be a multiple of 16 bytes for GPU uniform buffers.
        let size = std::mem::size_of::<SsrUniforms>();
        assert_eq!(
            size % 16,
            0,
            "SsrUniforms size ({size}) must be 16-byte aligned"
        );
        // Two mat4x4 (128 bytes) + two vec4 (32 bytes) = 160 bytes
        assert_eq!(size, 160);
    }
}
