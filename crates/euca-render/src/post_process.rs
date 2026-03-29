//! Post-processing pipeline: ordered stack of fullscreen passes.
//!
//! # Architecture
//! The `PostProcessStack` manages a sequence of GPU render passes that
//! transform the HDR scene texture into the final LDR output. Each pass
//! reads from one texture and writes to another, ping-ponging between two
//! intermediate textures to avoid extra allocations.
//!
//! # Pass order (when all effects enabled)
//! 1. SSAO — computes ambient occlusion from depth, composites into HDR
//! 2. Main — bloom + color grading + ACES tonemapping + vignette (HDR -> LDR)
//! 3. FXAA — edge-detect anti-aliasing on the final LDR image
//!
//! # Integration with renderer
//! The renderer resolves MSAA HDR into `PostProcessStack::ping_view()`.
//! After that, `PostProcessStack::execute()` runs all enabled passes and
//! writes the final result into the swapchain surface view.

/// Post-processing configuration. Intended to be stored as an ECS resource.
#[derive(Clone, Debug)]
pub struct PostProcessSettings {
    // TAA
    /// Temporal anti-aliasing (sub-pixel jitter + history accumulation).
    pub taa_enabled: bool,

    // SSAO
    pub ssao_enabled: bool,
    /// Sampling radius in view-space units (default 0.5).
    pub ssao_radius: f32,
    /// Occlusion strength multiplier (default 1.0).
    pub ssao_intensity: f32,

    // SSGI
    /// Screen-space global illumination (indirect diffuse via depth ray-march).
    pub ssgi_enabled: bool,
    /// Number of rays cast per half-res pixel (4-8 recommended).
    pub ssgi_ray_count: u32,
    /// Maximum world-space ray distance.
    pub ssgi_max_distance: f32,
    /// Indirect lighting intensity multiplier.
    pub ssgi_intensity: f32,
    /// Temporal accumulation blend factor (0 = no history, 1 = all history).
    pub ssgi_temporal_blend: f32,

    // SSR
    pub ssr_enabled: bool,
    pub ssr: crate::ssr::SsrSettings,

    // FXAA
    pub fxaa_enabled: bool,

    // Bloom (existing functionality)
    pub bloom_enabled: bool,
    pub bloom_threshold: f32,

    // Motion blur
    pub motion_blur: crate::motion_blur::MotionBlurSettings,

    // Depth of field
    pub dof: crate::dof::DofSettings,

    // Image-Based Lighting
    /// Enable IBL (environment cubemap) for indirect specular/diffuse.
    /// The renderer uses this to decide whether to activate IBL resources.
    pub ibl_enabled: bool,
    /// IBL intensity multiplier (default 1.0).
    pub ibl_intensity: f32,

    // PCSS Soft Shadows
    /// Enable Percentage-Closer Soft Shadows. When disabled the renderer
    /// sets `light_size` to 0.0, producing hard shadow edges.
    pub pcss_enabled: bool,

    // Color grading
    /// EV stops: final color *= 2^exposure (default 0.0 = no change).
    pub exposure: f32,
    /// Contrast around mid-gray (default 1.0 = no change).
    pub contrast: f32,
    /// Saturation multiplier (default 1.0 = no change, 0.0 = grayscale).
    pub saturation: f32,
    /// Color temperature shift in Kelvin offset (default 0.0 = neutral).
    /// Positive = warmer (yellow), negative = cooler (blue).
    pub temperature: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            taa_enabled: false,
            ssao_enabled: true,
            ssao_radius: 0.5,
            ssao_intensity: 1.0,
            ssgi_enabled: false,
            ssgi_ray_count: 4,
            ssgi_max_distance: 10.0,
            ssgi_intensity: 1.0,
            ssgi_temporal_blend: 0.9,
            ssr_enabled: false,
            ssr: crate::ssr::SsrSettings::default(),
            motion_blur: crate::motion_blur::MotionBlurSettings::default(),
            dof: crate::dof::DofSettings::default(),
            ibl_enabled: false,
            ibl_intensity: 1.0,
            pcss_enabled: true,
            fxaa_enabled: true,
            bloom_enabled: true,
            bloom_threshold: 0.8,
            exposure: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            temperature: 0.0,
        }
    }
}

/// GPU-side uniform for the color-grading + tonemapping pass.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PostProcessUniforms {
    /// x = exposure, y = contrast, z = saturation, w = temperature
    pub color_grade: [f32; 4],
    /// x = bloom_enabled (0/1), y = bloom_threshold, z = ssao_enabled (0/1), w = ssao_intensity
    pub flags: [f32; 4],
    /// x = fxaa_enabled (0/1), y = ssao_radius, zw = unused
    pub flags2: [f32; 4],
    /// x = screen_width, y = screen_height, zw = unused
    pub screen_size: [f32; 4],
}

impl PostProcessUniforms {
    pub fn from_settings(settings: &PostProcessSettings, width: u32, height: u32) -> Self {
        Self {
            color_grade: [
                settings.exposure,
                settings.contrast,
                settings.saturation,
                settings.temperature,
            ],
            flags: [
                if settings.bloom_enabled { 1.0 } else { 0.0 },
                settings.bloom_threshold,
                if settings.ssao_enabled { 1.0 } else { 0.0 },
                settings.ssao_intensity,
            ],
            flags2: [
                if settings.fxaa_enabled { 1.0 } else { 0.0 },
                settings.ssao_radius,
                0.0,
                0.0,
            ],
            screen_size: [width as f32, height as f32, 0.0, 0.0],
        }
    }
}

/// SSAO uniform data sent to the GPU.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoUniforms {
    /// Inverse projection matrix — used to reconstruct view-space position from depth.
    inv_projection: [[f32; 4]; 4],
    /// x = radius, y = intensity, z = screen_width, w = screen_height
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SsrNormalsUniforms {
    pub inv_projection: [[f32; 4]; 4],
    pub params: [f32; 4],
}

/// Generic over [`RenderDevice`] — defaults to [`WgpuDevice`] for backward compatibility.
pub struct PostProcessStack<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    // Ping-pong intermediate textures (HDR format, full resolution).
    ping_texture: D::Texture,
    ping_view: D::TextureView,
    pong_texture: D::Texture,
    pong_view: D::TextureView,

    // Resolved depth for SSAO (single-sample R32Float).
    #[allow(dead_code)]
    depth_resolve_texture: D::Texture,
    pub(crate) depth_resolve_view: D::TextureView,

    // SSAO resources (half-resolution)
    #[allow(dead_code)]
    ssao_texture: D::Texture,
    ssao_view: D::TextureView,
    #[allow(dead_code)]
    ssao_blur_texture: D::Texture,
    ssao_blur_view: D::TextureView,
    #[allow(dead_code)]
    ssao_noise_texture: D::Texture,
    ssao_noise_view: D::TextureView,
    ssao_pipeline: D::RenderPipeline,
    ssao_blur_pipeline: D::RenderPipeline,
    ssao_composite_pipeline: D::RenderPipeline,
    ssao_bgl: D::BindGroupLayout,
    ssao_blur_bgl: D::BindGroupLayout,
    ssao_composite_bgl: D::BindGroupLayout,
    ssao_bind_group: D::BindGroup,
    ssao_blur_bind_group: D::BindGroup,
    ssao_composite_bind_group: D::BindGroup,
    ssao_uniform_buffer: D::Buffer,

    // SSR resources
    ssr_pass: crate::ssr::SsrPass<D>,
    #[allow(dead_code)]
    ssr_normals_texture: D::Texture,
    ssr_normals_view: D::TextureView,
    ssr_normals_pipeline: D::RenderPipeline,
    ssr_normals_bgl: D::BindGroupLayout,
    ssr_normals_bind_group: D::BindGroup,
    ssr_normals_uniform_buffer: D::Buffer,
    ssr_composite_pipeline: D::RenderPipeline,
    ssr_composite_bgl: D::BindGroupLayout,

    // Main post-process (bloom + color grade + tonemap + vignette)
    main_pipeline: D::RenderPipeline,
    main_to_hdr_pipeline: D::RenderPipeline,
    main_bgl: D::BindGroupLayout,
    main_bind_group: D::BindGroup,
    uniform_buffer: D::Buffer,

    // FXAA
    fxaa_pipeline: D::RenderPipeline,
    fxaa_bgl: D::BindGroupLayout,
    fxaa_bind_group: D::BindGroup,

    // Shared samplers
    linear_sampler: D::Sampler,
    nearest_sampler: D::Sampler,

    width: u32,
    height: u32,
    #[allow(dead_code)]
    surface_format: euca_rhi::TextureFormat,
}

impl<D: euca_rhi::RenderDevice> PostProcessStack<D> {
    pub fn new(
        device: &D,
        width: u32,
        height: u32,
        surface_format: euca_rhi::TextureFormat,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);

        let linear_sampler = device.create_sampler(&euca_rhi::SamplerDesc {
            label: Some("PostProcess Linear Sampler"),
            mag_filter: euca_rhi::FilterMode::Linear,
            min_filter: euca_rhi::FilterMode::Linear,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&euca_rhi::SamplerDesc {
            label: Some("PostProcess Nearest Sampler"),
            mag_filter: euca_rhi::FilterMode::Nearest,
            min_filter: euca_rhi::FilterMode::Nearest,
            ..Default::default()
        });

        let (ping_texture, ping_view) = create_hdr_target(device, width, height, "PP Ping");
        let (pong_texture, pong_view) = create_hdr_target(device, width, height, "PP Pong");
        let (depth_resolve_texture, depth_resolve_view) =
            create_depth_resolve_target(device, width, height);

        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);
        let (ssao_texture, ssao_view) = create_r8_target(device, half_w, half_h, "SSAO Raw");
        let (ssao_blur_texture, ssao_blur_view) =
            create_r8_target(device, half_w, half_h, "SSAO Blurred");
        let (ssao_noise_texture, ssao_noise_view) = create_ssao_noise_texture(device);

        let ssao_uniform_buffer = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some("SSAO Uniforms"),
            size: std::mem::size_of::<SsaoUniforms>() as u64,
            usage: euca_rhi::BufferUsages::UNIFORM | euca_rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some("PostProcess Uniforms"),
            size: std::mem::size_of::<PostProcessUniforms>() as u64,
            usage: euca_rhi::BufferUsages::UNIFORM | euca_rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Bind group layouts ──
        let ssao_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("SSAO BGL"),
            entries: &[
                // Depth resolve texture is R32Float — NOT filterable on most GPUs
                bgl_texture_entry(0, euca_rhi::TextureSampleType::Float { filterable: false }),
                bgl_texture_entry(1, euca_rhi::TextureSampleType::Float { filterable: false }),
                // NonFiltering sampler — required when any texture is non-filterable
                euca_rhi::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Sampler(euca_rhi::SamplerBindingType::NonFiltering),
                    count: None,
                },
                bgl_uniform_entry(3, std::mem::size_of::<SsaoUniforms>() as u64),
            ],
        });

        let ssao_blur_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("SSAO Blur BGL"),
            entries: &[
                bgl_texture_entry(0, euca_rhi::TextureSampleType::Float { filterable: true }),
                bgl_sampler_entry(1),
            ],
        });

        let ssao_composite_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("SSAO Composite BGL"),
            entries: &[
                bgl_texture_entry(0, euca_rhi::TextureSampleType::Float { filterable: true }),
                bgl_texture_entry(1, euca_rhi::TextureSampleType::Float { filterable: true }),
                bgl_sampler_entry(2),
                bgl_uniform_entry(3, std::mem::size_of::<PostProcessUniforms>() as u64),
            ],
        });

        // Main + FXAA share the same layout: texture + sampler + uniforms
        let tex_sampler_uniform_bgl =
            device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
                label: Some("PP Tex+Sampler+Uniform BGL"),
                entries: &[
                    bgl_texture_entry(0, euca_rhi::TextureSampleType::Float { filterable: true }),
                    bgl_sampler_entry(1),
                    bgl_uniform_entry(2, std::mem::size_of::<PostProcessUniforms>() as u64),
                ],
            });

        // ── Pipelines ──
        let ssao_pipeline =
            create_fullscreen_pipeline(device, &ssao_bgl, &ssao_shader(), "SSAO", R8_FORMAT);
        let ssao_blur_pipeline = create_fullscreen_pipeline(
            device,
            &ssao_blur_bgl,
            &ssao_blur_shader(),
            "SSAO Blur",
            R8_FORMAT,
        );
        let ssao_composite_pipeline = create_fullscreen_pipeline(
            device,
            &ssao_composite_bgl,
            &ssao_composite_shader(),
            "SSAO Composite",
            HDR_FORMAT,
        );
        // When FXAA is on: main → HDR intermediate → FXAA → surface
        // When FXAA is off: main → surface directly
        let main_pipeline = create_fullscreen_pipeline(
            device,
            &tex_sampler_uniform_bgl,
            &main_postprocess_shader(),
            "PP Main",
            surface_format,
        );
        let main_to_hdr_pipeline = create_fullscreen_pipeline(
            device,
            &tex_sampler_uniform_bgl,
            &main_postprocess_shader(),
            "PP Main (to HDR)",
            HDR_FORMAT,
        );
        let fxaa_pipeline = create_fullscreen_pipeline(
            device,
            &tex_sampler_uniform_bgl,
            &fxaa_shader(),
            "FXAA",
            surface_format,
        );

        // ── Bind groups ──
        let ssao_bind_group = create_ssao_bind_group(
            device,
            &ssao_bgl,
            &depth_resolve_view,
            &ssao_noise_view,
            &nearest_sampler,
            &ssao_uniform_buffer,
        );

        let ssao_blur_bind_group = create_tex_sampler_bind_group(
            device,
            &ssao_blur_bgl,
            &ssao_view,
            &linear_sampler,
            "SSAO Blur BG",
        );

        let ssao_composite_bind_group = create_ssao_composite_bind_group(
            device,
            &ssao_composite_bgl,
            &ping_view,
            &ssao_blur_view,
            &linear_sampler,
            &uniform_buffer,
        );

        let main_bind_group = create_tex_sampler_uniform_bind_group(
            device,
            &tex_sampler_uniform_bgl,
            &ping_view,
            &linear_sampler,
            &uniform_buffer,
            "PP Main BG",
        );

        let fxaa_bind_group = create_tex_sampler_uniform_bind_group(
            device,
            &tex_sampler_uniform_bgl,
            &pong_view,
            &linear_sampler,
            &uniform_buffer,
            "FXAA BG",
        );

        let main_bgl = tex_sampler_uniform_bgl;

        let ssr_pass = crate::ssr::SsrPass::new(device, width, height);
        let (ssr_normals_texture, ssr_normals_view) =
            create_texture_target(device, width, height, "SSR Normals", HDR_FORMAT);
        let ssr_normals_uniform_buffer = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some("SSR Normals Uniforms"),
            size: std::mem::size_of::<SsrNormalsUniforms>() as u64,
            usage: euca_rhi::BufferUsages::UNIFORM | euca_rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ssr_normals_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("SSR Normals BGL"),
            entries: &[
                // Depth resolve is R32Float — not filterable
                bgl_texture_entry(0, euca_rhi::TextureSampleType::Float { filterable: false }),
                euca_rhi::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Sampler(euca_rhi::SamplerBindingType::NonFiltering),
                    count: None,
                },
                bgl_uniform_entry(2, std::mem::size_of::<SsrNormalsUniforms>() as u64),
            ],
        });
        let ssr_normals_pipeline = create_fullscreen_pipeline(
            device,
            &ssr_normals_bgl,
            include_str!("../shaders/ssr_normals.wgsl"),
            "SSR Normals",
            HDR_FORMAT,
        );
        let ssr_normals_bind_group = create_tex_sampler_uniform_bind_group(
            device,
            &ssr_normals_bgl,
            &depth_resolve_view,
            &nearest_sampler,
            &ssr_normals_uniform_buffer,
            "SSR Normals BG",
        );
        let ssr_composite_bgl = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("SSR Composite BGL"),
            entries: &[
                bgl_texture_entry(0, euca_rhi::TextureSampleType::Float { filterable: true }),
                bgl_texture_entry(1, euca_rhi::TextureSampleType::Float { filterable: true }),
                bgl_sampler_entry(2),
            ],
        });
        let ssr_composite_pipeline = create_fullscreen_pipeline(
            device,
            &ssr_composite_bgl,
            include_str!("../shaders/ssr_composite.wgsl"),
            "SSR Composite",
            HDR_FORMAT,
        );

        Self {
            ping_texture,
            ping_view,
            pong_texture,
            pong_view,
            depth_resolve_texture,
            depth_resolve_view,
            ssao_texture,
            ssao_view,
            ssao_blur_texture,
            ssao_blur_view,
            ssao_noise_texture,
            ssao_noise_view,
            ssao_pipeline,
            ssao_blur_pipeline,
            ssao_composite_pipeline,
            ssao_bgl,
            ssao_blur_bgl,
            ssao_composite_bgl,
            ssao_bind_group,
            ssao_blur_bind_group,
            ssao_composite_bind_group,
            ssao_uniform_buffer,
            ssr_pass,
            ssr_normals_texture,
            ssr_normals_view,
            ssr_normals_pipeline,
            ssr_normals_bgl,
            ssr_normals_bind_group,
            ssr_normals_uniform_buffer,
            ssr_composite_pipeline,
            ssr_composite_bgl,
            main_pipeline,
            main_to_hdr_pipeline,
            main_bgl,
            main_bind_group,
            uniform_buffer,
            fxaa_pipeline,
            // fxaa uses main_bgl — store a duplicate reference trick won't work
            // in Rust. We need to recreate the layout or accept the duplication.
            // Since BGL is cheap and the layout is identical, just reuse main_bgl
            // in execute() when creating dynamic bind groups.
            fxaa_bgl: device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
                label: Some("FXAA BGL"),
                entries: &[
                    bgl_texture_entry(0, euca_rhi::TextureSampleType::Float { filterable: true }),
                    bgl_sampler_entry(1),
                    bgl_uniform_entry(2, std::mem::size_of::<PostProcessUniforms>() as u64),
                ],
            }),
            fxaa_bind_group,
            linear_sampler,
            nearest_sampler,
            width,
            height,
            surface_format,
        }
    }

    /// Recreate all resolution-dependent textures and bind groups.
    pub fn resize(&mut self, device: &D, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.width = width;
        self.height = height;

        let (ping_texture, ping_view) = create_hdr_target(device, width, height, "PP Ping");
        let (pong_texture, pong_view) = create_hdr_target(device, width, height, "PP Pong");
        let (depth_resolve_texture, depth_resolve_view) =
            create_depth_resolve_target(device, width, height);

        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);
        let (ssao_texture, ssao_view) = create_r8_target(device, half_w, half_h, "SSAO Raw");
        let (ssao_blur_texture, ssao_blur_view) =
            create_r8_target(device, half_w, half_h, "SSAO Blurred");

        // Recreate bind groups with new texture views
        self.ssao_bind_group = create_ssao_bind_group(
            device,
            &self.ssao_bgl,
            &depth_resolve_view,
            &self.ssao_noise_view,
            &self.nearest_sampler,
            &self.ssao_uniform_buffer,
        );
        self.ssao_blur_bind_group = create_tex_sampler_bind_group(
            device,
            &self.ssao_blur_bgl,
            &ssao_view,
            &self.linear_sampler,
            "SSAO Blur BG",
        );
        self.ssao_composite_bind_group = create_ssao_composite_bind_group(
            device,
            &self.ssao_composite_bgl,
            &ping_view,
            &ssao_blur_view,
            &self.linear_sampler,
            &self.uniform_buffer,
        );
        self.main_bind_group = create_tex_sampler_uniform_bind_group(
            device,
            &self.main_bgl,
            &ping_view,
            &self.linear_sampler,
            &self.uniform_buffer,
            "PP Main BG",
        );
        self.fxaa_bind_group = create_tex_sampler_uniform_bind_group(
            device,
            &self.fxaa_bgl,
            &pong_view,
            &self.linear_sampler,
            &self.uniform_buffer,
            "FXAA BG",
        );

        self.ssr_pass.resize(device, width, height);
        let (ssr_normals_texture, ssr_normals_view) =
            create_texture_target(device, width, height, "SSR Normals", HDR_FORMAT);
        self.ssr_normals_bind_group = create_tex_sampler_uniform_bind_group(
            device,
            &self.ssr_normals_bgl,
            &depth_resolve_view,
            &self.nearest_sampler,
            &self.ssr_normals_uniform_buffer,
            "SSR Normals BG",
        );
        self.ssr_normals_texture = ssr_normals_texture;
        self.ssr_normals_view = ssr_normals_view;

        self.ping_texture = ping_texture;
        self.ping_view = ping_view;
        self.pong_texture = pong_texture;
        self.pong_view = pong_view;
        self.depth_resolve_texture = depth_resolve_texture;
        self.depth_resolve_view = depth_resolve_view;
        self.ssao_texture = ssao_texture;
        self.ssao_view = ssao_view;
        self.ssao_blur_texture = ssao_blur_texture;
        self.ssao_blur_view = ssao_blur_view;
    }

    /// Execute the full post-processing chain.
    ///
    /// Assumes the scene has been rendered with MSAA resolved into `self.ping_view()`,
    /// and if SSAO is enabled, `self.depth_resolve_view` contains the resolved depth.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        rhi: &D,
        encoder: &mut D::CommandEncoder,
        output_view: &D::TextureView,
        settings: &PostProcessSettings,
        inv_projection: &[[f32; 4]; 4],
        projection: &[[f32; 4]; 4],
    ) {
        let uniforms = PostProcessUniforms::from_settings(settings, self.width, self.height);
        rhi.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        if settings.ssao_enabled {
            let ssao_uniforms = SsaoUniforms {
                inv_projection: *inv_projection,
                params: [
                    settings.ssao_radius,
                    settings.ssao_intensity,
                    self.width as f32,
                    self.height as f32,
                ],
            };
            rhi.write_buffer(
                &self.ssao_uniform_buffer,
                0,
                bytemuck::bytes_of(&ssao_uniforms),
            );
        }

        if settings.ssao_enabled {
            run_fullscreen_pass(
                rhi,
                encoder,
                &self.ssao_pipeline,
                &self.ssao_bind_group,
                &self.ssao_view,
                "SSAO Generate",
            );
            run_fullscreen_pass(
                rhi,
                encoder,
                &self.ssao_blur_pipeline,
                &self.ssao_blur_bind_group,
                &self.ssao_blur_view,
                "SSAO Blur",
            );
            run_fullscreen_pass(
                rhi,
                encoder,
                &self.ssao_composite_pipeline,
                &self.ssao_composite_bind_group,
                &self.pong_view,
                "SSAO Composite",
            );
        }
        let hdr_after_ssao = if settings.ssao_enabled {
            &self.pong_view
        } else {
            &self.ping_view
        };
        let hdr_after_ssr = if settings.ssr_enabled && settings.ssr.enabled {
            let normals_uniforms = SsrNormalsUniforms {
                inv_projection: *inv_projection,
                params: [self.width as f32, self.height as f32, 0.0, 0.0],
            };
            rhi.write_buffer(
                &self.ssr_normals_uniform_buffer,
                0,
                bytemuck::bytes_of(&normals_uniforms),
            );
            run_fullscreen_pass(
                rhi,
                encoder,
                &self.ssr_normals_pipeline,
                &self.ssr_normals_bind_group,
                &self.ssr_normals_view,
                "SSR Normals",
            );
            self.ssr_pass.execute(crate::ssr::SsrExecuteParams {
                rhi,
                encoder,
                depth_view: &self.depth_resolve_view,
                normal_material_view: &self.ssr_normals_view,
                color_view: hdr_after_ssao,
                settings: &settings.ssr,
                inv_projection,
                projection,
            });
            let composite_target = if settings.ssao_enabled {
                &self.ping_view
            } else {
                &self.pong_view
            };
            let ssr_composite_bg = create_two_tex_sampler_bind_group(
                rhi,
                &self.ssr_composite_bgl,
                hdr_after_ssao,
                self.ssr_pass.output_view(),
                &self.linear_sampler,
                "SSR Composite BG",
            );
            run_fullscreen_pass(
                rhi,
                encoder,
                &self.ssr_composite_pipeline,
                &ssr_composite_bg,
                composite_target,
                "SSR Composite",
            );
            composite_target
        } else {
            hdr_after_ssao
        };
        let main_bg = create_tex_sampler_uniform_bind_group(
            rhi,
            &self.main_bgl,
            hdr_after_ssr,
            &self.linear_sampler,
            &self.uniform_buffer,
            "PP Main (dynamic)",
        );
        if settings.fxaa_enabled {
            let ldr_intermediate = if std::ptr::eq(hdr_after_ssr, &self.ping_view) {
                &self.pong_view
            } else {
                &self.ping_view
            };
            run_fullscreen_pass(
                rhi,
                encoder,
                &self.main_to_hdr_pipeline,
                &main_bg,
                ldr_intermediate,
                "PP Main (+FXAA)",
            );
            let fxaa_bg = create_tex_sampler_uniform_bind_group(
                rhi,
                &self.fxaa_bgl,
                ldr_intermediate,
                &self.linear_sampler,
                &self.uniform_buffer,
                "FXAA (dynamic)",
            );
            run_fullscreen_pass(
                rhi,
                encoder,
                &self.fxaa_pipeline,
                &fxaa_bg,
                output_view,
                "FXAA",
            );
        } else {
            run_fullscreen_pass(
                rhi,
                encoder,
                &self.main_pipeline,
                &main_bg,
                output_view,
                "PP Main",
            );
        }
    }

    /// The view the renderer should use as the MSAA resolve target.
    pub fn ping_view(&self) -> &D::TextureView {
        &self.ping_view
    }

    /// The ping texture (for copy operations, e.g., TAA output → ping).
    pub fn ping_texture(&self) -> &D::Texture {
        &self.ping_texture
    }

    /// The depth resolve texture. The renderer should resolve MSAA depth into this.
    #[allow(dead_code)]
    pub fn depth_resolve_texture(&self) -> &D::Texture {
        &self.depth_resolve_texture
    }
}

// ────────────────────────────────────────────────────────────────────────
// Bind group creation helpers (shared between new() and resize())
// ────────────────────────────────────────────────────────────────────────

fn create_ssao_bind_group<D: euca_rhi::RenderDevice>(
    device: &D,
    layout: &D::BindGroupLayout,
    depth_view: &D::TextureView,
    noise_view: &D::TextureView,
    sampler: &D::Sampler,
    uniform_buffer: &D::Buffer,
) -> D::BindGroup {
    device.create_bind_group(&euca_rhi::BindGroupDesc {
        label: Some("SSAO BG"),
        layout,
        entries: &[
            euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::TextureView(depth_view),
            },
            euca_rhi::BindGroupEntry {
                binding: 1,
                resource: euca_rhi::BindingResource::TextureView(noise_view),
            },
            euca_rhi::BindGroupEntry {
                binding: 2,
                resource: euca_rhi::BindingResource::Sampler(sampler),
            },
            euca_rhi::BindGroupEntry {
                binding: 3,
                resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                    buffer: uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    })
}

fn create_ssao_composite_bind_group<D: euca_rhi::RenderDevice>(
    device: &D,
    layout: &D::BindGroupLayout,
    hdr_view: &D::TextureView,
    ao_view: &D::TextureView,
    sampler: &D::Sampler,
    uniform_buffer: &D::Buffer,
) -> D::BindGroup {
    device.create_bind_group(&euca_rhi::BindGroupDesc {
        label: Some("SSAO Composite BG"),
        layout,
        entries: &[
            euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::TextureView(hdr_view),
            },
            euca_rhi::BindGroupEntry {
                binding: 1,
                resource: euca_rhi::BindingResource::TextureView(ao_view),
            },
            euca_rhi::BindGroupEntry {
                binding: 2,
                resource: euca_rhi::BindingResource::Sampler(sampler),
            },
            euca_rhi::BindGroupEntry {
                binding: 3,
                resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                    buffer: uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    })
}

fn create_tex_sampler_bind_group<D: euca_rhi::RenderDevice>(
    device: &D,
    layout: &D::BindGroupLayout,
    texture_view: &D::TextureView,
    sampler: &D::Sampler,
    label: &str,
) -> D::BindGroup {
    device.create_bind_group(&euca_rhi::BindGroupDesc {
        label: Some(label),
        layout,
        entries: &[
            euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::TextureView(texture_view),
            },
            euca_rhi::BindGroupEntry {
                binding: 1,
                resource: euca_rhi::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn create_tex_sampler_uniform_bind_group<D: euca_rhi::RenderDevice>(
    device: &D,
    layout: &D::BindGroupLayout,
    texture_view: &D::TextureView,
    sampler: &D::Sampler,
    uniform_buffer: &D::Buffer,
    label: &str,
) -> D::BindGroup {
    device.create_bind_group(&euca_rhi::BindGroupDesc {
        label: Some(label),
        layout,
        entries: &[
            euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::TextureView(texture_view),
            },
            euca_rhi::BindGroupEntry {
                binding: 1,
                resource: euca_rhi::BindingResource::Sampler(sampler),
            },
            euca_rhi::BindGroupEntry {
                binding: 2,
                resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                    buffer: uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    })
}

fn create_two_tex_sampler_bind_group<D: euca_rhi::RenderDevice>(
    device: &D,
    layout: &D::BindGroupLayout,
    tex0: &D::TextureView,
    tex1: &D::TextureView,
    sampler: &D::Sampler,
    label: &str,
) -> D::BindGroup {
    device.create_bind_group(&euca_rhi::BindGroupDesc {
        label: Some(label),
        layout,
        entries: &[
            euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::TextureView(tex0),
            },
            euca_rhi::BindGroupEntry {
                binding: 1,
                resource: euca_rhi::BindingResource::TextureView(tex1),
            },
            euca_rhi::BindGroupEntry {
                binding: 2,
                resource: euca_rhi::BindingResource::Sampler(sampler),
            },
        ],
    })
}

// Texture and pipeline creation helpers
// ────────────────────────────────────────────────────────────────────────

fn run_fullscreen_pass<D: euca_rhi::RenderDevice>(
    rhi: &D,
    encoder: &mut D::CommandEncoder,
    pipeline: &D::RenderPipeline,
    bind_group: &D::BindGroup,
    target: &D::TextureView,
    label: &str,
) {
    use euca_rhi::pass::RenderPassOps;

    let mut pass = rhi.begin_render_pass(
        encoder,
        &euca_rhi::RenderPassDesc {
            label: Some(label),
            color_attachments: &[Some(euca_rhi::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: euca_rhi::Operations {
                    load: euca_rhi::LoadOp::Clear(euca_rhi::Color::BLACK),
                    store: euca_rhi::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
        },
    );
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

const HDR_FORMAT: euca_rhi::TextureFormat = euca_rhi::TextureFormat::Rgba16Float;
const R8_FORMAT: euca_rhi::TextureFormat = euca_rhi::TextureFormat::R8Unorm;

fn create_hdr_target<D: euca_rhi::RenderDevice>(
    device: &D,
    width: u32,
    height: u32,
    label: &str,
) -> (D::Texture, D::TextureView) {
    create_texture_target(device, width, height, label, HDR_FORMAT)
}

fn create_r8_target<D: euca_rhi::RenderDevice>(
    device: &D,
    width: u32,
    height: u32,
    label: &str,
) -> (D::Texture, D::TextureView) {
    create_texture_target(device, width, height, label, R8_FORMAT)
}

fn create_texture_target<D: euca_rhi::RenderDevice>(
    device: &D,
    width: u32,
    height: u32,
    label: &str,
    format: euca_rhi::TextureFormat,
) -> (D::Texture, D::TextureView) {
    let texture = device.create_texture(&euca_rhi::TextureDesc {
        label: Some(label),
        size: euca_rhi::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: euca_rhi::TextureDimension::D2,
        format,
        usage: euca_rhi::TextureUsages::RENDER_ATTACHMENT
            | euca_rhi::TextureUsages::TEXTURE_BINDING
            | euca_rhi::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = device.create_texture_view(&texture, &euca_rhi::TextureViewDesc::default());
    (texture, view)
}

fn create_depth_resolve_target<D: euca_rhi::RenderDevice>(
    device: &D,
    width: u32,
    height: u32,
) -> (D::Texture, D::TextureView) {
    create_texture_target(
        device,
        width,
        height,
        "Depth Resolve",
        euca_rhi::TextureFormat::R32Float,
    )
}

fn create_ssao_noise_texture<D: euca_rhi::RenderDevice>(
    device: &D,
) -> (D::Texture, D::TextureView) {
    // 4x4 tile of random unit vectors for per-pixel rotation in SSAO kernel.
    let noise_data: [[f32; 2]; 16] = [
        [0.536, 0.844],
        [-0.731, 0.682],
        [0.954, -0.300],
        [-0.281, -0.960],
        [0.141, 0.990],
        [-0.989, 0.146],
        [0.625, -0.781],
        [-0.437, 0.899],
        [0.831, 0.556],
        [-0.556, -0.831],
        [0.300, -0.954],
        [-0.899, 0.437],
        [0.682, 0.731],
        [-0.844, -0.536],
        [0.990, 0.141],
        [-0.146, -0.989],
    ];

    let mut rgba = [0u8; 16 * 4];
    for (i, dir) in noise_data.iter().enumerate() {
        rgba[i * 4] = ((dir[0] * 0.5 + 0.5) * 255.0) as u8;
        rgba[i * 4 + 1] = ((dir[1] * 0.5 + 0.5) * 255.0) as u8;
        rgba[i * 4 + 2] = 0;
        rgba[i * 4 + 3] = 255;
    }

    let texture = device.create_texture(&euca_rhi::TextureDesc {
        label: Some("SSAO Noise 4x4"),
        size: euca_rhi::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: euca_rhi::TextureDimension::D2,
        format: euca_rhi::TextureFormat::Rgba8Unorm,
        usage: euca_rhi::TextureUsages::TEXTURE_BINDING | euca_rhi::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    device.write_texture(
        &euca_rhi::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: euca_rhi::Origin3d::default(),
            aspect: euca_rhi::TextureAspect::All,
        },
        &rgba,
        &euca_rhi::TextureDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * 4),
            rows_per_image: Some(4),
        },
        euca_rhi::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
    );

    let view = device.create_texture_view(&texture, &euca_rhi::TextureViewDesc::default());
    (texture, view)
}

fn bgl_texture_entry(
    binding: u32,
    sample_type: euca_rhi::TextureSampleType,
) -> euca_rhi::BindGroupLayoutEntry {
    euca_rhi::BindGroupLayoutEntry {
        binding,
        visibility: euca_rhi::ShaderStages::FRAGMENT,
        ty: euca_rhi::BindingType::Texture {
            sample_type,
            view_dimension: euca_rhi::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_sampler_entry(binding: u32) -> euca_rhi::BindGroupLayoutEntry {
    euca_rhi::BindGroupLayoutEntry {
        binding,
        visibility: euca_rhi::ShaderStages::FRAGMENT,
        ty: euca_rhi::BindingType::Sampler(euca_rhi::SamplerBindingType::Filtering),
        count: None,
    }
}

fn bgl_uniform_entry(binding: u32, min_size: u64) -> euca_rhi::BindGroupLayoutEntry {
    euca_rhi::BindGroupLayoutEntry {
        binding,
        visibility: euca_rhi::ShaderStages::FRAGMENT,
        ty: euca_rhi::BindingType::Buffer {
            ty: euca_rhi::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: if min_size > 0 { Some(min_size) } else { None },
        },
        count: None,
    }
}

fn create_fullscreen_pipeline<D: euca_rhi::RenderDevice>(
    device: &D,
    bgl: &D::BindGroupLayout,
    shader_source: &str,
    label: &str,
    target_format: euca_rhi::TextureFormat,
) -> D::RenderPipeline {
    let shader = device.create_shader(&euca_rhi::ShaderDesc {
        label: Some(label),
        source: euca_rhi::ShaderSource::Wgsl(shader_source.into()),
    });
    device.create_render_pipeline(&euca_rhi::RenderPipelineDesc {
        label: Some(label),
        layout: &[bgl],
        vertex: euca_rhi::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(euca_rhi::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(euca_rhi::ColorTargetState {
                format: target_format,
                blend: Some(euca_rhi::BlendState::REPLACE),
                write_mask: euca_rhi::ColorWrites::ALL,
            })],
        }),
        primitive: euca_rhi::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
    })
}

// ════════════════════════════════════════════════════════════════════════
// WGSL Shaders — shared fragments to avoid duplication
// ════════════════════════════════════════════════════════════════════════

/// Shared fullscreen triangle vertex shader + VertexOutput struct.
const FULLSCREEN_VS_WGSL: &str = include_str!("../shaders/fullscreen_vs.wgsl");

/// Shared PostProcessUniforms struct declaration (WGSL).
const PP_UNIFORMS_WGSL: &str = include_str!("../shaders/pp_uniforms.wgsl");

// ── Shader constructors (concatenate shared fragments + pass-specific code) ──

fn ssao_shader() -> String {
    // SSAO has its own vertex shader (same logic but different uniform struct)
    format!("{FULLSCREEN_VS_WGSL}
struct SsaoUniforms {{
    inv_projection: mat4x4<f32>,
    params: vec4<f32>,
}};
@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var noise_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> ssao: SsaoUniforms;
const PI: f32 = 3.14159265359;
const KERNEL_SIZE: u32 = 16u;
fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {{
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let ndc_fixed = vec4<f32>(ndc.x, -ndc.y, ndc.z, 1.0);
    let view_h = ssao.inv_projection * ndc_fixed;
    return view_h.xyz / view_h.w;
}}
fn hash(p: vec2<f32>) -> f32 {{
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{
    let screen_size = ssao.params.zw;
    let radius = ssao.params.x;
    let intensity = ssao.params.y;
    let dims = textureDimensions(depth_tex);
    let px = vec2<i32>(in.uv * vec2<f32>(dims));
    let depth = textureLoad(depth_tex, px, 0).r;
    if depth >= 1.0 {{ return vec4<f32>(1.0, 0.0, 0.0, 1.0); }}
    let view_pos = view_pos_from_depth(in.uv, depth);
    let texel = 1.0 / screen_size;
    let depth_r = textureLoad(depth_tex, px + vec2<i32>(1, 0), 0).r;
    let depth_u = textureLoad(depth_tex, px + vec2<i32>(0, 1), 0).r;
    let pos_r = view_pos_from_depth(in.uv + vec2<f32>(texel.x, 0.0), depth_r);
    let pos_u = view_pos_from_depth(in.uv + vec2<f32>(0.0, texel.y), depth_u);
    let normal = normalize(cross(pos_r - view_pos, pos_u - view_pos));
    let noise_px = vec2<i32>(in.uv * screen_size) % vec2<i32>(4, 4);
    let noise = textureLoad(noise_tex, noise_px, 0).rg * 2.0 - 1.0;
    let tangent = normalize(vec3<f32>(noise.x, noise.y, 0.0) - normal * dot(vec3<f32>(noise.x, noise.y, 0.0), normal));
    let bitangent = cross(normal, tangent);
    var occlusion = 0.0;
    for (var i = 0u; i < KERNEL_SIZE; i++) {{
        let fi = f32(i);
        let xi1 = hash(in.uv * screen_size + vec2<f32>(fi, fi * 0.7));
        let xi2 = hash(in.uv * screen_size + vec2<f32>(fi * 1.3, fi * 0.3));
        let xi3 = hash(in.uv * screen_size + vec2<f32>(fi * 0.5, fi * 1.1));
        let r_val = sqrt(xi1);
        let theta = 2.0 * PI * xi2;
        let s = vec3<f32>(r_val * cos(theta), r_val * sin(theta), sqrt(1.0 - xi1));
        let sample_dir = tangent * s.x + bitangent * s.y + normal * s.z;
        let scale = mix(0.1, 1.0, xi3 * xi3);
        let sample_pos = view_pos + sample_dir * radius * scale;
        let offset_uv = in.uv + sample_dir.xy * radius / max(abs(view_pos.z), 0.1) * 0.5;
        let sample_px = vec2<i32>(offset_uv * vec2<f32>(dims));
        let sample_depth = textureLoad(depth_tex, clamp(sample_px, vec2<i32>(0), vec2<i32>(dims) - 1), 0).r;
        let sample_view = view_pos_from_depth(offset_uv, sample_depth);
        let range_check = smoothstep(0.0, 1.0, radius / max(abs(view_pos.z - sample_view.z), 0.001));
        let is_occluded = select(0.0, 1.0, sample_view.z < sample_pos.z);
        occlusion += is_occluded * range_check;
    }}
    let ao = 1.0 - (occlusion / f32(KERNEL_SIZE)) * intensity;
    return vec4<f32>(ao, 0.0, 0.0, 1.0);
}}")
}

fn ssao_blur_shader() -> String {
    format!(
        "{FULLSCREEN_VS_WGSL}
@group(0) @binding(0) var ao_tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{
    let dims = vec2<f32>(textureDimensions(ao_tex));
    let texel = 1.0 / dims;
    let center_ao = textureSample(ao_tex, tex_sampler, in.uv).r;
    var total_ao = 0.0;
    var total_weight = 0.0;
    for (var x = -2i; x <= 2i; x++) {{
        for (var y = -2i; y <= 2i; y++) {{
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            let sample_ao = textureSample(ao_tex, tex_sampler, in.uv + offset).r;
            let spatial = exp(-f32(x * x + y * y) / 4.0);
            let range_w = exp(-abs(sample_ao - center_ao) * 8.0);
            let w = spatial * range_w;
            total_ao += sample_ao * w;
            total_weight += w;
        }}
    }}
    let result = total_ao / max(total_weight, 0.001);
    return vec4<f32>(result, 0.0, 0.0, 1.0);
}}"
    )
}

fn ssao_composite_shader() -> String {
    format!(
        "{FULLSCREEN_VS_WGSL}{PP_UNIFORMS_WGSL}
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var ao_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> uniforms: PostProcessUniforms;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{
    let hdr = textureSample(hdr_tex, tex_sampler, in.uv);
    let ao = textureSample(ao_tex, tex_sampler, in.uv).r;
    let intensity = uniforms.flags.w;
    let ao_factor = mix(1.0, ao, intensity);
    return vec4<f32>(hdr.rgb * ao_factor, hdr.a);
}}"
    )
}

fn main_postprocess_shader() -> String {
    format!("{FULLSCREEN_VS_WGSL}{PP_UNIFORMS_WGSL}
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: PostProcessUniforms;
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {{
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}}
fn bloom_sample(uv: vec2<f32>, texel: vec2<f32>, threshold: f32) -> vec3<f32> {{
    let center = textureSample(hdr_tex, tex_sampler, uv).rgb;
    let offsets = array<vec2<f32>, 12>(
        vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(-0.7, -0.7), vec2<f32>(0.7, -0.7), vec2<f32>(-0.7, 0.7), vec2<f32>(0.7, 0.7),
        vec2<f32>(-2.0, 0.0), vec2<f32>(2.0, 0.0), vec2<f32>(0.0, -2.0), vec2<f32>(0.0, 2.0),
    );
    var bloom = vec3<f32>(0.0);
    let radius = 4.0;
    for (var i = 0u; i < 12u; i++) {{
        let sample_uv = uv + offsets[i] * texel * radius;
        let s = textureSample(hdr_tex, tex_sampler, sample_uv).rgb;
        let luminance = dot(s, vec3<f32>(0.2126, 0.7152, 0.0722));
        let bright = max(luminance - threshold, 0.0) / max(luminance, 0.001);
        bloom += s * bright;
    }}
    return center + bloom * 0.08;
}}
fn color_grade(color: vec3<f32>, exposure: f32, contrast: f32, saturation: f32, temperature: f32) -> vec3<f32> {{
    var c = color * pow(2.0, exposure);
    c = (c - 0.18) * contrast + 0.18;
    c = max(c, vec3<f32>(0.0));
    let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    c = mix(vec3<f32>(lum), c, saturation);
    let temp_shift = temperature * 0.01;
    c = vec3<f32>(c.r + temp_shift * 0.5, c.g, c.b - temp_shift * 0.5);
    c = max(c, vec3<f32>(0.0));
    return c;
}}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{
    let dims = vec2<f32>(textureDimensions(hdr_tex));
    let texel = 1.0 / dims;
    let exposure = uniforms.color_grade.x;
    let contrast = uniforms.color_grade.y;
    let saturation = uniforms.color_grade.z;
    let temperature = uniforms.color_grade.w;
    let bloom_enabled = uniforms.flags.x > 0.5;
    let bloom_threshold = uniforms.flags.y;
    var hdr: vec3<f32>;
    if bloom_enabled {{ hdr = bloom_sample(in.uv, texel, bloom_threshold); }}
    else {{ hdr = textureSample(hdr_tex, tex_sampler, in.uv).rgb; }}
    hdr = color_grade(hdr, exposure, contrast, saturation, temperature);
    let mapped = aces_tonemap(hdr);
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
    let center_dist = length(in.uv - 0.5) * 1.4;
    let vignette = 1.0 - center_dist * center_dist * 0.35;
    let final_color = gamma * vignette;
    return vec4<f32>(final_color, 1.0);
}}")
}

fn fxaa_shader() -> String {
    format!("{FULLSCREEN_VS_WGSL}{PP_UNIFORMS_WGSL}
@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: PostProcessUniforms;
fn luma(color: vec3<f32>) -> f32 {{ return dot(color, vec3<f32>(0.299, 0.587, 0.114)); }}
const EDGE_THRESHOLD_MIN: f32 = 0.0625;
const EDGE_THRESHOLD: f32 = 0.125;
const SUBPIXEL_QUALITY: f32 = 0.75;
const SEARCH_STEPS: i32 = 10;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{
    let screen_size = uniforms.screen_size.xy;
    let texel = 1.0 / screen_size;
    let center = textureSample(color_tex, tex_sampler, in.uv);
    let luma_center = luma(center.rgb);
    let luma_n = luma(textureSample(color_tex, tex_sampler, in.uv + vec2<f32>(0.0, -texel.y)).rgb);
    let luma_s = luma(textureSample(color_tex, tex_sampler, in.uv + vec2<f32>(0.0, texel.y)).rgb);
    let luma_e = luma(textureSample(color_tex, tex_sampler, in.uv + vec2<f32>(texel.x, 0.0)).rgb);
    let luma_w = luma(textureSample(color_tex, tex_sampler, in.uv + vec2<f32>(-texel.x, 0.0)).rgb);
    let luma_max = max(max(luma_n, luma_s), max(max(luma_e, luma_w), luma_center));
    let luma_min = min(min(luma_n, luma_s), min(min(luma_e, luma_w), luma_center));
    let luma_range = luma_max - luma_min;
    if luma_range < max(EDGE_THRESHOLD_MIN, luma_max * EDGE_THRESHOLD) {{ return center; }}
    let luma_ne = luma(textureSample(color_tex, tex_sampler, in.uv + vec2<f32>(texel.x, -texel.y)).rgb);
    let luma_nw = luma(textureSample(color_tex, tex_sampler, in.uv + vec2<f32>(-texel.x, -texel.y)).rgb);
    let luma_se = luma(textureSample(color_tex, tex_sampler, in.uv + vec2<f32>(texel.x, texel.y)).rgb);
    let luma_sw = luma(textureSample(color_tex, tex_sampler, in.uv + vec2<f32>(-texel.x, texel.y)).rgb);
    let horizontal = abs((luma_nw + luma_ne) - 2.0 * luma_n) + abs((luma_w + luma_e) - 2.0 * luma_center) * 2.0 + abs((luma_sw + luma_se) - 2.0 * luma_s);
    let vertical = abs((luma_nw + luma_sw) - 2.0 * luma_w) + abs((luma_n + luma_s) - 2.0 * luma_center) * 2.0 + abs((luma_ne + luma_se) - 2.0 * luma_e);
    let is_horizontal = horizontal >= vertical;
    let step_length = select(texel.x, texel.y, is_horizontal);
    var luma1: f32; var luma2: f32;
    if is_horizontal {{ luma1 = luma_n; luma2 = luma_s; }} else {{ luma1 = luma_w; luma2 = luma_e; }}
    let gradient1 = abs(luma1 - luma_center); let gradient2 = abs(luma2 - luma_center);
    let is_steeper1 = gradient1 >= gradient2;
    let luma_local_avg = select((luma2 + luma_center) * 0.5, (luma1 + luma_center) * 0.5, is_steeper1);
    let gradient_scaled = max(gradient1, gradient2) * 0.25;
    var step_dir: f32;
    if is_steeper1 {{ step_dir = -step_length; }} else {{ step_dir = step_length; }}
    var current_uv = in.uv;
    if is_horizontal {{ current_uv.y += step_dir * 0.5; }} else {{ current_uv.x += step_dir * 0.5; }}
    var uv_offset: vec2<f32>;
    if is_horizontal {{ uv_offset = vec2<f32>(texel.x, 0.0); }} else {{ uv_offset = vec2<f32>(0.0, texel.y); }}
    var uv_pos = current_uv + uv_offset; var uv_neg = current_uv - uv_offset;
    var reached_pos = false; var reached_neg = false;
    var luma_end_pos = 0.0; var luma_end_neg = 0.0;
    for (var i = 0i; i < SEARCH_STEPS; i++) {{
        if !reached_pos {{ luma_end_pos = luma(textureSample(color_tex, tex_sampler, uv_pos).rgb) - luma_local_avg; reached_pos = abs(luma_end_pos) >= gradient_scaled; }}
        if !reached_neg {{ luma_end_neg = luma(textureSample(color_tex, tex_sampler, uv_neg).rgb) - luma_local_avg; reached_neg = abs(luma_end_neg) >= gradient_scaled; }}
        if reached_pos && reached_neg {{ break; }}
        if !reached_pos {{ uv_pos += uv_offset; }}
        if !reached_neg {{ uv_neg -= uv_offset; }}
    }}
    var dist_pos: f32; var dist_neg: f32;
    if is_horizontal {{ dist_pos = uv_pos.x - in.uv.x; dist_neg = in.uv.x - uv_neg.x; }}
    else {{ dist_pos = uv_pos.y - in.uv.y; dist_neg = in.uv.y - uv_neg.y; }}
    let is_closer_pos = dist_pos < dist_neg;
    let dist_final = min(dist_pos, dist_neg);
    let edge_length = dist_pos + dist_neg;
    let pixel_offset = -dist_final / edge_length + 0.5;
    let luma_end = select(luma_end_neg, luma_end_pos, is_closer_pos);
    let is_direction_correct = ((luma_center - luma_local_avg) < 0.0) != (luma_end < 0.0);
    let final_offset = select(0.0, pixel_offset, is_direction_correct);
    let luma_avg = (luma_n + luma_s + luma_e + luma_w) * 0.25;
    let subpixel_offset = clamp(abs(luma_avg - luma_center) / luma_range, 0.0, 1.0);
    let subpixel_offset_final = (-2.0 * subpixel_offset + 3.0) * subpixel_offset * subpixel_offset;
    let subpixel = subpixel_offset_final * subpixel_offset_final * SUBPIXEL_QUALITY;
    let best_offset = max(final_offset, subpixel);
    var final_uv = in.uv;
    if is_horizontal {{ final_uv.y += best_offset * step_dir; }}
    else {{ final_uv.x += best_offset * step_dir; }}
    return textureSample(color_tex, tex_sampler, final_uv);
}}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_produce_identity_color_grade() {
        let settings = PostProcessSettings::default();
        assert_eq!(settings.exposure, 0.0);
        assert_eq!(settings.contrast, 1.0);
        assert_eq!(settings.saturation, 1.0);
        assert_eq!(settings.temperature, 0.0);
    }

    #[test]
    fn default_settings_have_ssao_fxaa_bloom_enabled() {
        let settings = PostProcessSettings::default();
        assert!(settings.ssao_enabled);
        assert!(settings.fxaa_enabled);
        assert!(settings.bloom_enabled);
    }

    #[test]
    fn ssr_settings_defaults_in_post_process() {
        let settings = PostProcessSettings::default();
        assert!(settings.ssr.enabled);
        assert_eq!(settings.ssr.max_steps, 64);
    }

    #[test]
    fn uniforms_encode_correctly() {
        let settings = PostProcessSettings {
            taa_enabled: false,
            ssao_enabled: true,
            ssao_radius: 0.75,
            ssao_intensity: 1.5,
            ssgi_enabled: false,
            ssgi_ray_count: 4,
            ssgi_max_distance: 10.0,
            ssgi_intensity: 1.0,
            ssgi_temporal_blend: 0.9,
            fxaa_enabled: true,
            bloom_enabled: false,
            bloom_threshold: 1.0,
            exposure: 0.5,
            contrast: 1.2,
            saturation: 0.8,
            temperature: -10.0,
            ssr_enabled: false,
            ssr: crate::ssr::SsrSettings::default(),
            motion_blur: crate::motion_blur::MotionBlurSettings::default(),
            dof: crate::dof::DofSettings::default(),
            ibl_enabled: false,
            ibl_intensity: 1.0,
            pcss_enabled: true,
        };
        let u = PostProcessUniforms::from_settings(&settings, 1920, 1080);
        assert_eq!(u.color_grade[0], 0.5);
        assert_eq!(u.color_grade[1], 1.2);
        assert_eq!(u.color_grade[2], 0.8);
        assert_eq!(u.color_grade[3], -10.0);
        assert_eq!(u.flags[0], 0.0); // bloom disabled
        assert_eq!(u.flags[2], 1.0); // ssao enabled
        assert_eq!(u.flags2[0], 1.0); // fxaa enabled
        assert_eq!(u.screen_size[0], 1920.0);
        assert_eq!(u.screen_size[1], 1080.0);
    }

    #[test]
    fn ssao_uniforms_size() {
        assert_eq!(std::mem::size_of::<SsaoUniforms>(), 80);
    }

    #[test]
    fn postprocess_uniforms_size() {
        assert_eq!(std::mem::size_of::<PostProcessUniforms>(), 64);
    }

    #[test]
    fn ssr_normals_uniforms_size() {
        let size = std::mem::size_of::<SsrNormalsUniforms>();
        assert_eq!(size % 16, 0);
        assert_eq!(size, 80);
    }
}
