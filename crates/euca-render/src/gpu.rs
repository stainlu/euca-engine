use std::sync::Arc;
use winit::window::Window;

use crate::hardware::{AdapterInfo, HardwareSurvey, RenderBackend, adapter_info_from_wgpu};

/// Owns the wgpu device, queue, surface — everything needed to talk to the GPU.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub window: Arc<Window>,
    /// Info about the adapter actually in use (from surface-compatible selection).
    pub adapter_info: AdapterInfo,
    /// Rendering backend chosen by the hardware survey.
    pub render_backend: RenderBackend,
    /// Whether the GPU has unified memory (Apple Silicon).
    pub unified_memory: bool,
    /// Whether the GPU supports `multi_draw_indexed_indirect`.
    pub has_multi_draw_indirect: bool,
    /// Whether the GPU supports `multi_draw_indexed_indirect_count`.
    pub has_multi_draw_indirect_count: bool,
}

impl GpuContext {
    /// Initialize wgpu from a winit window, hardware survey, and the wgpu Instance.
    ///
    /// Both `survey` and `instance` come from `HardwareSurvey::detect()`.
    /// Reusing the same Instance guarantees adapter consistency with the survey.
    pub fn new(window: Window, survey: &HardwareSurvey, instance: &wgpu::Instance) -> Self {
        let window = Arc::new(window);

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter found");

        let adapter_info = adapter_info_from_wgpu(&adapter);

        // Verify surface-compatible adapter matches survey selection
        if adapter_info.name != survey.selected().name {
            log::warn!(
                "Surface-compatible adapter '{}' differs from survey selection '{}'",
                adapter_info.name,
                survey.selected().name,
            );
        }

        // Request optional GPU features that improve performance when available.
        // MULTI_DRAW_INDIRECT_COUNT enables both multi_draw_indexed_indirect and
        // multi_draw_indexed_indirect_count, collapsing N per-entity draw calls
        // into a single API call.
        let supported = adapter.features();
        let mut required_features = wgpu::Features::empty();
        if supported.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
            required_features |= wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        }

        // Bindless materials: texture binding arrays + non-uniform indexing.
        let bindless_features = crate::bindless::BINDLESS_FEATURES;
        if supported.contains(bindless_features) {
            required_features |= bindless_features;
            log::info!("GPU supports TEXTURE_BINDING_ARRAY — bindless materials enabled");
        }

        let has_multi_draw_indirect =
            required_features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        let has_multi_draw_indirect_count = has_multi_draw_indirect;

        if has_multi_draw_indirect_count {
            log::info!("GPU supports MULTI_DRAW_INDIRECT_COUNT — GPU-driven draw calls enabled");
        }

        // Build limits: start from defaults, then raise binding array limits
        // if bindless features are available.
        let mut limits = wgpu::Limits::default();
        if required_features.contains(bindless_features) {
            let adapter_limits = adapter.limits();
            limits.max_binding_array_elements_per_shader_stage =
                adapter_limits.max_binding_array_elements_per_shader_stage.max(512);
            limits.max_bindings_per_bind_group =
                adapter_limits.max_bindings_per_bind_group.max(514); // 512 textures + buffer + sampler
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Euca GPU Device"),
            required_features,
            required_limits: limits,
            ..Default::default()
        }))
        .expect("Failed to create GPU device");

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        Self {
            device,
            queue,
            surface,
            surface_config,
            window,
            adapter_info,
            render_backend: survey.render_backend,
            unified_memory: survey.supports_unified_memory(),
            has_multi_draw_indirect,
            has_multi_draw_indirect_count,
        }
    }

    /// Handle window resize.
    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width > 0 && new_height > 0 {
            self.surface_config.width = new_width;
            self.surface_config.height = new_height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    /// Current surface aspect ratio.
    pub fn aspect_ratio(&self) -> f32 {
        self.surface_config.width as f32 / self.surface_config.height as f32
    }
}
