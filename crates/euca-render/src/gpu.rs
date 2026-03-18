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

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Euca GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
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
