use std::sync::Arc;
use winit::window::Window;

use crate::hardware::{AdapterInfo, HardwareSurvey, RenderBackend, adapter_info_from_wgpu};
use euca_rhi::wgpu_backend::WgpuDevice;
use euca_rhi::{Capabilities, RenderDevice};

/// Owns the GPU device, queue, surface — everything needed to talk to the GPU.
///
/// Wraps [`WgpuDevice`] (the RHI backend) and adds engine-level metadata
/// (adapter info, render backend selection). Access the underlying wgpu
/// objects via `Deref` (e.g., `gpu.device`, `gpu.queue`).
pub struct GpuContext {
    /// The RHI backend device. Access wgpu objects via Deref:
    /// `gpu.device`, `gpu.queue`, `gpu.surface`, etc.
    rhi: WgpuDevice,
    /// Info about the adapter actually in use (from surface-compatible selection).
    pub adapter_info: AdapterInfo,
    /// Rendering backend chosen by the hardware survey.
    pub render_backend: RenderBackend,
}

impl std::ops::Deref for GpuContext {
    type Target = WgpuDevice;
    fn deref(&self) -> &WgpuDevice {
        &self.rhi
    }
}

impl std::ops::DerefMut for GpuContext {
    fn deref_mut(&mut self) -> &mut WgpuDevice {
        &mut self.rhi
    }
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
            limits.max_binding_array_elements_per_shader_stage = adapter_limits
                .max_binding_array_elements_per_shader_stage
                .max(512);
            limits.max_bindings_per_bind_group =
                adapter_limits.max_bindings_per_bind_group.max(514);
        }

        // Snapshot limit values before `limits` is moved into request_device.
        let cap_max_tex_dim = limits.max_texture_dimension_2d;
        let cap_max_bind_groups = limits.max_bind_groups;
        let cap_max_bindings = limits.max_bindings_per_bind_group;
        let cap_max_binding_array = limits.max_binding_array_elements_per_shader_stage;

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

        let unified_memory = survey.supports_unified_memory();

        let capabilities = Capabilities {
            unified_memory,
            multi_draw_indirect: has_multi_draw_indirect,
            multi_draw_indirect_count: has_multi_draw_indirect_count,
            texture_binding_array: required_features.contains(bindless_features),
            non_uniform_indexing: required_features.contains(bindless_features),
            max_texture_dimension_2d: cap_max_tex_dim,
            max_bind_groups: cap_max_bind_groups,
            max_bindings_per_bind_group: cap_max_bindings,
            max_binding_array_elements: cap_max_binding_array,
            device_name: adapter_info.name.clone(),
            ..Default::default()
        };

        let rhi = WgpuDevice::new(device, queue, surface, surface_config, window, capabilities);

        Self {
            rhi,
            adapter_info,
            render_backend: survey.render_backend,
        }
    }

    /// Whether the GPU has unified memory (Apple Silicon).
    pub fn unified_memory(&self) -> bool {
        self.rhi.capabilities().unified_memory
    }

    /// Whether the GPU supports `multi_draw_indexed_indirect`.
    pub fn has_multi_draw_indirect(&self) -> bool {
        self.rhi.capabilities().multi_draw_indirect
    }

    /// Whether the GPU supports `multi_draw_indexed_indirect_count`.
    pub fn has_multi_draw_indirect_count(&self) -> bool {
        self.rhi.capabilities().multi_draw_indirect_count
    }

    /// Handle window resize.
    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        self.rhi.resize_surface(new_width, new_height);
    }

    /// Current surface aspect ratio.
    pub fn aspect_ratio(&self) -> f32 {
        self.rhi.aspect_ratio()
    }
}
