use std::sync::Arc;
use winit::window::Window;

use crate::hardware::{AdapterInfo, HardwareSurvey, RenderBackend, adapter_info_from_wgpu};
use euca_rhi::wgpu_backend::WgpuDevice;
use euca_rhi::{Capabilities, RenderDevice};

/// Owns the GPU device, queue, surface — everything needed to talk to the GPU.
///
/// Generic over `D: RenderDevice` to support multiple backends:
/// - `GpuContext` (default) = `GpuContext<WgpuDevice>` — cross-platform via wgpu
/// - `GpuContext<MetalDevice>` — native Metal on Apple Silicon
///
/// Access the underlying RHI device via `Deref` (e.g., `gpu.create_buffer()`).
pub struct GpuContext<D: RenderDevice = WgpuDevice> {
    /// The RHI backend device.
    rhi: D,
    /// Info about the adapter actually in use.
    pub adapter_info: AdapterInfo,
    /// Rendering backend chosen by the hardware survey.
    pub render_backend: RenderBackend,
    /// Window reference for request_redraw and size queries.
    pub window: Arc<Window>,
}

impl<D: RenderDevice> std::ops::Deref for GpuContext<D> {
    type Target = D;
    fn deref(&self) -> &D {
        &self.rhi
    }
}

impl<D: RenderDevice> std::ops::DerefMut for GpuContext<D> {
    fn deref_mut(&mut self) -> &mut D {
        &mut self.rhi
    }
}

// =========================================================================
// Generic methods (work for any backend)
// =========================================================================

impl<D: RenderDevice> GpuContext<D> {
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

// =========================================================================
// wgpu backend: GpuContext (= GpuContext<WgpuDevice>)
// =========================================================================

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
        let bindless_features = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
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

        let rhi = WgpuDevice::new(
            device,
            queue,
            surface,
            surface_config,
            window.clone(),
            capabilities,
        );

        Self {
            rhi,
            adapter_info,
            render_backend: survey.render_backend,
            window,
        }
    }
}

// =========================================================================
// wgpu backend: async init (for WASM where pollster::block_on is unavailable)
// =========================================================================

impl GpuContext {
    /// Async version of [`GpuContext::new`] for environments where blocking is
    /// unavailable (WASM).
    pub async fn new_async(
        window: winit::window::Window,
        survey: &HardwareSurvey,
        instance: &wgpu::Instance,
    ) -> Self {
        let window = Arc::new(window);

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found");

        let adapter_info = adapter_info_from_wgpu(&adapter);

        if adapter_info.name != survey.selected().name {
            log::warn!(
                "Surface-compatible adapter '{}' differs from survey selection '{}'",
                adapter_info.name,
                survey.selected().name,
            );
        }

        let supported = adapter.features();
        let mut required_features = wgpu::Features::empty();
        if supported.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
            required_features |= wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        }

        let bindless_features = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        if supported.contains(bindless_features) {
            required_features |= bindless_features;
        }

        let has_multi_draw_indirect =
            required_features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);

        let mut limits = wgpu::Limits::downlevel_webgl2_defaults();
        if required_features.contains(bindless_features) {
            let adapter_limits = adapter.limits();
            limits.max_binding_array_elements_per_shader_stage = adapter_limits
                .max_binding_array_elements_per_shader_stage
                .max(512);
            limits.max_bindings_per_bind_group =
                adapter_limits.max_bindings_per_bind_group.max(514);
        }

        let cap_max_tex_dim = limits.max_texture_dimension_2d;
        let cap_max_bind_groups = limits.max_bind_groups;
        let cap_max_bindings = limits.max_bindings_per_bind_group;
        let cap_max_binding_array = limits.max_binding_array_elements_per_shader_stage;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Euca GPU Device"),
                required_features,
                required_limits: limits,
                ..Default::default()
            })
            .await
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
            multi_draw_indirect_count: has_multi_draw_indirect,
            texture_binding_array: required_features.contains(bindless_features),
            non_uniform_indexing: required_features.contains(bindless_features),
            max_texture_dimension_2d: cap_max_tex_dim,
            max_bind_groups: cap_max_bind_groups,
            max_bindings_per_bind_group: cap_max_bindings,
            max_binding_array_elements: cap_max_binding_array,
            device_name: adapter_info.name.clone(),
            ..Default::default()
        };

        let rhi = WgpuDevice::new(
            device,
            queue,
            surface,
            surface_config,
            window.clone(),
            capabilities,
        );

        Self {
            rhi,
            adapter_info,
            render_backend: survey.render_backend,
            window,
        }
    }
}

// =========================================================================
// Metal backend: GpuContext<MetalDevice>
// =========================================================================

#[cfg(all(target_os = "macos", feature = "metal-native"))]
impl GpuContext<euca_rhi::metal_backend::MetalDevice> {
    /// Create a GpuContext backed by native Metal.
    ///
    /// Bypasses wgpu to access Metal 3/4 features: mesh shaders, tile shading,
    /// indirect command buffers, MetalFX upscaling, memoryless render targets.
    pub fn new_metal(window: Arc<Window>) -> Self {
        use crate::hardware::GpuVendor;

        let device = euca_rhi::metal_backend::MetalDevice::from_window(&window);
        let caps = device.capabilities().clone();

        Self {
            adapter_info: AdapterInfo {
                name: caps.device_name.clone(),
                vendor: GpuVendor::Apple,
                vendor_id: 0x106B, // Apple vendor ID
                device_type: wgpu::DeviceType::IntegratedGpu,
                wgpu_backend: wgpu::Backend::Metal,
                driver: "Native Metal".into(),
                driver_info: String::new(),
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            render_backend: RenderBackend::MetalNative,
            rhi: device,
            window,
        }
    }
}
