//! Hardware survey module — enumerates all GPUs and system info at startup.
//!
//! Runs before window creation using `wgpu::Instance::enumerate_adapters()`.
//! Follows the UE5 pattern: detect → select → log → store as resource.

/// Known GPU vendors by PCI vendor ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Apple,
    Nvidia,
    Amd,
    Intel,
    Qualcomm,
    Unknown(u32),
}

impl GpuVendor {
    /// Identify vendor from PCI vendor ID and adapter name.
    ///
    /// The Metal backend reports `vendor: 0` — it doesn't expose PCI IDs.
    /// In that case, we fall back to matching the adapter name string.
    pub fn from_id_and_name(id: u32, name: &str) -> Self {
        // Try PCI vendor ID first
        match id {
            0x106B => return Self::Apple,
            0x10DE => return Self::Nvidia,
            0x1002 => return Self::Amd,
            0x8086 => return Self::Intel,
            0x5143 => return Self::Qualcomm,
            _ => {}
        }

        // Fallback: match adapter name (Metal backend reports vendor=0)
        let lower = name.to_lowercase();
        if lower.starts_with("apple") {
            Self::Apple
        } else if lower.contains("nvidia") || lower.contains("geforce") || lower.contains("rtx") {
            Self::Nvidia
        } else if lower.contains("amd") || lower.contains("radeon") {
            Self::Amd
        } else if lower.contains("intel") {
            Self::Intel
        } else if lower.contains("qualcomm") || lower.contains("adreno") {
            Self::Qualcomm
        } else {
            Self::Unknown(id)
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Apple => "Apple",
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Qualcomm => "Qualcomm",
            Self::Unknown(_) => "Unknown",
        }
    }
}

/// Rendering backend for this session. Chosen once at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderBackend {
    /// Cross-platform wgpu path (default).
    Wgpu,
    /// Native Metal path for Apple Silicon (future: ray tracing, tile memory, MetalFX).
    #[cfg(feature = "metal-native")]
    MetalNative,
}

/// Capabilities of a single GPU adapter.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub name: String,
    pub vendor: GpuVendor,
    pub vendor_id: u32,
    pub device_type: wgpu::DeviceType,
    pub wgpu_backend: wgpu::Backend,
    pub driver: String,
    pub driver_info: String,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
}

/// Basic system info from std (no extra deps).
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub os: &'static str,
    pub arch: &'static str,
    pub cpu_cores: usize,
}

/// Full hardware survey — run once at startup, before window creation.
#[derive(Debug, Clone)]
pub struct HardwareSurvey {
    pub system: SystemInfo,
    pub adapters: Vec<AdapterInfo>,
    pub selected_adapter: usize,
    pub render_backend: RenderBackend,
}

impl HardwareSurvey {
    /// Enumerate all GPUs and system info. No window needed.
    ///
    /// Returns the survey data and the wgpu Instance. Pass both to
    /// `GpuContext::new()` so it reuses the same Instance (avoids
    /// creating a second one that might see different adapters).
    pub fn detect() -> (Self, wgpu::Instance) {
        let system = SystemInfo {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        };

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapters: Vec<AdapterInfo> = instance
            .enumerate_adapters(wgpu::Backends::PRIMARY)
            .into_iter()
            .map(|adapter| {
                let info = adapter.get_info();
                let vendor = GpuVendor::from_id_and_name(info.vendor, &info.name);
                AdapterInfo {
                    name: info.name,
                    vendor,
                    vendor_id: info.vendor,
                    device_type: info.device_type,
                    wgpu_backend: info.backend,
                    driver: info.driver,
                    driver_info: info.driver_info,
                    features: adapter.features(),
                    limits: adapter.limits(),
                }
            })
            .collect();

        let selected_adapter = Self::select_adapter(&adapters);
        let render_backend = Self::select_backend(&adapters, selected_adapter);

        let survey = Self {
            system,
            adapters,
            selected_adapter,
            render_backend,
        };
        survey.log_results();
        (survey, instance)
    }

    /// Prefer DiscreteGpu > IntegratedGpu > VirtualGpu > Cpu > Other.
    fn select_adapter(adapters: &[AdapterInfo]) -> usize {
        fn priority(dt: wgpu::DeviceType) -> u8 {
            match dt {
                wgpu::DeviceType::DiscreteGpu => 4,
                wgpu::DeviceType::IntegratedGpu => 3,
                wgpu::DeviceType::VirtualGpu => 2,
                wgpu::DeviceType::Cpu => 1,
                wgpu::DeviceType::Other => 0,
            }
        }

        adapters
            .iter()
            .enumerate()
            .max_by_key(|(_, a)| priority(a.device_type))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Determine render backend from selected adapter.
    fn select_backend(adapters: &[AdapterInfo], selected: usize) -> RenderBackend {
        #[cfg(feature = "metal-native")]
        if let Some(adapter) = adapters.get(selected) {
            if adapter.vendor == GpuVendor::Apple && cfg!(target_arch = "aarch64") {
                return RenderBackend::MetalNative;
            }
        }

        #[cfg(not(feature = "metal-native"))]
        let _ = (adapters, selected);

        RenderBackend::Wgpu
    }

    /// Get the selected adapter info.
    pub fn selected(&self) -> &AdapterInfo {
        &self.adapters[self.selected_adapter]
    }

    /// Log hardware survey results.
    fn log_results(&self) {
        log::info!("── Hardware Survey ──");
        log::info!(
            "System: {} {}, {} CPU cores",
            self.system.os,
            self.system.arch,
            self.system.cpu_cores
        );
        log::info!("GPU adapters found: {}", self.adapters.len());

        for (i, adapter) in self.adapters.iter().enumerate() {
            let marker = if i == self.selected_adapter { " *" } else { "" };
            log::info!(
                "  [{}]{} {} ({:?})",
                i,
                marker,
                adapter.name,
                adapter.device_type,
            );
            log::info!(
                "      Vendor: {}, Backend: {:?}, Driver: {}",
                adapter.vendor.name(),
                adapter.wgpu_backend,
                adapter.driver,
            );
            log::info!(
                "      Max texture: {}, Max storage buffer: {} MB",
                adapter.limits.max_texture_dimension_2d,
                adapter.limits.max_storage_buffer_binding_size / (1024 * 1024),
            );
        }

        log::info!(
            "Selected: [{}] {}",
            self.selected_adapter,
            self.selected().name,
        );
        log::info!("Render backend: {:?}", self.render_backend);
        log::info!("─────────────────────");
    }
}

/// Build an `AdapterInfo` from a wgpu adapter (reused by both survey and GpuContext).
pub(crate) fn adapter_info_from_wgpu(adapter: &wgpu::Adapter) -> AdapterInfo {
    let info = adapter.get_info();
    let vendor = GpuVendor::from_id_and_name(info.vendor, &info.name);
    AdapterInfo {
        name: info.name,
        vendor,
        vendor_id: info.vendor,
        device_type: info.device_type,
        wgpu_backend: info.backend,
        driver: info.driver,
        driver_info: info.driver_info,
        features: adapter.features(),
        limits: adapter.limits(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_from_known_ids() {
        assert_eq!(GpuVendor::from_id_and_name(0x106B, ""), GpuVendor::Apple);
        assert_eq!(GpuVendor::from_id_and_name(0x10DE, ""), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_id_and_name(0x1002, ""), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_id_and_name(0x8086, ""), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_id_and_name(0x5143, ""), GpuVendor::Qualcomm);
        assert_eq!(
            GpuVendor::from_id_and_name(0xFFFF, ""),
            GpuVendor::Unknown(0xFFFF)
        );
    }

    #[test]
    fn vendor_from_name_fallback() {
        // Metal backend reports vendor=0, so we fall back to name matching
        assert_eq!(
            GpuVendor::from_id_and_name(0, "Apple M4 Pro"),
            GpuVendor::Apple
        );
        assert_eq!(
            GpuVendor::from_id_and_name(0, "NVIDIA GeForce RTX 4090"),
            GpuVendor::Nvidia
        );
        assert_eq!(
            GpuVendor::from_id_and_name(0, "AMD Radeon RX 7900"),
            GpuVendor::Amd
        );
        assert_eq!(
            GpuVendor::from_id_and_name(0, "Intel UHD 770"),
            GpuVendor::Intel
        );
    }

    #[test]
    fn vendor_names() {
        assert_eq!(GpuVendor::Apple.name(), "Apple");
        assert_eq!(GpuVendor::Nvidia.name(), "NVIDIA");
        assert_eq!(GpuVendor::Unknown(0).name(), "Unknown");
    }

    #[test]
    fn select_adapter_prefers_discrete() {
        let adapters = vec![
            AdapterInfo {
                name: "Integrated".into(),
                vendor: GpuVendor::Intel,
                vendor_id: 0x8086,
                device_type: wgpu::DeviceType::IntegratedGpu,
                wgpu_backend: wgpu::Backend::Vulkan,
                driver: String::new(),
                driver_info: String::new(),
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            AdapterInfo {
                name: "Discrete".into(),
                vendor: GpuVendor::Nvidia,
                vendor_id: 0x10DE,
                device_type: wgpu::DeviceType::DiscreteGpu,
                wgpu_backend: wgpu::Backend::Vulkan,
                driver: String::new(),
                driver_info: String::new(),
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
        ];
        assert_eq!(HardwareSurvey::select_adapter(&adapters), 1);
    }

    #[test]
    fn select_backend_defaults_to_wgpu() {
        let adapters = vec![AdapterInfo {
            name: "Apple M2".into(),
            vendor: GpuVendor::Apple,
            vendor_id: 0x106B,
            device_type: wgpu::DeviceType::IntegratedGpu,
            wgpu_backend: wgpu::Backend::Metal,
            driver: String::new(),
            driver_info: String::new(),
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        }];
        // Without metal-native feature, always returns Wgpu
        assert_eq!(
            HardwareSurvey::select_backend(&adapters, 0),
            RenderBackend::Wgpu
        );
    }

    #[test]
    fn system_info_is_populated() {
        let info = SystemInfo {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        };
        assert!(!info.os.is_empty());
        assert!(!info.arch.is_empty());
        assert!(info.cpu_cores >= 1);
    }
}
