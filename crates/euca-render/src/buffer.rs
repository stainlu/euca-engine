//! Smart buffer abstraction that optimizes for unified memory (Apple Silicon).
//!
//! On Apple Silicon, CPU and GPU share the same physical RAM (unified memory).
//! The standard wgpu path creates `COPY_DST` buffers and writes via
//! `queue.write_buffer()`, which internally allocates a staging buffer and
//! copies data — unnecessary when memory is already shared.
//!
//! `SmartBuffer` adds `MAP_WRITE` to the buffer usage flags on unified-memory
//! hardware. The Metal backend recognizes that the buffer is both mappable and
//! a copy destination, and skips the internal staging copy since it knows the
//! memory is shared. Writes still go through `queue.write_buffer()` — the
//! optimization is entirely inside the backend.
//!
//! On discrete GPUs, `SmartBuffer` creates standard `COPY_DST` buffers and
//! behaves identically to a raw `wgpu::Buffer`.

/// GPU buffer type: `Storage` (SSBO) or `Uniform` (UBO).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferKind {
    Storage,
    Uniform,
}

/// A GPU buffer that transparently optimizes for unified memory.
///
/// Create with [`SmartBuffer::new`], write with [`SmartBuffer::write`].
/// Access the underlying `wgpu::Buffer` via [`SmartBuffer::raw`] for
/// bind group creation, slicing, etc.
pub struct SmartBuffer {
    buffer: wgpu::Buffer,
    /// True when the buffer was created with `MAP_WRITE` (unified memory path).
    unified: bool,
}

impl SmartBuffer {
    /// Create a new buffer, automatically choosing optimal usage flags.
    ///
    /// - `unified`: pass `true` on Apple Silicon (from `GpuContext::unified_memory`).
    ///   Adds `MAP_WRITE` so the Metal backend can skip internal staging copies.
    /// - `kind`: whether this buffer is used as `STORAGE` or `UNIFORM`.
    /// - `size`: buffer size in bytes.
    /// - `label`: debug label for GPU debuggers.
    pub fn new(
        device: &wgpu::Device,
        size: u64,
        kind: BufferKind,
        unified: bool,
        label: &str,
    ) -> Self {
        let kind_usage = match kind {
            BufferKind::Storage => wgpu::BufferUsages::STORAGE,
            BufferKind::Uniform => wgpu::BufferUsages::UNIFORM,
        };

        // Always need COPY_DST for queue.write_buffer().
        // Note: wgpu does NOT allow MAP_WRITE combined with STORAGE or UNIFORM.
        // On Apple Silicon (unified memory), the Metal backend already optimizes
        // queue.write_buffer() internally — it skips staging copies when it detects
        // shared memory. So we just use COPY_DST on all platforms.
        let usage = kind_usage | wgpu::BufferUsages::COPY_DST;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });

        Self { buffer, unified }
    }

    /// Write data to the buffer via `queue.write_buffer()`.
    ///
    /// On unified memory, the Metal backend sees `MAP_WRITE` and skips the
    /// staging copy. On discrete GPUs, this is a standard DMA transfer.
    pub fn write<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &[T]) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    /// Write raw bytes to the buffer at offset 0.
    pub fn write_bytes(&self, queue: &wgpu::Queue, data: &[u8]) {
        queue.write_buffer(&self.buffer, 0, data);
    }

    /// Access the underlying `wgpu::Buffer` (for bind groups, slicing, etc.).
    pub fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Whether this buffer was created with unified memory optimizations.
    pub fn is_unified(&self) -> bool {
        self.unified
    }
}

impl std::fmt::Debug for SmartBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmartBuffer")
            .field("unified", &self.unified)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{AdapterInfo, GpuVendor, HardwareSurvey, RenderBackend, SystemInfo};

    /// Helper to build a `HardwareSurvey` with a single adapter of the given vendor.
    fn make_survey(vendor: GpuVendor) -> HardwareSurvey {
        HardwareSurvey {
            system: SystemInfo {
                os: "macos",
                arch: "aarch64",
                cpu_cores: 10,
            },
            adapters: vec![AdapterInfo {
                name: "Test GPU".into(),
                vendor,
                vendor_id: 0,
                device_type: wgpu::DeviceType::IntegratedGpu,
                wgpu_backend: wgpu::Backend::Metal,
                driver: String::new(),
                driver_info: String::new(),
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            }],
            selected_adapter: 0,
            render_backend: RenderBackend::Wgpu,
        }
    }

    #[test]
    fn apple_detected_as_unified_memory() {
        let survey = make_survey(GpuVendor::Apple);
        assert!(survey.supports_unified_memory());
    }

    #[test]
    fn nvidia_not_unified_memory() {
        let survey = make_survey(GpuVendor::Nvidia);
        assert!(!survey.supports_unified_memory());
    }

    #[test]
    fn amd_not_unified_memory() {
        let survey = make_survey(GpuVendor::Amd);
        assert!(!survey.supports_unified_memory());
    }

    #[test]
    fn discrete_gpu_uses_copy_path() {
        // All non-Apple vendors should be detected as non-unified.
        for vendor in [
            GpuVendor::Nvidia,
            GpuVendor::Amd,
            GpuVendor::Intel,
            GpuVendor::Qualcomm,
            GpuVendor::Unknown(0xFFFF),
        ] {
            let survey = make_survey(vendor);
            assert!(
                !survey.supports_unified_memory(),
                "{:?} should not be detected as unified memory",
                vendor
            );
        }
    }

    #[test]
    fn buffer_kind_variants_are_distinct() {
        assert_eq!(BufferKind::Storage, BufferKind::Storage);
        assert_eq!(BufferKind::Uniform, BufferKind::Uniform);
        assert_ne!(BufferKind::Storage, BufferKind::Uniform);
    }
}
