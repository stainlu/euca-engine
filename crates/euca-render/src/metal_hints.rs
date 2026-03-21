//! Metal-optimized render path hints for Apple Silicon GPUs.
//!
//! Apple Silicon uses tile-based deferred rendering (TBDR), where the GPU
//! processes geometry in screen-space tiles and intermediate data can remain
//! in fast on-chip tile memory instead of being written to VRAM. While wgpu
//! does not expose raw Metal APIs, we can structure render passes to be
//! TBDR-friendly:
//!
//! - Combine geometry and lighting in fewer passes so intermediates stay on-chip.
//! - Use `StoreOp::Discard` for attachments that are not needed after the pass
//!   (avoids writing tile data to VRAM).
//! - Use `LoadOp::Clear` instead of `LoadOp::Load` when possible (avoids
//!   loading stale tile data from VRAM).
//! - Use 32-thread workgroups for compute dispatches (matches Apple SIMD width).

use crate::hardware::HardwareSurvey;

// ---------------------------------------------------------------------------
// MetalRenderHints
// ---------------------------------------------------------------------------

/// GPU-architecture hints derived from the hardware survey.
///
/// These are advisory values that the renderer uses to choose between
/// TBDR-optimized and traditional multi-pass strategies. All fields are
/// determined once at startup and remain constant for the session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalRenderHints {
    /// Whether the selected adapter is an Apple GPU (M-series / A-series).
    pub is_apple_gpu: bool,
    /// Whether to prefer merging geometry + lighting into a single render pass.
    /// True on TBDR architectures where tile memory avoids VRAM round-trips.
    pub prefer_single_pass: bool,
    /// SIMD group width for compute shaders: 32 on Apple, 64 on discrete GPUs.
    pub optimal_threadgroup_size: u32,
    /// Whether the GPU supports memoryless (tile-only) attachments.
    /// On Apple Silicon, intermediate render targets that are never read back
    /// can be stored exclusively in tile memory.
    pub supports_memoryless: bool,
}

impl MetalRenderHints {
    /// Detect render hints from the hardware survey.
    ///
    /// Call this once after `HardwareSurvey::detect()` and store the result
    /// as a resource for the renderer to consult.
    pub fn detect(survey: &HardwareSurvey) -> Self {
        use crate::hardware::GpuVendor;

        let selected = survey.selected();
        let is_apple_gpu = selected.vendor == GpuVendor::Apple;

        Self {
            is_apple_gpu,
            // TBDR architectures benefit from single-pass rendering because
            // intermediate results stay in fast tile memory.
            prefer_single_pass: is_apple_gpu,
            // Apple GPUs use 32-wide SIMD groups. Desktop GPUs (NVIDIA, AMD)
            // use 32 or 64, but 64 is the safe common denominator for warp/
            // wavefront alignment.
            optimal_threadgroup_size: if is_apple_gpu { 32 } else { 64 },
            // All Apple Silicon GPUs support memoryless render targets. These
            // attachments exist only in tile memory and are never backed by VRAM.
            supports_memoryless: is_apple_gpu,
        }
    }

    /// Returns the optimal 3D workgroup size for compute dispatches.
    ///
    /// The returned array is `[N, 1, 1]` where N matches the GPU's SIMD width.
    /// Callers should compile shaders with `@workgroup_size(N, 1, 1)` or
    /// configure dispatches accordingly.
    pub fn optimal_workgroup_size(&self) -> [u32; 3] {
        [self.optimal_threadgroup_size, 1, 1]
    }

    /// Calculate the number of workgroups needed to cover `total_invocations`.
    ///
    /// Divides with ceiling so no invocations are missed.
    pub fn workgroup_count(&self, total_invocations: u32) -> u32 {
        total_invocations.div_ceil(self.optimal_threadgroup_size)
    }
}

// ---------------------------------------------------------------------------
// RenderPassLayout
// ---------------------------------------------------------------------------

/// High-level render pass strategy chosen based on GPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPassLayout {
    /// TBDR-friendly: combine shadow, geometry, and lighting in fewer passes.
    /// Intermediate attachments stay in tile memory and are never written to VRAM.
    SinglePass,
    /// Traditional multi-pass: separate geometry, shadow, and lighting passes.
    /// Each pass writes its results to VRAM-backed textures.
    MultiPass,
}

// ---------------------------------------------------------------------------
// Attachment operations
// ---------------------------------------------------------------------------

/// Recommended `wgpu::Operations` for a color attachment.
///
/// Encapsulates the load/store strategy so callers do not need to reason
/// about TBDR vs. immediate-mode differences themselves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttachmentOps {
    pub load: wgpu::LoadOp<wgpu::Color>,
    pub store: wgpu::StoreOp,
}

/// Recommended `wgpu::Operations` for a depth/stencil attachment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthAttachmentOps {
    pub load: wgpu::LoadOp<f32>,
    pub store: wgpu::StoreOp,
}

// ---------------------------------------------------------------------------
// RenderPassOptimizer
// ---------------------------------------------------------------------------

/// Chooses render-pass structure and attachment operations based on
/// `MetalRenderHints`.
///
/// This is a stateless helper: all methods are pure functions of the hints.
pub struct RenderPassOptimizer;

impl RenderPassOptimizer {
    /// Choose the pass layout based on GPU architecture hints.
    pub fn optimize_pass_structure(hints: &MetalRenderHints) -> RenderPassLayout {
        if hints.prefer_single_pass {
            RenderPassLayout::SinglePass
        } else {
            RenderPassLayout::MultiPass
        }
    }

    /// Recommended operations for an intermediate color attachment that is
    /// consumed within the same pass and not needed afterwards.
    ///
    /// On TBDR GPUs, discarding the store avoids an expensive tile-to-VRAM
    /// write. On immediate-mode GPUs, the driver may still benefit from the
    /// discard hint.
    pub fn intermediate_color_ops(hints: &MetalRenderHints) -> AttachmentOps {
        AttachmentOps {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: if hints.supports_memoryless {
                wgpu::StoreOp::Discard
            } else {
                wgpu::StoreOp::Store
            },
        }
    }

    /// Recommended operations for a final color attachment that will be
    /// presented or sampled later.
    ///
    /// Always uses `Store` because the result must survive past the pass.
    pub fn final_color_ops() -> AttachmentOps {
        AttachmentOps {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: wgpu::StoreOp::Store,
        }
    }

    /// Recommended operations for a depth attachment in a geometry pass.
    ///
    /// Uses `Clear` to avoid loading stale tile data. The store action
    /// depends on whether the depth buffer is needed by later passes.
    pub fn depth_ops(hints: &MetalRenderHints, needed_later: bool) -> DepthAttachmentOps {
        DepthAttachmentOps {
            load: wgpu::LoadOp::Clear(1.0),
            store: if needed_later || !hints.supports_memoryless {
                wgpu::StoreOp::Store
            } else {
                wgpu::StoreOp::Discard
            },
        }
    }

    /// Recommended operations for a G-buffer attachment (position, normals, etc.).
    ///
    /// In single-pass mode the G-buffer is consumed within the same pass,
    /// so the attachment can be discarded. In multi-pass mode the G-buffer
    /// must survive for the lighting pass.
    pub fn gbuffer_ops(hints: &MetalRenderHints) -> AttachmentOps {
        let layout = Self::optimize_pass_structure(hints);
        match layout {
            RenderPassLayout::SinglePass => AttachmentOps {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Discard,
            },
            RenderPassLayout::MultiPass => AttachmentOps {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{AdapterInfo, GpuVendor, HardwareSurvey, RenderBackend, SystemInfo};

    /// Helper: build a minimal `HardwareSurvey` with one adapter.
    fn make_survey(vendor: GpuVendor, device_type: wgpu::DeviceType) -> HardwareSurvey {
        let adapter = AdapterInfo {
            name: match vendor {
                GpuVendor::Apple => "Apple M4 Pro".into(),
                GpuVendor::Nvidia => "NVIDIA GeForce RTX 4090".into(),
                GpuVendor::Amd => "AMD Radeon RX 7900".into(),
                GpuVendor::Intel => "Intel UHD 770".into(),
                _ => "Unknown GPU".into(),
            },
            vendor,
            vendor_id: 0,
            device_type,
            wgpu_backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        };
        HardwareSurvey {
            system: SystemInfo {
                os: "macos",
                arch: "aarch64",
                cpu_cores: 10,
            },
            adapters: vec![adapter],
            selected_adapter: 0,
            render_backend: RenderBackend::Wgpu,
        }
    }

    // -----------------------------------------------------------------------
    // Detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn detect_apple_gpu() {
        let survey = make_survey(GpuVendor::Apple, wgpu::DeviceType::IntegratedGpu);
        let hints = MetalRenderHints::detect(&survey);

        assert!(hints.is_apple_gpu);
        assert!(hints.prefer_single_pass);
        assert!(hints.supports_memoryless);
        assert_eq!(hints.optimal_threadgroup_size, 32);
    }

    #[test]
    fn detect_non_apple_gpu() {
        let survey = make_survey(GpuVendor::Nvidia, wgpu::DeviceType::DiscreteGpu);
        let hints = MetalRenderHints::detect(&survey);

        assert!(!hints.is_apple_gpu);
        assert!(!hints.prefer_single_pass);
        assert!(!hints.supports_memoryless);
        assert_eq!(hints.optimal_threadgroup_size, 64);
    }

    #[test]
    fn detect_amd_gpu() {
        let survey = make_survey(GpuVendor::Amd, wgpu::DeviceType::DiscreteGpu);
        let hints = MetalRenderHints::detect(&survey);

        assert!(!hints.is_apple_gpu);
        assert!(!hints.prefer_single_pass);
        assert_eq!(hints.optimal_threadgroup_size, 64);
    }

    // -----------------------------------------------------------------------
    // Workgroup size tests
    // -----------------------------------------------------------------------

    #[test]
    fn optimal_workgroup_size_apple() {
        let survey = make_survey(GpuVendor::Apple, wgpu::DeviceType::IntegratedGpu);
        let hints = MetalRenderHints::detect(&survey);

        assert_eq!(hints.optimal_workgroup_size(), [32, 1, 1]);
    }

    #[test]
    fn optimal_workgroup_size_discrete() {
        let survey = make_survey(GpuVendor::Nvidia, wgpu::DeviceType::DiscreteGpu);
        let hints = MetalRenderHints::detect(&survey);

        assert_eq!(hints.optimal_workgroup_size(), [64, 1, 1]);
    }

    #[test]
    fn workgroup_count_rounding() {
        let survey = make_survey(GpuVendor::Apple, wgpu::DeviceType::IntegratedGpu);
        let hints = MetalRenderHints::detect(&survey);

        // 32-wide workgroups
        assert_eq!(hints.workgroup_count(0), 0);
        assert_eq!(hints.workgroup_count(1), 1);
        assert_eq!(hints.workgroup_count(32), 1);
        assert_eq!(hints.workgroup_count(33), 2);
        assert_eq!(hints.workgroup_count(64), 2);
        assert_eq!(hints.workgroup_count(65), 3);
    }

    // -----------------------------------------------------------------------
    // Pass optimization tests
    // -----------------------------------------------------------------------

    #[test]
    fn pass_layout_single_on_apple() {
        let survey = make_survey(GpuVendor::Apple, wgpu::DeviceType::IntegratedGpu);
        let hints = MetalRenderHints::detect(&survey);

        assert_eq!(
            RenderPassOptimizer::optimize_pass_structure(&hints),
            RenderPassLayout::SinglePass
        );
    }

    #[test]
    fn pass_layout_multi_on_discrete() {
        let survey = make_survey(GpuVendor::Nvidia, wgpu::DeviceType::DiscreteGpu);
        let hints = MetalRenderHints::detect(&survey);

        assert_eq!(
            RenderPassOptimizer::optimize_pass_structure(&hints),
            RenderPassLayout::MultiPass
        );
    }

    // -----------------------------------------------------------------------
    // Store action selection tests
    // -----------------------------------------------------------------------

    #[test]
    fn intermediate_color_discard_on_apple() {
        let survey = make_survey(GpuVendor::Apple, wgpu::DeviceType::IntegratedGpu);
        let hints = MetalRenderHints::detect(&survey);
        let ops = RenderPassOptimizer::intermediate_color_ops(&hints);

        assert_eq!(ops.store, wgpu::StoreOp::Discard);
        assert!(matches!(ops.load, wgpu::LoadOp::Clear(_)));
    }

    #[test]
    fn intermediate_color_store_on_discrete() {
        let survey = make_survey(GpuVendor::Nvidia, wgpu::DeviceType::DiscreteGpu);
        let hints = MetalRenderHints::detect(&survey);
        let ops = RenderPassOptimizer::intermediate_color_ops(&hints);

        assert_eq!(ops.store, wgpu::StoreOp::Store);
    }

    #[test]
    fn final_color_always_stores() {
        let ops = RenderPassOptimizer::final_color_ops();

        assert_eq!(ops.store, wgpu::StoreOp::Store);
        assert!(matches!(ops.load, wgpu::LoadOp::Clear(_)));
    }

    #[test]
    fn depth_discard_when_not_needed_on_apple() {
        let survey = make_survey(GpuVendor::Apple, wgpu::DeviceType::IntegratedGpu);
        let hints = MetalRenderHints::detect(&survey);
        let ops = RenderPassOptimizer::depth_ops(&hints, false);

        assert_eq!(ops.store, wgpu::StoreOp::Discard);
        assert!(matches!(ops.load, wgpu::LoadOp::Clear(_)));
    }

    #[test]
    fn depth_store_when_needed_later() {
        let survey = make_survey(GpuVendor::Apple, wgpu::DeviceType::IntegratedGpu);
        let hints = MetalRenderHints::detect(&survey);
        let ops = RenderPassOptimizer::depth_ops(&hints, true);

        assert_eq!(ops.store, wgpu::StoreOp::Store);
    }

    #[test]
    fn depth_always_stores_on_discrete() {
        let survey = make_survey(GpuVendor::Nvidia, wgpu::DeviceType::DiscreteGpu);
        let hints = MetalRenderHints::detect(&survey);

        // Even if not needed later, discrete GPUs don't support memoryless
        let ops = RenderPassOptimizer::depth_ops(&hints, false);
        assert_eq!(ops.store, wgpu::StoreOp::Store);
    }

    // -----------------------------------------------------------------------
    // G-buffer ops tests
    // -----------------------------------------------------------------------

    #[test]
    fn gbuffer_discard_in_single_pass() {
        let survey = make_survey(GpuVendor::Apple, wgpu::DeviceType::IntegratedGpu);
        let hints = MetalRenderHints::detect(&survey);
        let ops = RenderPassOptimizer::gbuffer_ops(&hints);

        assert_eq!(ops.store, wgpu::StoreOp::Discard);
    }

    #[test]
    fn gbuffer_store_in_multi_pass() {
        let survey = make_survey(GpuVendor::Nvidia, wgpu::DeviceType::DiscreteGpu);
        let hints = MetalRenderHints::detect(&survey);
        let ops = RenderPassOptimizer::gbuffer_ops(&hints);

        assert_eq!(ops.store, wgpu::StoreOp::Store);
    }
}
