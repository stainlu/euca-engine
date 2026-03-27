//! Native Metal backend stub (Phase B).
//!
//! This module will provide a `MetalDevice` implementing [`RenderDevice`]
//! using `objc2-metal` for direct Metal API access on Apple Silicon.
//! Currently a compile-time placeholder — all methods panic with `todo!()`.
//!
//! Phase B will implement:
//! - `MTLDevice` / `MTLCommandQueue` lifecycle
//! - Resource creation via `MTLBuffer`, `MTLTexture`, `MTLSamplerState`
//! - Render/compute pipeline state compilation from MSL
//! - `CAMetalLayer` surface management
//! - `MTLCommandBuffer` / `MTLRenderCommandEncoder` command encoding

/// Placeholder for the native Metal GPU backend.
///
/// Will be implemented in Phase B using `objc2-metal` and `objc2-foundation`.
pub struct MetalDevice {
    _private: (),
}

impl MetalDevice {
    /// Phase B: create a MetalDevice from a `CAMetalLayer` and system GPU.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        todo!("MetalDevice::new — Phase B: native Metal backend")
    }
}
