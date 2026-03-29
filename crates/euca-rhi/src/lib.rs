//! Rendering Hardware Interface (RHI) — backend-agnostic GPU abstraction.
//!
//! This crate defines the [`RenderDevice`] trait that abstracts over GPU
//! backends (wgpu, native Metal, etc.). The engine's renderer is generic
//! over this trait, enabling compile-time backend selection.
//!
//! # Backends
//!
//! - **wgpu** (default) — cross-platform via the `wgpu-backend` feature.
//! - **Metal** (macOS) — native Metal via `objc2-metal` (Phase B).

pub mod pass;
pub mod types;

#[cfg(feature = "wgpu-backend")]
pub mod wgpu_backend;

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
pub mod metal_backend;

pub use pass::{ComputePassOps, RenderPassOps};
pub use types::*;

/// Backend-agnostic GPU device trait.
///
/// Implementations wrap a platform-specific GPU device (wgpu, Metal, Vulkan)
/// and provide resource creation, data upload, command encoding, and surface
/// management through a uniform interface.
///
/// Associated types are opaque GPU handles. The renderer stores and passes
/// them around but never inspects their internals — only the backend knows
/// how to use them.
pub trait RenderDevice: Send + Sync + 'static {
    // -- Opaque GPU handle types --
    type Buffer: Send + Sync;
    type Texture: Send + Sync;
    type TextureView: Send + Sync;
    type Sampler: Send + Sync;
    type BindGroupLayout: Send + Sync;
    type BindGroup: Send + Sync;
    type ShaderModule: Send + Sync;
    type RenderPipeline: Send + Sync;
    type ComputePipeline: Send + Sync;
    type CommandEncoder;
    type RenderPass<'a>: RenderPassOps<Self>
    where
        Self: 'a;
    type ComputePass<'a>: ComputePassOps<Self>
    where
        Self: 'a;
    type SurfaceTexture;

    // -- Capabilities --

    /// Query GPU capabilities (feature flags, limits).
    fn capabilities(&self) -> &Capabilities;

    // -- Resource creation --

    fn create_buffer(&self, desc: &BufferDesc) -> Self::Buffer;
    fn create_texture(&self, desc: &TextureDesc) -> Self::Texture;
    fn create_texture_view(
        &self,
        texture: &Self::Texture,
        desc: &TextureViewDesc,
    ) -> Self::TextureView;
    fn create_sampler(&self, desc: &SamplerDesc) -> Self::Sampler;
    fn create_shader(&self, desc: &ShaderDesc) -> Self::ShaderModule;
    fn create_bind_group_layout(&self, desc: &BindGroupLayoutDesc) -> Self::BindGroupLayout;
    fn create_bind_group(&self, desc: &BindGroupDesc<Self>) -> Self::BindGroup;
    fn create_render_pipeline(&self, desc: &RenderPipelineDesc<Self>) -> Self::RenderPipeline;
    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc<Self>) -> Self::ComputePipeline;

    // -- Data upload --

    fn write_buffer(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]);
    fn write_texture(
        &self,
        dst: &TexelCopyTextureInfo<Self>,
        data: &[u8],
        layout: &TextureDataLayout,
        size: Extent3d,
    );

    // -- Command encoding --

    fn create_command_encoder(&self, label: Option<&str>) -> Self::CommandEncoder;

    fn begin_render_pass<'a>(
        &self,
        encoder: &'a mut Self::CommandEncoder,
        desc: &RenderPassDesc<'_, Self>,
    ) -> Self::RenderPass<'a>;

    fn begin_compute_pass<'a>(
        &self,
        encoder: &'a mut Self::CommandEncoder,
        label: Option<&str>,
    ) -> Self::ComputePass<'a>;

    fn clear_buffer(
        &self,
        encoder: &mut Self::CommandEncoder,
        buffer: &Self::Buffer,
        offset: u64,
        size: Option<u64>,
    );

    fn copy_texture_to_texture(
        &self,
        encoder: &mut Self::CommandEncoder,
        src: &TexelCopyTextureInfo<Self>,
        dst: &TexelCopyTextureInfo<Self>,
        size: Extent3d,
    );

    fn submit(&self, encoder: Self::CommandEncoder);

    /// Submit multiple command encoders in a single batch.
    ///
    /// The GPU executes them in order within the submission, but the driver
    /// may overlap independent work between encoders. This enables splitting
    /// compute and render work into separate encoders for async compute.
    fn submit_multiple(&self, encoders: Vec<Self::CommandEncoder>);

    // -- Surface management --

    fn get_current_texture(&self) -> Result<Self::SurfaceTexture, SurfaceError>;

    /// Create a default view of the surface texture for rendering.
    fn surface_texture_view(&self, surface_texture: &Self::SurfaceTexture) -> Self::TextureView;

    fn present(&self, texture: Self::SurfaceTexture);
    fn resize_surface(&mut self, width: u32, height: u32);
    fn surface_format(&self) -> TextureFormat;
    fn surface_size(&self) -> (u32, u32);

    fn aspect_ratio(&self) -> f32 {
        let (w, h) = self.surface_size();
        w as f32 / h as f32
    }
}
