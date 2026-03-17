//! Deferred rendering infrastructure.
//!
//! Adds G-buffer render targets and a deferred lighting pass alongside
//! the existing forward renderer. The renderer can be configured to use
//! either path via `RenderPath`.
//!
//! # G-Buffer layout
//! - RT0: Rgba16Float — world-space position (xyz) + depth (w)
//! - RT1: Rgba8Unorm — normal (xy, octahedral encoded) + metallic (z) + roughness (w)
//! - RT2: Rgba8UnormSrgb — albedo (rgb) + ao (a)
//! - Depth: Depth32Float (shared with forward pass)
//!
//! # Architecture
//! 1. G-buffer pass: render geometry, output material properties to render targets
//! 2. Lighting pass: fullscreen quad reads G-buffer, computes PBR lighting
//! 3. Forward pass: render transparent objects on top (future)

/// Which rendering path the engine uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderPath {
    /// Traditional forward rendering (current default).
    /// Good for scenes with few lights and lots of transparency.
    Forward,
    /// Deferred rendering with G-buffer.
    /// Good for scenes with many lights (100+). Transparency requires extra pass.
    Deferred,
}

impl Default for RenderPath {
    fn default() -> Self {
        Self::Forward
    }
}

/// G-buffer format definitions.
pub struct GBufferFormats;

impl GBufferFormats {
    /// Position + linear depth
    pub const POSITION: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    /// Normal (octahedral) + metallic + roughness
    pub const NORMAL_MATERIAL: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    /// Albedo (sRGB) + ambient occlusion
    pub const ALBEDO: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
}

/// G-buffer render targets.
pub struct GBuffer {
    pub position_texture: wgpu::Texture,
    pub position_view: wgpu::TextureView,
    pub normal_material_texture: wgpu::Texture,
    pub normal_material_view: wgpu::TextureView,
    pub albedo_texture: wgpu::Texture,
    pub albedo_view: wgpu::TextureView,
}

impl GBuffer {
    /// Create G-buffer render targets for the given surface dimensions.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let create_rt = |label: &str, format: wgpu::TextureFormat| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };

        let (position_texture, position_view) =
            create_rt("G-Buffer Position", GBufferFormats::POSITION);
        let (normal_material_texture, normal_material_view) =
            create_rt("G-Buffer Normal+Material", GBufferFormats::NORMAL_MATERIAL);
        let (albedo_texture, albedo_view) = create_rt("G-Buffer Albedo", GBufferFormats::ALBEDO);

        Self {
            position_texture,
            position_view,
            normal_material_texture,
            normal_material_view,
            albedo_texture,
            albedo_view,
        }
    }

    /// Resize G-buffer to match new surface dimensions.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        *self = Self::new(device, width, height);
    }
}
