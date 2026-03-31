use euca_rhi::RenderDevice;

/// Handle referencing a GPU-uploaded texture by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

impl TextureHandle {
    /// The default white texture handle (always index 0).
    pub const DEFAULT_WHITE: Self = Self(0);
}

/// GPU texture storage, managing uploaded textures and their views.
///
/// A default 1x1 white texture is always available at index 0
/// ([`TextureHandle::DEFAULT_WHITE`]) and is used as the fallback when a
/// material has no albedo texture assigned.
pub struct TextureStore<D: RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    textures: Vec<GpuTexture<D>>,
}

struct GpuTexture<D: RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pub view: D::TextureView,
    #[allow(dead_code)]
    pub texture: D::Texture,
}

impl<D: RenderDevice> TextureStore<D> {
    /// Create a new store with a default 1x1 white texture at index 0.
    pub fn new(device: &D) -> Self {
        let mut store = Self {
            textures: Vec::new(),
        };
        // Index 0: default white texture (used when material has no albedo texture)
        store.upload_rgba(device, 1, 1, &[255, 255, 255, 255]);
        store
    }

    /// Upload raw RGBA8 pixel data as a 2D texture.
    ///
    /// Uses a single mip level (no CPU mipmap generation) for fast loading.
    /// GPU-side mip generation or pre-baked mipmaps from the asset cooker
    /// will replace this in the future.
    pub fn upload_rgba(
        &mut self,
        device: &D,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> TextureHandle {
        let mip_level_count = 1;

        let size = euca_rhi::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&euca_rhi::TextureDesc {
            label: Some("Texture"),
            size,
            mip_level_count,
            sample_count: 1,
            dimension: euca_rhi::TextureDimension::D2,
            format: euca_rhi::TextureFormat::Rgba8UnormSrgb,
            usage: euca_rhi::TextureUsages::TEXTURE_BINDING | euca_rhi::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload base mip (level 0)
        device.write_texture(
            &euca_rhi::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: euca_rhi::Origin3d::default(),
                aspect: euca_rhi::TextureAspect::All,
            },
            rgba,
            &euca_rhi::TextureDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        let view = device.create_texture_view(&texture, &euca_rhi::TextureViewDesc::default());
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(GpuTexture { view, texture });
        handle
    }

    /// Upload a pre-compressed texture (BC7, BC5, etc.) with explicit format.
    ///
    /// For compressed formats, the data must already be block-compressed.
    /// No mip generation is performed -- provide all mip levels in `data`.
    /// `bytes_per_row` must account for block size (e.g., BC7 = ceil(w/4)*16).
    // clippy::too_many_arguments -- GPU texture upload requires device,
    // dimensions, format, data, and bytes_per_row; all are independent
    // parameters dictated by the RHI API.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_compressed(
        &mut self,
        device: &D,
        width: u32,
        height: u32,
        format: euca_rhi::TextureFormat,
        data: &[u8],
        bytes_per_row: u32,
    ) -> TextureHandle {
        let size = euca_rhi::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&euca_rhi::TextureDesc {
            label: Some("Compressed Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: euca_rhi::TextureDimension::D2,
            format,
            usage: euca_rhi::TextureUsages::TEXTURE_BINDING | euca_rhi::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        device.write_texture(
            &euca_rhi::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: euca_rhi::Origin3d::default(),
                aspect: euca_rhi::TextureAspect::All,
            },
            data,
            &euca_rhi::TextureDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            size,
        );
        let view = device.create_texture_view(&texture, &euca_rhi::TextureViewDesc::default());
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(GpuTexture { view, texture });
        handle
    }

    /// Upload an image file (PNG, JPEG, etc.) as a texture.
    pub fn upload_image(&mut self, device: &D, data: &[u8]) -> TextureHandle {
        let img = image::load_from_memory(data)
            .expect("Failed to decode image")
            .to_rgba8();
        self.upload_rgba(device, img.width(), img.height(), &img)
    }

    /// Get the texture view for a previously uploaded texture.
    ///
    /// # Panics
    ///
    /// Panics if `handle` does not correspond to a valid uploaded texture.
    pub fn view(&self, handle: TextureHandle) -> &D::TextureView {
        &self.textures[handle.0 as usize].view
    }

    /// Generate a checkerboard texture for testing.
    pub fn checkerboard(&mut self, device: &D, size: u32, tile: u32) -> TextureHandle {
        let mut rgba = vec![0u8; (size * size * 4) as usize];
        for y in 0..size {
            for x in 0..size {
                let is_white = ((x / tile) + (y / tile)) % 2 == 0;
                let color: u8 = if is_white { 220 } else { 60 };
                let i = ((y * size + x) * 4) as usize;
                rgba[i] = color;
                rgba[i + 1] = color;
                rgba[i + 2] = color;
                rgba[i + 3] = 255;
            }
        }
        self.upload_rgba(device, size, size, &rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_white_handle_is_zero() {
        assert_eq!(TextureHandle::DEFAULT_WHITE, TextureHandle(0));
    }

    #[test]
    fn texture_handle_equality() {
        let a = TextureHandle(1);
        let b = TextureHandle(1);
        let c = TextureHandle(2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
