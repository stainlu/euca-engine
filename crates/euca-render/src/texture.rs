/// Handle referencing a GPU-uploaded texture by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

/// GPU texture storage, managing uploaded textures and their views.
///
/// A default 1x1 white texture is always available at index 0
/// ([`TextureStore::default_white`]) and is used as the fallback when a
/// material has no albedo texture assigned.
pub struct TextureStore<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    textures: Vec<GpuTexture<D>>,
}

struct GpuTexture<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pub view: D::TextureView,
    #[allow(dead_code)]
    pub texture: D::Texture,
}

// ---------------------------------------------------------------------------
// Generic implementation (works for ANY backend)
// ---------------------------------------------------------------------------

impl<D: euca_rhi::RenderDevice> TextureStore<D> {
    /// Get the texture view for a previously uploaded texture.
    ///
    /// # Panics
    ///
    /// Panics if `handle` does not correspond to a valid uploaded texture.
    pub fn view(&self, handle: TextureHandle) -> &D::TextureView {
        &self.textures[handle.0 as usize].view
    }
}

// ---------------------------------------------------------------------------
// wgpu-specific backward-compatibility methods
// ---------------------------------------------------------------------------

impl TextureStore {
    /// The default white texture handle (always index 0).
    pub fn default_white() -> TextureHandle {
        TextureHandle(0)
    }

    /// Create a new store with a default 1×1 white texture at index 0.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let mut store = Self {
            textures: Vec::new(),
        };
        // Index 0: default white texture (used when material has no albedo texture)
        store.upload_rgba(device, queue, 1, 1, &[255, 255, 255, 255]);
        store
    }

    /// Upload raw RGBA8 pixel data as a 2D texture with auto-generated mipmaps.
    pub fn upload_rgba(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> TextureHandle {
        let mip_level_count = 1 + (width.max(height) as f32).log2().floor() as u32;

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture"),
            size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload base mip (level 0)
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        // Generate remaining mip levels on CPU (box filter downscale)
        let mut prev_data = rgba.to_vec();
        let mut mip_w = width;
        let mut mip_h = height;

        for level in 1..mip_level_count {
            let new_w = (mip_w / 2).max(1);
            let new_h = (mip_h / 2).max(1);
            let mut new_data = vec![0u8; (new_w * new_h * 4) as usize];

            // Box filter: average 2x2 blocks
            for y in 0..new_h {
                for x in 0..new_w {
                    let dst = ((y * new_w + x) * 4) as usize;
                    let sx = (x * 2).min(mip_w - 1);
                    let sy = (y * 2).min(mip_h - 1);

                    for c in 0..4u32 {
                        let s00 = prev_data[((sy * mip_w + sx) * 4 + c) as usize] as u32;
                        let s10 = prev_data
                            [(((sy) * mip_w + (sx + 1).min(mip_w - 1)) * 4 + c) as usize]
                            as u32;
                        let s01 = prev_data
                            [(((sy + 1).min(mip_h - 1) * mip_w + sx) * 4 + c) as usize]
                            as u32;
                        let s11 = prev_data[(((sy + 1).min(mip_h - 1) * mip_w
                            + (sx + 1).min(mip_w - 1))
                            * 4
                            + c) as usize] as u32;
                        new_data[dst + c as usize] = ((s00 + s10 + s01 + s11) / 4) as u8;
                    }
                }
            }

            let mip_size = wgpu::Extent3d {
                width: new_w,
                height: new_h,
                depth_or_array_layers: 1,
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &new_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * new_w),
                    rows_per_image: Some(new_h),
                },
                mip_size,
            );

            prev_data = new_data;
            mip_w = new_w;
            mip_h = new_h;
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(GpuTexture { view, texture });
        handle
    }

    /// Upload a pre-compressed texture (BC7, BC5, etc.) with explicit format.
    ///
    /// For compressed formats, the data must already be block-compressed.
    /// No mip generation is performed — provide all mip levels in `data`.
    /// `bytes_per_row` must account for block size (e.g., BC7 = ceil(w/4)*16).
    // clippy::too_many_arguments — GPU texture upload requires device, queue,
    // dimensions, format, data, and bytes_per_row; all are independent
    // parameters dictated by the wgpu API.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_compressed(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        data: &[u8],
        bytes_per_row: u32,
    ) -> TextureHandle {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Compressed Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(GpuTexture { view, texture });
        handle
    }

    /// Upload an image file (PNG, JPEG, etc.) as a texture.
    pub fn upload_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
    ) -> TextureHandle {
        let img = image::load_from_memory(data)
            .expect("Failed to decode image")
            .to_rgba8();
        self.upload_rgba(device, queue, img.width(), img.height(), &img)
    }

    /// Generate a checkerboard texture for testing.
    pub fn checkerboard(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
        tile: u32,
    ) -> TextureHandle {
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
        self.upload_rgba(device, queue, size, size, &rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_white_handle_is_zero() {
        assert_eq!(TextureStore::default_white(), TextureHandle(0));
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
