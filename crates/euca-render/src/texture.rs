/// Handle referencing a GPU-uploaded texture by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

/// Stores GPU textures and their views.
pub struct TextureStore {
    textures: Vec<GpuTexture>,
}

struct GpuTexture {
    pub view: wgpu::TextureView,
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
}

impl TextureStore {
    /// Create a new store with a default 1×1 white texture at index 0.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let mut store = Self {
            textures: Vec::new(),
        };
        // Index 0: default white texture (used when material has no albedo texture)
        store.upload_rgba(device, queue, 1, 1, &[255, 255, 255, 255]);
        store
    }

    /// Upload raw RGBA8 pixel data as a 2D texture.
    pub fn upload_rgba(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> TextureHandle {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
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

    /// Get the texture view for a handle.
    pub fn view(&self, handle: TextureHandle) -> &wgpu::TextureView {
        &self.textures[handle.0 as usize].view
    }

    /// The default white texture handle (always index 0).
    pub fn default_white() -> TextureHandle {
        TextureHandle(0)
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
