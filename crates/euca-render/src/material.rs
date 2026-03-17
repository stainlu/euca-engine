use crate::texture::TextureHandle;

/// PBR material properties.
#[derive(Clone, Debug)]
pub struct Material {
    pub albedo: [f32; 4],                      // Base color RGBA
    pub metallic: f32,                         // 0.0 = dielectric, 1.0 = metal
    pub roughness: f32,                        // 0.0 = mirror smooth, 1.0 = fully rough
    pub albedo_texture: Option<TextureHandle>, // Optional albedo texture (sampled × color)
}

impl Material {
    pub fn new(albedo: [f32; 4], metallic: f32, roughness: f32) -> Self {
        Self {
            albedo,
            metallic,
            roughness,
            albedo_texture: None,
        }
    }

    /// Set the albedo texture for this material.
    pub fn with_texture(mut self, texture: TextureHandle) -> Self {
        self.albedo_texture = Some(texture);
        self
    }

    /// Matte red plastic.
    pub fn red_plastic() -> Self {
        Self::new([0.9, 0.1, 0.1, 1.0], 0.0, 0.7)
    }

    /// Shiny blue plastic.
    pub fn blue_plastic() -> Self {
        Self::new([0.1, 0.2, 0.9, 1.0], 0.0, 0.3)
    }

    /// Polished gold metal.
    pub fn gold() -> Self {
        Self::new([1.0, 0.84, 0.0, 1.0], 1.0, 0.2)
    }

    /// Brushed silver metal.
    pub fn silver() -> Self {
        Self::new([0.95, 0.95, 0.95, 1.0], 1.0, 0.4)
    }

    /// Matte gray (good for ground planes).
    pub fn gray() -> Self {
        Self::new([0.5, 0.5, 0.5, 1.0], 0.0, 0.9)
    }

    /// Matte green.
    pub fn green() -> Self {
        Self::new([0.2, 0.8, 0.2, 1.0], 0.0, 0.6)
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new([0.8, 0.8, 0.8, 1.0], 0.0, 0.5)
    }
}

/// Builder-style texture assignment.
impl Material {
    /// Create a material with a texture (color acts as tint multiplier).
    pub fn textured(texture: TextureHandle) -> Self {
        Self {
            albedo: [1.0, 1.0, 1.0, 1.0], // white tint = pure texture color
            metallic: 0.0,
            roughness: 0.5,
            albedo_texture: Some(texture),
        }
    }
}

/// Handle referencing a GPU-uploaded material.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialHandle(pub u32);

/// ECS component: which material to use for rendering.
#[derive(Clone, Copy, Debug, euca_reflect::Reflect)]
pub struct MaterialRef {
    pub handle: MaterialHandle,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_defaults() {
        let m = Material::default();
        assert_eq!(m.metallic, 0.0);
        assert!(m.roughness > 0.0);
        assert_eq!(m.albedo[3], 1.0); // alpha = 1
    }

    #[test]
    fn material_presets() {
        let gold = Material::gold();
        assert_eq!(gold.metallic, 1.0); // metal

        let plastic = Material::red_plastic();
        assert_eq!(plastic.metallic, 0.0); // dielectric
        assert!(plastic.albedo[0] > 0.5); // red channel high
    }
}
