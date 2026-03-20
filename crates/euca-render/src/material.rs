use crate::texture::TextureHandle;

/// How the renderer handles alpha for this material.
#[derive(Clone, Debug, PartialEq)]
pub enum AlphaMode {
    /// Fully opaque -- alpha is ignored.
    Opaque,
    /// Alpha testing: fragments with alpha < cutoff are discarded.
    Mask { cutoff: f32 },
    /// Alpha blending: fragments are composited using source alpha.
    Blend,
}

impl Default for AlphaMode {
    fn default() -> Self {
        Self::Opaque
    }
}

impl AlphaMode {
    pub fn is_transparent(&self) -> bool {
        matches!(self, AlphaMode::Blend)
    }
    pub fn as_f32(&self) -> f32 {
        match self {
            AlphaMode::Opaque => 0.0,
            AlphaMode::Mask { .. } => 1.0,
            AlphaMode::Blend => 2.0,
        }
    }
    pub fn cutoff(&self) -> f32 {
        match self {
            AlphaMode::Mask { cutoff } => *cutoff,
            _ => 0.5,
        }
    }
}

/// PBR material properties.
#[derive(Clone, Debug)]
pub struct Material {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub albedo_texture: Option<TextureHandle>,
    pub normal_texture: Option<TextureHandle>,
    pub emissive: [f32; 3],
    pub emissive_texture: Option<TextureHandle>,
    pub metallic_roughness_texture: Option<TextureHandle>,
    pub ao_texture: Option<TextureHandle>,
    pub alpha_mode: AlphaMode,
}

impl Material {
    pub fn new(albedo: [f32; 4], metallic: f32, roughness: f32) -> Self {
        Self {
            albedo,
            metallic,
            roughness,
            albedo_texture: None,
            normal_texture: None,
            emissive: [0.0; 3],
            emissive_texture: None,
            metallic_roughness_texture: None,
            ao_texture: None,
            alpha_mode: AlphaMode::Opaque,
        }
    }
    pub fn with_normal_map(mut self, texture: TextureHandle) -> Self {
        self.normal_texture = Some(texture);
        self
    }
    pub fn with_texture(mut self, texture: TextureHandle) -> Self {
        self.albedo_texture = Some(texture);
        self
    }
    pub fn with_emissive(mut self, emissive: [f32; 3]) -> Self {
        self.emissive = emissive;
        self
    }
    pub fn with_emissive_texture(mut self, texture: TextureHandle) -> Self {
        self.emissive_texture = Some(texture);
        self
    }
    pub fn with_metallic_roughness_texture(mut self, texture: TextureHandle) -> Self {
        self.metallic_roughness_texture = Some(texture);
        self
    }
    pub fn with_ao_texture(mut self, texture: TextureHandle) -> Self {
        self.ao_texture = Some(texture);
        self
    }
    pub fn with_alpha_mode(mut self, mode: AlphaMode) -> Self {
        self.alpha_mode = mode;
        self
    }
    pub fn red_plastic() -> Self {
        Self::new([0.9, 0.1, 0.1, 1.0], 0.0, 0.7)
    }
    pub fn blue_plastic() -> Self {
        Self::new([0.1, 0.2, 0.9, 1.0], 0.0, 0.3)
    }
    pub fn gold() -> Self {
        Self::new([1.0, 0.84, 0.0, 1.0], 1.0, 0.2)
    }
    pub fn silver() -> Self {
        Self::new([0.95, 0.95, 0.95, 1.0], 1.0, 0.4)
    }
    pub fn gray() -> Self {
        Self::new([0.5, 0.5, 0.5, 1.0], 0.0, 0.9)
    }
    pub fn green() -> Self {
        Self::new([0.2, 0.8, 0.2, 1.0], 0.0, 0.6)
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new([0.8, 0.8, 0.8, 1.0], 0.0, 0.5)
    }
}

impl Material {
    pub fn textured(texture: TextureHandle) -> Self {
        Self {
            albedo: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            albedo_texture: Some(texture),
            normal_texture: None,
            emissive: [0.0; 3],
            emissive_texture: None,
            metallic_roughness_texture: None,
            ao_texture: None,
            alpha_mode: AlphaMode::Opaque,
        }
    }
}

/// Handle referencing a GPU-uploaded material.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, euca_reflect::Reflect)]
pub struct MaterialHandle(pub u32);

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
        assert_eq!(m.albedo[3], 1.0);
        assert_eq!(m.emissive, [0.0; 3]);
        assert_eq!(m.alpha_mode, AlphaMode::Opaque);
    }

    #[test]
    fn material_presets() {
        let gold = Material::gold();
        assert_eq!(gold.metallic, 1.0);
        let plastic = Material::red_plastic();
        assert_eq!(plastic.metallic, 0.0);
        assert!(plastic.albedo[0] > 0.5);
    }

    #[test]
    fn alpha_mode_encoding() {
        assert_eq!(AlphaMode::Opaque.as_f32(), 0.0);
        assert_eq!(AlphaMode::Mask { cutoff: 0.5 }.as_f32(), 1.0);
        assert_eq!(AlphaMode::Blend.as_f32(), 2.0);
    }

    #[test]
    fn alpha_mode_transparency() {
        assert!(!AlphaMode::Opaque.is_transparent());
        assert!(!AlphaMode::Mask { cutoff: 0.5 }.is_transparent());
        assert!(AlphaMode::Blend.is_transparent());
    }

    #[test]
    fn builder_chain() {
        let m = Material::new([1.0, 1.0, 1.0, 0.5], 0.0, 0.5)
            .with_emissive([1.0, 0.5, 0.0])
            .with_alpha_mode(AlphaMode::Blend);
        assert_eq!(m.emissive, [1.0, 0.5, 0.0]);
        assert!(m.alpha_mode.is_transparent());
    }
}
