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
    /// Returns `true` if this mode uses alpha blending (i.e., [`AlphaMode::Blend`]).
    pub fn is_transparent(&self) -> bool {
        matches!(self, AlphaMode::Blend)
    }

    /// Encode the alpha mode as a float for GPU uniform upload.
    ///
    /// `Opaque` = 0.0, `Mask` = 1.0, `Blend` = 2.0.
    pub fn as_f32(&self) -> f32 {
        match self {
            AlphaMode::Opaque => 0.0,
            AlphaMode::Mask { .. } => 1.0,
            AlphaMode::Blend => 2.0,
        }
    }

    /// Returns the alpha cutoff threshold.
    ///
    /// For [`AlphaMode::Mask`] this is the user-specified value; for other modes
    /// a default of `0.5` is returned (unused by the shader).
    pub fn cutoff(&self) -> f32 {
        match self {
            AlphaMode::Mask { cutoff } => *cutoff,
            _ => 0.5,
        }
    }
}

/// Physically-based rendering (PBR) material definition.
///
/// Describes surface appearance using the metallic-roughness workflow.
/// Each field can optionally be driven by a texture; the texture value is
/// multiplied with the constant to produce the final shading input.
#[derive(Clone, Debug)]
pub struct Material {
    /// Base color (RGBA, linear). The alpha channel is used when
    /// `alpha_mode` is [`AlphaMode::Mask`] or [`AlphaMode::Blend`].
    pub albedo: [f32; 4],
    /// Metallic factor in `[0.0, 1.0]`. `0.0` = dielectric, `1.0` = metal.
    pub metallic: f32,
    /// Roughness factor in `[0.0, 1.0]`. `0.0` = mirror-smooth, `1.0` = fully rough.
    pub roughness: f32,
    /// Optional albedo (base color) texture, sampled and multiplied with `albedo`.
    pub albedo_texture: Option<TextureHandle>,
    /// Optional tangent-space normal map for per-pixel surface detail.
    pub normal_texture: Option<TextureHandle>,
    /// Emissive color (linear RGB). Added on top of reflected light.
    pub emissive: [f32; 3],
    /// Optional emissive texture, sampled and multiplied with `emissive`.
    pub emissive_texture: Option<TextureHandle>,
    /// Optional combined metallic-roughness texture (green = roughness,
    /// blue = metallic, following glTF convention).
    pub metallic_roughness_texture: Option<TextureHandle>,
    /// Optional ambient occlusion texture (red channel). Attenuates indirect
    /// lighting to approximate self-shadowing in crevices.
    pub ao_texture: Option<TextureHandle>,
    /// Controls how the renderer treats this material's alpha channel.
    pub alpha_mode: AlphaMode,
}

impl Material {
    /// Create a new material with the given albedo color, metallic, and roughness.
    ///
    /// All texture slots default to `None`, emissive to black, and alpha mode
    /// to [`AlphaMode::Opaque`].
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
    /// Builder: attach a tangent-space normal map texture.
    pub fn with_normal_map(mut self, texture: TextureHandle) -> Self {
        self.normal_texture = Some(texture);
        self
    }

    /// Builder: attach an albedo (base color) texture.
    pub fn with_texture(mut self, texture: TextureHandle) -> Self {
        self.albedo_texture = Some(texture);
        self
    }

    /// Builder: set the emissive color (linear RGB).
    pub fn with_emissive(mut self, emissive: [f32; 3]) -> Self {
        self.emissive = emissive;
        self
    }

    /// Builder: attach an emissive texture.
    pub fn with_emissive_texture(mut self, texture: TextureHandle) -> Self {
        self.emissive_texture = Some(texture);
        self
    }

    /// Builder: attach a combined metallic-roughness texture.
    pub fn with_metallic_roughness_texture(mut self, texture: TextureHandle) -> Self {
        self.metallic_roughness_texture = Some(texture);
        self
    }

    /// Builder: attach an ambient occlusion texture.
    pub fn with_ao_texture(mut self, texture: TextureHandle) -> Self {
        self.ao_texture = Some(texture);
        self
    }

    /// Builder: set the alpha blending mode.
    pub fn with_alpha_mode(mut self, mode: AlphaMode) -> Self {
        self.alpha_mode = mode;
        self
    }

    /// Preset: a glossy red dielectric (plastic) material.
    pub fn red_plastic() -> Self {
        Self::new([0.9, 0.1, 0.1, 1.0], 0.0, 0.7)
    }

    /// Preset: a smooth blue dielectric (plastic) material.
    pub fn blue_plastic() -> Self {
        Self::new([0.1, 0.2, 0.9, 1.0], 0.0, 0.3)
    }

    /// Preset: a polished gold metallic material.
    pub fn gold() -> Self {
        Self::new([1.0, 0.84, 0.0, 1.0], 1.0, 0.2)
    }

    /// Preset: a brushed silver metallic material.
    pub fn silver() -> Self {
        Self::new([0.95, 0.95, 0.95, 1.0], 1.0, 0.4)
    }

    /// Preset: a rough gray dielectric material.
    pub fn gray() -> Self {
        Self::new([0.5, 0.5, 0.5, 1.0], 0.0, 0.9)
    }

    /// Preset: a medium-roughness green dielectric material.
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
    /// Create a white material with an albedo texture applied.
    ///
    /// Useful for textured meshes where the color comes entirely from the
    /// texture rather than a constant albedo tint.
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

/// Component that associates an entity with a GPU-uploaded material.
#[derive(Clone, Copy, Debug, euca_reflect::Reflect)]
pub struct MaterialRef {
    /// The handle of the material to use when rendering this entity.
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
