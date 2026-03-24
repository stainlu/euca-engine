/// Directional light component (infinite distance, parallel rays — like the sun).
#[derive(Clone, Debug)]
pub struct DirectionalLight {
    /// Direction the light is shining (normalized).
    pub direction: [f32; 3],
    /// Light color (linear RGB).
    pub color: [f32; 3],
    /// Intensity multiplier.
    pub intensity: f32,
    /// Apparent size of the light source for PCSS soft shadow penumbra.
    ///
    /// Larger values produce wider, softer shadow penumbrae. A value of 0.0
    /// effectively disables PCSS and produces hard shadow edges identical to
    /// standard PCF. Reasonable values for a sun-like light are 0.5–2.0.
    pub light_size: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: [0.3, -1.0, 0.5], // From upper-right
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            light_size: 1.0,
        }
    }
}

/// Point light component (emits in all directions from a position).
///
/// Position is determined by the entity's transform component, not stored here.
#[derive(Clone, Debug)]
pub struct PointLight {
    /// Light color (linear RGB).
    pub color: [f32; 3],
    /// Intensity multiplier.
    pub intensity: f32,
    /// Maximum range. Light attenuates to zero at this distance.
    pub range: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: 10.0,
        }
    }
}

/// Spot light component (cone-shaped light from a position).
///
/// Position is determined by the entity's transform component.
#[derive(Clone, Debug)]
pub struct SpotLight {
    /// Direction the spotlight is aimed (normalized).
    pub direction: [f32; 3],
    /// Light color (linear RGB).
    pub color: [f32; 3],
    /// Intensity multiplier.
    pub intensity: f32,
    /// Inner cone angle (radians) -- full intensity inside this cone.
    pub inner_cone: f32,
    /// Outer cone angle (radians) -- light falls off to zero at this angle.
    pub outer_cone: f32,
    /// Maximum range. Light attenuates to zero at this distance.
    pub range: f32,
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            direction: [0.0, -1.0, 0.0],
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            inner_cone: 0.3,
            outer_cone: 0.5,
            range: 15.0,
        }
    }
}

/// Ambient light resource (global fill light applied uniformly to all surfaces).
#[derive(Clone, Debug)]
pub struct AmbientLight {
    /// Light color (linear RGB).
    pub color: [f32; 3],
    /// Intensity multiplier.
    pub intensity: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            intensity: 0.15,
        }
    }
}
