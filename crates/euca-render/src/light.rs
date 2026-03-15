/// Directional light component (infinite distance, parallel rays — like the sun).
#[derive(Clone, Debug)]
pub struct DirectionalLight {
    /// Direction the light is shining (normalized).
    pub direction: [f32; 3],
    /// Light color (linear RGB).
    pub color: [f32; 3],
    /// Intensity multiplier.
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: [0.3, -1.0, 0.5],  // From upper-right
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
        }
    }
}

/// Ambient light resource (global fill light).
#[derive(Clone, Debug)]
pub struct AmbientLight {
    pub color: [f32; 3],
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
