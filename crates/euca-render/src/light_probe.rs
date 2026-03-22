//! Light probes using L2 Spherical Harmonics for indirect diffuse lighting.
//!
//! Each probe stores 9 SH coefficients (3 bands) per RGB channel. At runtime,
//! the nearest probes are interpolated and sent to the PBR shader to replace
//! the flat ambient light with spatially-varying indirect illumination.

use euca_math::Vec3;

/// L2 Spherical Harmonics probe — 9 coefficients × 3 channels (RGB).
#[derive(Clone, Debug)]
pub struct LightProbe {
    /// World position of this probe.
    pub position: Vec3,
    /// SH coefficients: `sh[band][channel]` where band ∈ 0..9, channel ∈ {R,G,B}.
    pub sh: [[f32; 3]; 9],
}

impl LightProbe {
    /// Create a probe with uniform ambient color (all directions equal).
    pub fn uniform(position: Vec3, color: [f32; 3]) -> Self {
        // L0 band (DC term) coefficient for uniform irradiance:
        // Y_00 = 1 / (2√π) ≈ 0.282095
        // To reproduce color C uniformly: c_00 = C * π (convolution with cosine lobe)
        let scale = std::f32::consts::PI;
        let mut sh = [[0.0f32; 3]; 9];
        sh[0] = [color[0] * scale, color[1] * scale, color[2] * scale];
        Self { position, sh }
    }

    /// Create a probe from a directional light approximation.
    /// Useful for baking a single dominant light into SH.
    pub fn from_directional(position: Vec3, direction: Vec3, color: [f32; 3]) -> Self {
        let d = direction.normalize();
        let mut sh = [[0.0f32; 3]; 9];

        // SH basis functions evaluated at direction d
        let basis = [
            0.282095,                           // Y_00 (L0)
            0.488603 * d.y,                     // Y_1-1
            0.488603 * d.z,                     // Y_10
            0.488603 * d.x,                     // Y_11
            1.092548 * d.x * d.y,               // Y_2-2
            1.092548 * d.y * d.z,               // Y_2-1
            0.315392 * (3.0 * d.z * d.z - 1.0), // Y_20
            1.092548 * d.x * d.z,               // Y_21
            0.546274 * (d.x * d.x - d.y * d.y), // Y_22
        ];

        for (i, b) in basis.iter().enumerate() {
            sh[i] = [color[0] * b, color[1] * b, color[2] * b];
        }

        Self { position, sh }
    }
}

/// Grid of baked light probes for a region of the scene.
#[derive(Clone, Debug)]
pub struct LightProbeGrid {
    pub probes: Vec<LightProbe>,
}

impl LightProbeGrid {
    pub fn new() -> Self {
        Self { probes: Vec::new() }
    }

    /// Add a probe to the grid.
    pub fn add(&mut self, probe: LightProbe) {
        self.probes.push(probe);
    }

    /// Find the nearest probe and return its SH coefficients.
    /// For better quality, this should interpolate between multiple probes.
    pub fn sample_nearest(&self, world_pos: Vec3) -> Option<&[[f32; 3]; 9]> {
        self.probes
            .iter()
            .min_by(|a, b| {
                let da = (a.position - world_pos).length_squared();
                let db = (b.position - world_pos).length_squared();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| &p.sh)
    }

    /// Pack the N nearest probes into GPU-ready format.
    /// Returns (positions, sh_coefficients, weights, count).
    pub fn pack_nearest(
        &self,
        world_pos: Vec3,
        max_probes: usize,
    ) -> (Vec<[f32; 4]>, Vec<[[f32; 4]; 9]>, Vec<f32>) {
        if self.probes.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        // Sort by distance
        let mut indexed: Vec<(usize, f32)> = self
            .probes
            .iter()
            .enumerate()
            .map(|(i, p)| (i, (p.position - world_pos).length_squared()))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(max_probes);

        // Compute distance-based weights (inverse distance, normalized)
        let total_inv_dist: f32 = indexed
            .iter()
            .map(|(_, d2)| 1.0 / (d2.sqrt() + 0.001))
            .sum();

        let mut positions = Vec::new();
        let mut sh_data = Vec::new();
        let mut weights = Vec::new();

        for (idx, dist_sq) in &indexed {
            let probe = &self.probes[*idx];
            positions.push([probe.position.x, probe.position.y, probe.position.z, 0.0]);

            // Convert [f32; 3] SH to [f32; 4] for GPU alignment
            let mut sh_gpu = [[0.0f32; 4]; 9];
            for (i, coeffs) in probe.sh.iter().enumerate() {
                sh_gpu[i] = [coeffs[0], coeffs[1], coeffs[2], 0.0];
            }
            sh_data.push(sh_gpu);

            let w = (1.0 / (dist_sq.sqrt() + 0.001)) / total_inv_dist;
            weights.push(w);
        }

        (positions, sh_data, weights)
    }
}

impl Default for LightProbeGrid {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate L2 SH at a given normal direction (CPU-side, for testing/baking).
pub fn evaluate_sh(normal: Vec3, sh: &[[f32; 3]; 9]) -> [f32; 3] {
    let n = normal;
    let mut result = [0.0f32; 3];

    let basis = [
        0.282095,
        0.488603 * n.y,
        0.488603 * n.z,
        0.488603 * n.x,
        1.092548 * n.x * n.y,
        1.092548 * n.y * n.z,
        0.315392 * (3.0 * n.z * n.z - 1.0),
        1.092548 * n.x * n.z,
        0.546274 * (n.x * n.x - n.y * n.y),
    ];

    for (i, b) in basis.iter().enumerate() {
        result[0] += sh[i][0] * b;
        result[1] += sh[i][1] * b;
        result[2] += sh[i][2] * b;
    }

    [result[0].max(0.0), result[1].max(0.0), result[2].max(0.0)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_probe_evaluates_equally() {
        let probe = LightProbe::uniform(Vec3::ZERO, [0.5, 0.5, 0.5]);
        let up = evaluate_sh(Vec3::Y, &probe.sh);
        let right = evaluate_sh(Vec3::X, &probe.sh);
        // Uniform probe should give similar values in all directions
        // (not exactly equal due to L0-only approximation, but close)
        assert!((up[0] - right[0]).abs() < 0.01, "up={up:?} right={right:?}");
    }

    #[test]
    fn directional_probe_brightest_in_direction() {
        let dir = Vec3::new(0.0, 1.0, 0.0);
        let probe = LightProbe::from_directional(Vec3::ZERO, dir, [1.0, 1.0, 1.0]);

        let along = evaluate_sh(Vec3::Y, &probe.sh);
        let against = evaluate_sh(Vec3::new(0.0, -1.0, 0.0), &probe.sh);

        // Evaluating SH in the light direction should be brighter than opposite
        assert!(
            along[0] > against[0],
            "along={along:?} should be brighter than against={against:?}"
        );
    }

    #[test]
    fn grid_nearest_probe() {
        let mut grid = LightProbeGrid::new();
        grid.add(LightProbe::uniform(
            Vec3::new(0.0, 0.0, 0.0),
            [1.0, 0.0, 0.0],
        ));
        grid.add(LightProbe::uniform(
            Vec3::new(10.0, 0.0, 0.0),
            [0.0, 1.0, 0.0],
        ));

        // Query near first probe
        let sh = grid.sample_nearest(Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let color = evaluate_sh(Vec3::Y, sh);
        // Should be reddish (first probe)
        assert!(color[0] > color[1], "Expected red probe, got {color:?}");
    }
}
