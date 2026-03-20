//! Splat map for blending up to 4 terrain texture layers.
//!
//! Each grid cell stores a `Vec4` of blend weights.  At render time these
//! weights are interpolated across the terrain surface and used to mix the
//! four texture layers.

use euca_math::Vec4;

/// Per-cell blend weights for 4 texture layers.
///
/// Dimensions match the heightmap grid.  Each entry is a `Vec4` where
/// `(x, y, z, w)` correspond to layers 0..3 and should sum to 1.0.
#[derive(Clone, Debug)]
pub struct SplatMap {
    pub width: u32,
    pub height: u32,
    /// Row-major `Vec4` weights.
    pub data: Vec<Vec4>,
}

impl SplatMap {
    /// Create a splat map where layer 0 has weight 1.0 everywhere.
    pub fn uniform(width: u32, height: u32) -> Self {
        let default_weight = Vec4::new(1.0, 0.0, 0.0, 0.0);
        Self {
            width,
            height,
            data: vec![default_weight; (width as usize) * (height as usize)],
        }
    }

    /// Get the blend weights at grid coordinates. Returns uniform layer-0 for
    /// out-of-bounds.
    #[inline]
    pub fn get(&self, col: u32, row: u32) -> Vec4 {
        if col >= self.width || row >= self.height {
            return Vec4::new(1.0, 0.0, 0.0, 0.0);
        }
        self.data[(row as usize) * (self.width as usize) + (col as usize)]
    }

    /// Set the blend weights at grid coordinates. Silently ignores
    /// out-of-bounds writes.
    #[inline]
    pub fn set(&mut self, col: u32, row: u32, weights: Vec4) {
        if col < self.width && row < self.height {
            self.data[(row as usize) * (self.width as usize) + (col as usize)] = weights;
        }
    }

    /// Normalise a weight vector so all components sum to 1.0.
    /// If the sum is effectively zero, falls back to layer 0.
    pub fn normalize_weights(w: Vec4) -> Vec4 {
        let sum = w.x + w.y + w.z + w.w;
        if sum.abs() < 1e-8 {
            return Vec4::new(1.0, 0.0, 0.0, 0.0);
        }
        w * (1.0 / sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_splat_is_layer0() {
        let sm = SplatMap::uniform(4, 4);
        let w = sm.get(2, 2);
        assert!((w.x - 1.0).abs() < 1e-6);
        assert!((w.y).abs() < 1e-6);
    }

    #[test]
    fn normalize_weights_sums_to_one() {
        let raw = Vec4::new(2.0, 3.0, 0.0, 5.0);
        let n = SplatMap::normalize_weights(raw);
        let sum = n.x + n.y + n.z + n.w;
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((n.x - 0.2).abs() < 1e-6);
    }

    #[test]
    fn normalize_zero_falls_back() {
        let n = SplatMap::normalize_weights(Vec4::ZERO);
        assert!((n.x - 1.0).abs() < 1e-6);
    }
}
