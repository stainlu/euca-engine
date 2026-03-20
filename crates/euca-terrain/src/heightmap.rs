//! Heightmap resource: stores elevation data in a row-major grid and provides
//! bilinear sampling with surface normal computation.

use euca_math::Vec3;

/// A 2D grid of elevation values.
///
/// Coordinates follow the XZ ground plane convention: `x` maps to columns,
/// `z` maps to rows, and `y` is the up axis.  Heights stored in `data` are
/// normalised to `[0, 1]` and scaled by `max_height` when sampled.
#[derive(Clone, Debug)]
pub struct Heightmap {
    /// Number of columns (X axis).
    pub width: u32,
    /// Number of rows (Z axis).
    pub height: u32,
    /// Row-major elevation data, each value in `[0, 1]`.
    pub data: Vec<f32>,
    /// World-space distance between adjacent grid cells.
    pub cell_size: f32,
    /// Multiplier applied to stored heights to obtain world-space elevation.
    pub max_height: f32,
}

impl Heightmap {
    /// Construct a heightmap from pre-existing row-major data.
    ///
    /// # Panics
    /// Panics if `data.len() != (width * height) as usize`.
    pub fn from_raw(width: u32, height: u32, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize),
            "Heightmap data length must equal width * height"
        );
        Self {
            width,
            height,
            data,
            cell_size: 1.0,
            max_height: 1.0,
        }
    }

    /// Create a flat (all-zero) heightmap of the given dimensions.
    pub fn flat(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; (width as usize) * (height as usize)],
            cell_size: 1.0,
            max_height: 1.0,
        }
    }

    /// Builder helper: set the world-space cell size.
    pub fn with_cell_size(mut self, cell_size: f32) -> Self {
        self.cell_size = cell_size;
        self
    }

    /// Builder helper: set the maximum height scale factor.
    pub fn with_max_height(mut self, max_height: f32) -> Self {
        self.max_height = max_height;
        self
    }

    /// Total world-space width along the X axis.
    #[inline]
    pub fn world_width(&self) -> f32 {
        (self.width.saturating_sub(1)) as f32 * self.cell_size
    }

    /// Total world-space depth along the Z axis.
    #[inline]
    pub fn world_depth(&self) -> f32 {
        (self.height.saturating_sub(1)) as f32 * self.cell_size
    }

    /// Read the raw stored value at integer grid coordinates.
    /// Returns `0.0` for out-of-bounds indices.
    #[inline]
    pub(crate) fn raw_at(&self, col: u32, row: u32) -> f32 {
        if col >= self.width || row >= self.height {
            return 0.0;
        }
        self.data[(row as usize) * (self.width as usize) + (col as usize)]
    }

    /// Write a raw stored value at integer grid coordinates.
    /// Silently ignores out-of-bounds writes.
    #[inline]
    pub(crate) fn set_raw(&mut self, col: u32, row: u32, value: f32) {
        if col < self.width && row < self.height {
            self.data[(row as usize) * (self.width as usize) + (col as usize)] = value;
        }
    }

    /// Sample the world-space height at an arbitrary `(x, z)` position using
    /// bilinear interpolation.
    ///
    /// Points outside the heightmap footprint are clamped to the nearest edge.
    pub fn sample(&self, x: f32, z: f32) -> f32 {
        if self.width < 2 || self.height < 2 {
            // Degenerate heightmap: return the single value (or 0).
            return self.raw_at(0, 0) * self.max_height;
        }

        // Convert world coords to continuous grid coords.
        let gx = x / self.cell_size;
        let gz = z / self.cell_size;

        // Clamp to valid range.
        let max_col = (self.width - 1) as f32;
        let max_row = (self.height - 1) as f32;
        let gx = gx.clamp(0.0, max_col);
        let gz = gz.clamp(0.0, max_row);

        let col0 = (gx.floor() as u32).min(self.width - 2);
        let row0 = (gz.floor() as u32).min(self.height - 2);
        let col1 = col0 + 1;
        let row1 = row0 + 1;

        let fx = gx - col0 as f32;
        let fz = gz - row0 as f32;

        let h00 = self.raw_at(col0, row0);
        let h10 = self.raw_at(col1, row0);
        let h01 = self.raw_at(col0, row1);
        let h11 = self.raw_at(col1, row1);

        let h = h00 * (1.0 - fx) * (1.0 - fz)
            + h10 * fx * (1.0 - fz)
            + h01 * (1.0 - fx) * fz
            + h11 * fx * fz;

        h * self.max_height
    }

    /// Compute the surface normal at an arbitrary `(x, z)` position using
    /// central differences on the heightmap.
    pub fn normal_at(&self, x: f32, z: f32) -> Vec3 {
        let eps = self.cell_size * 0.5;
        let hx_pos = self.sample(x + eps, z);
        let hx_neg = self.sample(x - eps, z);
        let hz_pos = self.sample(x, z + eps);
        let hz_neg = self.sample(x, z - eps);

        // The gradient in X and Z gives us the tangent vectors:
        //   tangent_x = (2*eps, hx_pos - hx_neg, 0)
        //   tangent_z = (0, hz_pos - hz_neg, 2*eps)
        // Normal = tangent_z x tangent_x  (right-hand rule, Y-up).
        let dx = hx_pos - hx_neg;
        let dz = hz_pos - hz_neg;
        let span = 2.0 * eps;

        let normal = Vec3::new(-dx, span, -dz);
        normal.normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_heightmap_samples_zero() {
        let hm = Heightmap::flat(16, 16).with_max_height(100.0);
        assert_eq!(hm.sample(5.0, 5.0), 0.0);
        assert_eq!(hm.sample(0.0, 0.0), 0.0);
    }

    #[test]
    fn sample_bilinear_interpolation() {
        // 2x2 heightmap:
        //  (0,0)=0.0  (1,0)=1.0
        //  (0,1)=0.0  (1,1)=1.0
        let hm = Heightmap::from_raw(2, 2, vec![0.0, 1.0, 0.0, 1.0])
            .with_cell_size(1.0)
            .with_max_height(10.0);

        // At x=0.5, z=0 -> lerp(0, 1, 0.5) = 0.5 -> * 10 = 5.0
        let h = hm.sample(0.5, 0.0);
        assert!((h - 5.0).abs() < 1e-4, "Expected 5.0, got {h}");

        // At corners
        assert!((hm.sample(0.0, 0.0) - 0.0).abs() < 1e-4);
        assert!((hm.sample(1.0, 0.0) - 10.0).abs() < 1e-4);
    }

    #[test]
    fn sample_clamps_out_of_bounds() {
        let hm = Heightmap::from_raw(2, 2, vec![0.5, 0.5, 0.5, 0.5])
            .with_max_height(2.0);
        // Negative coordinates should clamp
        assert!((hm.sample(-10.0, -10.0) - 1.0).abs() < 1e-4);
        // Far positive should clamp
        assert!((hm.sample(100.0, 100.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn normal_on_flat_surface_points_up() {
        let hm = Heightmap::flat(8, 8).with_cell_size(1.0).with_max_height(10.0);
        let n = hm.normal_at(3.0, 3.0);
        // Should be approximately (0, 1, 0).
        assert!((n.x).abs() < 1e-4, "nx = {}", n.x);
        assert!((n.y - 1.0).abs() < 1e-4, "ny = {}", n.y);
        assert!((n.z).abs() < 1e-4, "nz = {}", n.z);
    }

    #[test]
    fn normal_on_slope_tilts() {
        // A 4x2 heightmap that slopes upward along X (identical rows):
        //   row 0: [0.0, 0.333, 0.666, 1.0]
        //   row 1: [0.0, 0.333, 0.666, 1.0]
        let hm = Heightmap::from_raw(
            4,
            2,
            vec![
                0.0, 0.333, 0.666, 1.0,
                0.0, 0.333, 0.666, 1.0,
            ],
        )
        .with_cell_size(1.0)
        .with_max_height(3.0);
        let n = hm.normal_at(1.5, 0.5);
        // Normal should tilt in -X direction (slope goes up in +X).
        assert!(n.x < -0.1, "Expected negative nx on +X slope, got {}", n.x);
        assert!(n.y > 0.5, "Expected positive ny, got {}", n.y);
    }

    #[test]
    fn from_raw_panics_on_wrong_size() {
        let result = std::panic::catch_unwind(|| {
            Heightmap::from_raw(4, 4, vec![0.0; 10]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn world_extents() {
        let hm = Heightmap::flat(5, 3).with_cell_size(2.0);
        assert!((hm.world_width() - 8.0).abs() < 1e-6); // (5-1)*2
        assert!((hm.world_depth() - 4.0).abs() < 1e-6); // (3-1)*2
    }
}
