//! Terrain editing API for runtime sculpting and painting.
//!
//! All operations work on the underlying `Heightmap` or `SplatMap` data and
//! use a brush defined by a centre point and radius.

use euca_math::Vec4;

use crate::heightmap::Heightmap;
use crate::splat::SplatMap;

/// Raise terrain within a circular brush.
///
/// `center_x`, `center_z` are world-space coordinates of the brush centre.
/// `radius` is the brush radius in world units.
/// `strength` is the amount to raise (in normalised heightmap units).
/// The effect falls off linearly from the centre to the edge of the brush.
pub fn raise_terrain(
    heightmap: &mut Heightmap,
    center_x: f32,
    center_z: f32,
    radius: f32,
    strength: f32,
) {
    apply_brush(heightmap, center_x, center_z, radius, |current, falloff| {
        current + strength * falloff
    });
}

/// Lower terrain within a circular brush (convenience wrapper for negative raise).
pub fn lower_terrain(
    heightmap: &mut Heightmap,
    center_x: f32,
    center_z: f32,
    radius: f32,
    strength: f32,
) {
    raise_terrain(heightmap, center_x, center_z, radius, -strength);
}

/// Flatten terrain within a circular brush to a target height.
///
/// `target_height` is in normalised heightmap units (before `max_height` scaling).
/// The flattening blends toward the target by `strength * falloff`.
pub fn flatten_terrain(
    heightmap: &mut Heightmap,
    center_x: f32,
    center_z: f32,
    radius: f32,
    target_height: f32,
    strength: f32,
) {
    apply_brush(heightmap, center_x, center_z, radius, |current, falloff| {
        let t = (strength * falloff).clamp(0.0, 1.0);
        current * (1.0 - t) + target_height * t
    });
}

/// Smooth terrain within a circular brush by averaging neighbours.
///
/// Each affected cell is blended toward the average of its 4 cardinal
/// neighbours, weighted by `strength * falloff`.
pub fn smooth_terrain(
    heightmap: &mut Heightmap,
    center_x: f32,
    center_z: f32,
    radius: f32,
    strength: f32,
) {
    let (col_min, col_max, row_min, row_max) = brush_bounds(
        center_x,
        center_z,
        radius,
        heightmap.cell_size,
        heightmap.width,
        heightmap.height,
    );

    // Collect edits so we read from the original data (two-pass to avoid
    // read-after-write within the same brush stroke).
    let mut edits: Vec<(u32, u32, f32)> = Vec::new();

    for row in row_min..=row_max {
        for col in col_min..=col_max {
            let Some(falloff) =
                brush_falloff(col, row, center_x, center_z, radius, heightmap.cell_size)
            else {
                continue;
            };

            let current = heightmap.raw_at(col, row);

            // Average of cardinal neighbours.
            let mut sum = 0.0;
            let mut count = 0u32;
            for (dc, dr) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                let nc = col as i32 + dc;
                let nr = row as i32 + dr;
                if nc >= 0
                    && (nc as u32) < heightmap.width
                    && nr >= 0
                    && (nr as u32) < heightmap.height
                {
                    sum += heightmap.raw_at(nc as u32, nr as u32);
                    count += 1;
                }
            }

            if count > 0 {
                let avg = sum / count as f32;
                let t = (strength * falloff).clamp(0.0, 1.0);
                let new_val = current * (1.0 - t) + avg * t;
                edits.push((col, row, new_val));
            }
        }
    }

    for (col, row, val) in edits {
        heightmap.set_raw(col, row, val);
    }
}

/// Paint a splat layer within a circular brush.
///
/// `layer` is 0..3. The specified layer's weight is increased by
/// `strength * falloff`, and weights are re-normalised.
pub fn paint_splat(
    splat_map: &mut SplatMap,
    center_x: f32,
    center_z: f32,
    radius: f32,
    cell_size: f32,
    layer: usize,
    strength: f32,
) {
    if layer >= 4 {
        return;
    }

    let (col_min, col_max, row_min, row_max) = brush_bounds(
        center_x,
        center_z,
        radius,
        cell_size,
        splat_map.width,
        splat_map.height,
    );

    for row in row_min..=row_max {
        for col in col_min..=col_max {
            let Some(falloff) = brush_falloff(col, row, center_x, center_z, radius, cell_size)
            else {
                continue;
            };

            let mut w = splat_map.get(col, row);
            let amount = strength * falloff;

            // Add weight to the target layer.
            match layer {
                0 => w.x += amount,
                1 => w.y += amount,
                2 => w.z += amount,
                3 => w.w += amount,
                _ => {}
            }

            // Clamp and renormalise.
            w = Vec4::new(w.x.max(0.0), w.y.max(0.0), w.z.max(0.0), w.w.max(0.0));
            w = SplatMap::normalize_weights(w);
            splat_map.set(col, row, w);
        }
    }
}

/// Compute the grid-cell bounds for a circular brush.
///
/// Returns `(col_min, col_max, row_min, row_max)` inclusive.
fn brush_bounds(
    center_x: f32,
    center_z: f32,
    radius: f32,
    cell_size: f32,
    grid_width: u32,
    grid_height: u32,
) -> (u32, u32, u32, u32) {
    let col_min = ((center_x - radius) / cell_size).floor().max(0.0) as u32;
    let col_max =
        (((center_x + radius) / cell_size).ceil() as u32).min(grid_width.saturating_sub(1));
    let row_min = ((center_z - radius) / cell_size).floor().max(0.0) as u32;
    let row_max =
        (((center_z + radius) / cell_size).ceil() as u32).min(grid_height.saturating_sub(1));
    (col_min, col_max, row_min, row_max)
}

/// Compute the linear falloff for a cell at `(col, row)` within a brush.
///
/// Returns `None` if the cell is outside the brush radius.
#[inline]
fn brush_falloff(
    col: u32,
    row: u32,
    center_x: f32,
    center_z: f32,
    radius: f32,
    cell_size: f32,
) -> Option<f32> {
    let dx = col as f32 * cell_size - center_x;
    let dz = row as f32 * cell_size - center_z;
    let dist = (dx * dx + dz * dz).sqrt();
    if dist > radius {
        return None;
    }
    Some(1.0 - (dist / radius))
}

/// Internal helper: iterate over heightmap cells within a brush and apply a
/// function `f(current_height, falloff) -> new_height`.
fn apply_brush(
    heightmap: &mut Heightmap,
    center_x: f32,
    center_z: f32,
    radius: f32,
    f: impl Fn(f32, f32) -> f32,
) {
    let (col_min, col_max, row_min, row_max) = brush_bounds(
        center_x,
        center_z,
        radius,
        heightmap.cell_size,
        heightmap.width,
        heightmap.height,
    );

    for row in row_min..=row_max {
        for col in col_min..=col_max {
            let Some(falloff) =
                brush_falloff(col, row, center_x, center_z, radius, heightmap.cell_size)
            else {
                continue;
            };
            let current = heightmap.raw_at(col, row);
            let new_val = f(current, falloff);
            heightmap.set_raw(col, row, new_val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raise_terrain_increases_center() {
        let mut hm = Heightmap::flat(8, 8)
            .with_cell_size(1.0)
            .with_max_height(10.0);
        let before = hm.sample(3.0, 3.0);
        raise_terrain(&mut hm, 3.0, 3.0, 2.0, 0.5);
        let after = hm.sample(3.0, 3.0);
        assert!(
            after > before,
            "Center should be raised: before={before}, after={after}"
        );
    }

    #[test]
    fn raise_terrain_falls_off_with_distance() {
        let mut hm = Heightmap::flat(16, 16)
            .with_cell_size(1.0)
            .with_max_height(1.0);
        raise_terrain(&mut hm, 8.0, 8.0, 4.0, 1.0);

        let h_center = hm.sample(8.0, 8.0);
        let h_edge = hm.sample(11.0, 8.0); // ~3 units away, radius=4
        let h_outside = hm.sample(13.0, 8.0); // 5 units away, outside

        assert!(h_center > h_edge, "Center should be higher than edge");
        assert!(
            (h_outside).abs() < 1e-4,
            "Outside brush should be unaffected"
        );
    }

    #[test]
    fn flatten_terrain_converges() {
        let mut hm = Heightmap::from_raw(5, 5, vec![1.0; 25])
            .with_cell_size(1.0)
            .with_max_height(10.0);

        flatten_terrain(&mut hm, 2.0, 2.0, 3.0, 0.5, 1.0);

        let h = hm.raw_at(2, 2);
        assert!(
            (h - 0.5).abs() < 0.3,
            "Center should move toward target 0.5, got {h}"
        );
    }

    #[test]
    fn smooth_terrain_reduces_spikes() {
        let mut data = vec![0.0; 25]; // 5x5
        data[12] = 1.0; // Spike at (2,2).
        let mut hm = Heightmap::from_raw(5, 5, data).with_cell_size(1.0);

        let before = hm.raw_at(2, 2);
        smooth_terrain(&mut hm, 2.0, 2.0, 3.0, 1.0);
        let after = hm.raw_at(2, 2);

        assert!(
            after < before,
            "Spike should be reduced: before={before}, after={after}"
        );
    }

    #[test]
    fn paint_splat_changes_layer() {
        let mut sm = SplatMap::uniform(8, 8);
        paint_splat(&mut sm, 4.0, 4.0, 3.0, 1.0, 2, 1.0);

        let w = sm.get(4, 4);
        assert!(w.z > 0.1, "Layer 2 should have weight, got {}", w.z);
        let sum = w.x + w.y + w.z + w.w;
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Weights should sum to 1, got {sum}"
        );
    }
}
