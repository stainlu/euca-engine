//! Blend spaces: parametric animation blending along 1D or 2D axes.
//!
//! A blend space maps a parameter value (e.g. movement speed) to a blend
//! of nearby animation clips. For example, `speed = 0.7` might blend
//! between "walk" (at 0.5) and "run" (at 1.0).

use crate::clip::AnimPose;

/// A sample point in a 1D blend space.
#[derive(Clone, Debug)]
pub struct BlendSample1D {
    /// Parameter value at which this clip is at full weight.
    pub position: f32,
    /// Index into the clip library.
    pub clip_index: usize,
}

/// A 1D blend space: blends between clips along a single axis.
///
/// Clips are positioned along a number line. Given a parameter value,
/// the two nearest clips are blended proportionally.
#[derive(Clone, Debug)]
pub struct BlendSpace1D {
    samples: Vec<BlendSample1D>,
}

impl BlendSpace1D {
    /// Create a new 1D blend space. Samples will be sorted by position.
    pub fn new(mut samples: Vec<BlendSample1D>) -> Self {
        samples.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
        Self { samples }
    }

    /// Evaluate the blend space at the given parameter value.
    ///
    /// Returns a list of `(clip_index, weight)` pairs. At most two clips
    /// will have nonzero weight.
    pub fn evaluate(&self, param: f32) -> Vec<(usize, f32)> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        if self.samples.len() == 1 {
            return vec![(self.samples[0].clip_index, 1.0)];
        }

        // Clamp to range
        let first = &self.samples[0];
        let last = &self.samples[self.samples.len() - 1];

        if param <= first.position {
            return vec![(first.clip_index, 1.0)];
        }
        if param >= last.position {
            return vec![(last.clip_index, 1.0)];
        }

        // Find the two bracketing samples
        for i in 0..self.samples.len() - 1 {
            let lo = &self.samples[i];
            let hi = &self.samples[i + 1];
            if param >= lo.position && param <= hi.position {
                let range = hi.position - lo.position;
                if range < f32::EPSILON {
                    return vec![(lo.clip_index, 1.0)];
                }
                let t = (param - lo.position) / range;
                return vec![(lo.clip_index, 1.0 - t), (hi.clip_index, t)];
            }
        }

        vec![(last.clip_index, 1.0)]
    }
}

/// A sample point in a 2D blend space.
#[derive(Clone, Debug)]
pub struct BlendSample2D {
    /// Position on the X axis.
    pub x: f32,
    /// Position on the Y axis.
    pub y: f32,
    /// Index into the clip library.
    pub clip_index: usize,
}

/// A 2D blend space: blends between clips on a 2D plane.
///
/// Uses inverse-distance weighting to compute blend weights from
/// the parameter point to each sample point.
#[derive(Clone, Debug)]
pub struct BlendSpace2D {
    samples: Vec<BlendSample2D>,
}

impl BlendSpace2D {
    /// Create a new 2D blend space.
    pub fn new(samples: Vec<BlendSample2D>) -> Self {
        Self { samples }
    }

    /// Evaluate the blend space at the given 2D parameter.
    ///
    /// Returns `(clip_index, weight)` pairs using inverse-distance weighting.
    /// If the parameter exactly matches a sample, that sample gets weight 1.0.
    pub fn evaluate(&self, px: f32, py: f32) -> Vec<(usize, f32)> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        if self.samples.len() == 1 {
            return vec![(self.samples[0].clip_index, 1.0)];
        }

        // Compute distances and check for exact match
        let snap_threshold = 1e-6;
        let mut distances: Vec<f32> = Vec::with_capacity(self.samples.len());

        for sample in &self.samples {
            let dx = px - sample.x;
            let dy = py - sample.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < snap_threshold {
                return vec![(sample.clip_index, 1.0)];
            }
            distances.push(dist);
        }

        // Inverse-distance weighting (power = 2)
        let inv_weights: Vec<f32> = distances.iter().map(|d| 1.0 / (d * d)).collect();
        let total: f32 = inv_weights.iter().sum();

        self.samples
            .iter()
            .zip(inv_weights.iter())
            .map(|(sample, &w)| (sample.clip_index, w / total))
            .collect()
    }
}

/// Blend multiple clip poses given weights from a blend space evaluation.
///
/// `evaluate_fn` takes a clip_index and returns the sampled pose at the current time.
pub fn blend_space_poses<F>(weights: &[(usize, f32)], mut evaluate_fn: F) -> Option<AnimPose>
where
    F: FnMut(usize) -> Option<AnimPose>,
{
    if weights.is_empty() {
        return None;
    }

    // Collect poses with their weights
    let mut pose_weights: Vec<(AnimPose, f32)> = Vec::new();
    for &(clip_idx, weight) in weights {
        if weight > 0.0
            && let Some(pose) = evaluate_fn(clip_idx)
        {
            pose_weights.push((pose, weight));
        }
    }

    if pose_weights.is_empty() {
        return None;
    }

    if pose_weights.len() == 1 {
        return Some(pose_weights.into_iter().next().unwrap().0);
    }

    // Blend all poses together with normalized weights
    let total: f32 = pose_weights.iter().map(|(_, w)| w).sum();
    let mut result = pose_weights[0].0.clone();
    let mut accumulated = pose_weights[0].1 / total;

    for (pose, weight) in &pose_weights[1..] {
        let normalized = weight / total;
        let blend_factor = normalized / (accumulated + normalized);
        result = result.blend(pose, blend_factor);
        accumulated += normalized;
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_space_1d_single_sample() {
        let bs = BlendSpace1D::new(vec![BlendSample1D {
            position: 0.5,
            clip_index: 0,
        }]);
        let weights = bs.evaluate(0.5);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], (0, 1.0));
    }

    #[test]
    fn blend_space_1d_between_two() {
        let bs = BlendSpace1D::new(vec![
            BlendSample1D {
                position: 0.0,
                clip_index: 0,
            },
            BlendSample1D {
                position: 1.0,
                clip_index: 1,
            },
        ]);

        let weights = bs.evaluate(0.5);
        assert_eq!(weights.len(), 2);
        assert!((weights[0].1 - 0.5).abs() < 1e-5);
        assert!((weights[1].1 - 0.5).abs() < 1e-5);
    }

    #[test]
    fn blend_space_1d_clamp_low() {
        let bs = BlendSpace1D::new(vec![
            BlendSample1D {
                position: 0.0,
                clip_index: 0,
            },
            BlendSample1D {
                position: 1.0,
                clip_index: 1,
            },
        ]);

        let weights = bs.evaluate(-5.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], (0, 1.0));
    }

    #[test]
    fn blend_space_1d_clamp_high() {
        let bs = BlendSpace1D::new(vec![
            BlendSample1D {
                position: 0.0,
                clip_index: 0,
            },
            BlendSample1D {
                position: 1.0,
                clip_index: 1,
            },
        ]);

        let weights = bs.evaluate(5.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], (1, 1.0));
    }

    #[test]
    fn blend_space_1d_three_samples() {
        let bs = BlendSpace1D::new(vec![
            BlendSample1D {
                position: 0.0,
                clip_index: 0,
            },
            BlendSample1D {
                position: 0.5,
                clip_index: 1,
            },
            BlendSample1D {
                position: 1.0,
                clip_index: 2,
            },
        ]);

        // At 0.25: between idle (0.0) and walk (0.5)
        let w = bs.evaluate(0.25);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].0, 0);
        assert_eq!(w[1].0, 1);
        assert!((w[0].1 - 0.5).abs() < 1e-5);

        // At 0.5: exactly walk
        let w = bs.evaluate(0.5);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].0, 0);
        assert!((w[0].1).abs() < 1e-5);
        assert!((w[1].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn blend_space_2d_exact_match() {
        let bs = BlendSpace2D::new(vec![
            BlendSample2D {
                x: 0.0,
                y: 0.0,
                clip_index: 0,
            },
            BlendSample2D {
                x: 1.0,
                y: 0.0,
                clip_index: 1,
            },
            BlendSample2D {
                x: 0.0,
                y: 1.0,
                clip_index: 2,
            },
        ]);

        let weights = bs.evaluate(0.0, 0.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], (0, 1.0));
    }

    #[test]
    fn blend_space_2d_inverse_distance() {
        let bs = BlendSpace2D::new(vec![
            BlendSample2D {
                x: 0.0,
                y: 0.0,
                clip_index: 0,
            },
            BlendSample2D {
                x: 2.0,
                y: 0.0,
                clip_index: 1,
            },
        ]);

        let weights = bs.evaluate(1.0, 0.0);
        assert_eq!(weights.len(), 2);
        assert!((weights[0].1 - 0.5).abs() < 1e-5);
        assert!((weights[1].1 - 0.5).abs() < 1e-5);
    }
}
