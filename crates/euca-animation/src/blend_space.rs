//! 1D blend space — parametric blending between animation clips along a single axis.
//!
//! Example: a "speed" parameter blends between walk (0.0) and run (1.0) clips.

use crate::blend::BlendLayer;

/// A sample point in a 1D blend space: maps a parameter value to a clip.
#[derive(Clone, Debug)]
pub struct BlendSpaceSample {
    /// The parameter value at which this clip has full weight.
    pub position: f32,
    /// Index into the animation library's clip list.
    pub clip_index: usize,
}

/// A 1D blend space that interpolates between clips based on a single parameter.
///
/// Samples must be sorted by position (ascending). The blend space linearly
/// interpolates between the two nearest samples.
#[derive(Clone, Debug)]
pub struct BlendSpace1D {
    /// Sorted list of sample points.
    samples: Vec<BlendSpaceSample>,
}

impl BlendSpace1D {
    /// Create a new blend space from samples. Samples are sorted by position automatically.
    ///
    /// # Panics
    /// Panics if `samples` is empty.
    pub fn new(mut samples: Vec<BlendSpaceSample>) -> Self {
        assert!(!samples.is_empty(), "BlendSpace1D requires at least one sample");
        samples.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
        Self { samples }
    }

    /// Evaluate the blend space at a given parameter value, returning blend layers
    /// with appropriate weights.
    ///
    /// The `time` parameter is the current playback time applied to both clips
    /// (assumes synchronized playback). For more control, use `evaluate_indices`
    /// and build layers manually.
    pub fn evaluate(&self, parameter: f32, time: f32) -> Vec<BlendLayer> {
        let (indices, weights) = self.evaluate_weights(parameter);
        indices
            .into_iter()
            .zip(weights)
            .filter(|(_, w)| *w > 0.0)
            .map(|(idx, w)| BlendLayer {
                clip_index: self.samples[idx].clip_index,
                time,
                weight: w,
            })
            .collect()
    }

    /// Compute which sample indices and weights are active for a given parameter value.
    ///
    /// Returns `(indices, weights)` where indices are into `self.samples`.
    /// At most two indices are returned (the two samples that bracket the parameter).
    fn evaluate_weights(&self, parameter: f32) -> (Vec<usize>, Vec<f32>) {
        // Clamp to range
        if parameter <= self.samples[0].position {
            return (vec![0], vec![1.0]);
        }
        let last = self.samples.len() - 1;
        if parameter >= self.samples[last].position {
            return (vec![last], vec![1.0]);
        }

        // Find bracketing samples.
        for i in 0..self.samples.len() - 1 {
            let lo = &self.samples[i];
            let hi = &self.samples[i + 1];
            if parameter >= lo.position && parameter <= hi.position {
                let range = hi.position - lo.position;
                if range <= 0.0 {
                    return (vec![i], vec![1.0]);
                }
                let t = (parameter - lo.position) / range;
                return (vec![i, i + 1], vec![1.0 - t, t]);
            }
        }

        // Should not reach here given the clamping above.
        (vec![0], vec![1.0])
    }

    /// Returns a reference to the samples.
    pub fn samples(&self) -> &[BlendSpaceSample] {
        &self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk_run_space() -> BlendSpace1D {
        BlendSpace1D::new(vec![
            BlendSpaceSample { position: 0.0, clip_index: 0 }, // walk
            BlendSpaceSample { position: 1.0, clip_index: 1 }, // run
        ])
    }

    #[test]
    fn at_first_sample_full_weight() {
        let space = walk_run_space();
        let layers = space.evaluate(0.0, 0.5);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].clip_index, 0);
        assert!((layers[0].weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn at_last_sample_full_weight() {
        let space = walk_run_space();
        let layers = space.evaluate(1.0, 0.5);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].clip_index, 1);
        assert!((layers[0].weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn midpoint_blends_equally() {
        let space = walk_run_space();
        let layers = space.evaluate(0.5, 0.0);
        assert_eq!(layers.len(), 2);
        assert!((layers[0].weight - 0.5).abs() < 0.001);
        assert!((layers[1].weight - 0.5).abs() < 0.001);
    }

    #[test]
    fn clamps_below_range() {
        let space = walk_run_space();
        let layers = space.evaluate(-1.0, 0.0);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].clip_index, 0);
    }

    #[test]
    fn clamps_above_range() {
        let space = walk_run_space();
        let layers = space.evaluate(5.0, 0.0);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].clip_index, 1);
    }

    #[test]
    fn three_sample_blend_space() {
        let space = BlendSpace1D::new(vec![
            BlendSpaceSample { position: 0.0, clip_index: 0 }, // idle
            BlendSpaceSample { position: 0.5, clip_index: 1 }, // walk
            BlendSpaceSample { position: 1.0, clip_index: 2 }, // run
        ]);

        // At 0.25: between idle (0.0) and walk (0.5)
        let layers = space.evaluate(0.25, 0.0);
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].clip_index, 0);
        assert_eq!(layers[1].clip_index, 1);
        assert!((layers[0].weight - 0.5).abs() < 0.001);
        assert!((layers[1].weight - 0.5).abs() < 0.001);

        // At 0.75: between walk (0.5) and run (1.0)
        let layers = space.evaluate(0.75, 0.0);
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].clip_index, 1);
        assert_eq!(layers[1].clip_index, 2);
    }
}
