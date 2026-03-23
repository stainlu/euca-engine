//! Reverb zones and weighted reverb parameter blending.

use crate::source::AudioListener;
use euca_ecs::World;
use euca_math::Vec3;
use euca_scene::GlobalTransform;

/// Spherical reverb region. Attach to an entity with [`GlobalTransform`] to
/// define a zone where audio receives reverb processing.
///
/// When an [`AudioListener`] overlaps this sphere, all active sounds have
/// weighted reverb applied (based on distance to the zone center).
pub struct ReverbZone {
    /// Radius of the reverb sphere.
    pub radius: f32,
    /// Wet/dry mix of the reverb effect (0.0 = fully dry, 1.0 = fully wet).
    pub mix: f32,
    /// Reverb decay / feedback (0.0 - 1.0).  Maps to kira's `feedback` param.
    pub decay: f32,
    /// High-frequency damping (0.0 - 1.0). Maps to kira's `damping` param.
    pub damping: f32,
}

impl ReverbZone {
    /// Create a new reverb zone with the given radius.
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            mix: 0.5,
            decay: 0.8,
            damping: 0.3,
        }
    }

    pub fn with_mix(mut self, mix: f32) -> Self {
        self.mix = mix.clamp(0.0, 1.0);
        self
    }

    pub fn with_decay(mut self, decay: f32) -> Self {
        self.decay = decay.clamp(0.0, 1.0);
        self
    }

    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }
}

/// Collects reverb zone data from the world.
pub(crate) fn collect_reverb_zones(world: &World) -> Vec<(Vec3, f32, f32, f32, f32)> {
    let query = euca_ecs::Query::<(&ReverbZone, &GlobalTransform)>::new(world);
    query
        .iter()
        .map(|(rz, gt)| (gt.0.translation, rz.radius, rz.mix, rz.decay, rz.damping))
        .collect()
}

/// Compute distance-weighted reverb parameters from all overlapping reverb zones.
///
/// For each zone that contains `listener_pos`, compute a weight based on how
/// close the listener is to the zone center (linear falloff). Then blend the
/// mix, feedback, and damping values proportionally.
pub(crate) fn compute_reverb_params(
    zones: &[(Vec3, f32, f32, f32, f32)],
    listener_pos: Vec3,
) -> (f32, f32, f32) {
    let mut total_weight = 0.0_f32;
    let mut weighted_mix = 0.0_f32;
    let mut weighted_feedback = 0.0_f32;
    let mut weighted_damping = 0.0_f32;

    for &(center, radius, mix, decay, damping) in zones {
        let dist = (listener_pos - center).length();
        if dist < radius {
            let weight = 1.0 - (dist / radius);
            total_weight += weight;
            weighted_mix += mix * weight;
            weighted_feedback += decay * weight;
            weighted_damping += damping * weight;
        }
    }

    if total_weight > 0.0 {
        (
            weighted_mix / total_weight,
            weighted_feedback / total_weight,
            weighted_damping / total_weight,
        )
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Returns the position of the first [`AudioListener`] entity, or `Vec3::ZERO` if none.
pub(crate) fn listener_position(world: &World) -> Vec3 {
    let query = euca_ecs::Query::<(&AudioListener, &GlobalTransform)>::new(world);
    query
        .iter()
        .next()
        .map(|(_, gt)| gt.0.translation)
        .unwrap_or(Vec3::ZERO)
}

/// Standalone reverb zone query: returns the blended reverb parameters
/// (mix, feedback, damping) for the current listener position.
///
/// Useful if you want to apply reverb through a kira send track yourself.
pub fn query_reverb_for_listener(world: &World) -> (f32, f32, f32) {
    compute_reverb_params(&collect_reverb_zones(world), listener_position(world))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverb_params_no_zones() {
        let (mix, feedback, damping) = compute_reverb_params(&[], Vec3::ZERO);
        assert!((mix).abs() < f32::EPSILON);
        assert!((feedback).abs() < f32::EPSILON);
        assert!((damping).abs() < f32::EPSILON);
    }

    #[test]
    fn reverb_params_listener_at_center() {
        let zones = vec![(Vec3::ZERO, 10.0, 0.6, 0.9, 0.2)];
        let (mix, feedback, damping) = compute_reverb_params(&zones, Vec3::ZERO);
        assert!((mix - 0.6).abs() < f32::EPSILON);
        assert!((feedback - 0.9).abs() < f32::EPSILON);
        assert!((damping - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn reverb_params_listener_outside_zone() {
        let zones = vec![(Vec3::ZERO, 5.0, 0.6, 0.9, 0.2)];
        let far_away = Vec3::new(100.0, 0.0, 0.0);
        let (mix, _fb, _damp) = compute_reverb_params(&zones, far_away);
        assert!((mix).abs() < f32::EPSILON);
    }

    #[test]
    fn reverb_params_two_zones_blended() {
        // Two zones centered at different positions, listener at origin.
        let zones = vec![
            (Vec3::ZERO, 10.0, 0.4, 0.8, 0.1), // weight = 1.0 (dist=0)
            (Vec3::new(5.0, 0.0, 0.0), 10.0, 0.8, 0.6, 0.5), // weight = 0.5 (dist=5)
        ];
        let (mix, feedback, damping) = compute_reverb_params(&zones, Vec3::ZERO);
        // Weighted blend: (0.4*1 + 0.8*0.5) / 1.5 = 0.8/1.5 = 0.5333...
        let expected_mix = (0.4 + 0.8 * 0.5) / 1.5;
        assert!((mix - expected_mix).abs() < 0.001);
        let expected_feedback = (0.8 + 0.6 * 0.5) / 1.5;
        assert!((feedback - expected_feedback).abs() < 0.001);
        let expected_damping = (0.1 + 0.5 * 0.5) / 1.5;
        assert!((damping - expected_damping).abs() < 0.001);
    }

    #[test]
    fn reverb_zone_builder() {
        let rz = ReverbZone::new(15.0)
            .with_mix(0.7)
            .with_decay(0.85)
            .with_damping(0.4);
        assert!((rz.radius - 15.0).abs() < f32::EPSILON);
        assert!((rz.mix - 0.7).abs() < f32::EPSILON);
        assert!((rz.decay - 0.85).abs() < f32::EPSILON);
        assert!((rz.damping - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn reverb_zone_clamps() {
        let rz = ReverbZone::new(10.0)
            .with_mix(1.5)
            .with_decay(-0.1)
            .with_damping(2.0);
        assert!((rz.mix - 1.0).abs() < f32::EPSILON);
        assert!((rz.decay).abs() < f32::EPSILON);
        assert!((rz.damping - 1.0).abs() < f32::EPSILON);
    }
}
