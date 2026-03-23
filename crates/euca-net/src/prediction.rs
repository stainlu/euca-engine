//! Client-side prediction and server reconciliation.
//!
//! Enables smooth multiplayer by predicting entity state locally without
//! waiting for server confirmation. When server state arrives, compares
//! against prediction and corrects if diverged.

use std::collections::VecDeque;

/// A snapshot of predicted state at a specific tick.
#[derive(Clone, Debug)]
pub struct PredictedState {
    pub tick: u64,
    /// Predicted position (x, y, z).
    pub position: [f32; 3],
    /// Input that produced this prediction (for replay during reconciliation).
    pub input_snapshot: Vec<u8>,
}

/// Client-side prediction buffer.
///
/// Stores recent predicted states. When server state arrives, compares
/// against the prediction at that tick. If diverged beyond threshold,
/// snaps to server state and replays subsequent inputs.
pub struct ClientPrediction {
    /// Ring buffer of recent predictions, ordered by tick.
    predictions: VecDeque<PredictedState>,
    /// Maximum predictions to keep in buffer.
    max_buffer_size: usize,
    /// Threshold below which prediction error is ignored (avoids jitter).
    pub correction_threshold: f32,
    /// Smoothing factor for corrections (0.0 = snap, 1.0 = never correct).
    pub smoothing: f32,
    /// Accumulated correction to apply smoothly over multiple frames.
    pub pending_correction: [f32; 3],
}

impl ClientPrediction {
    /// Create a new prediction buffer with default settings.
    pub fn new() -> Self {
        Self {
            predictions: VecDeque::new(),
            max_buffer_size: 128,
            correction_threshold: 0.01,
            smoothing: 0.1,
            pending_correction: [0.0; 3],
        }
    }

    /// Record a predicted state for the current tick.
    pub fn record_prediction(&mut self, tick: u64, position: [f32; 3], input: Vec<u8>) {
        self.predictions.push_back(PredictedState {
            tick,
            position,
            input_snapshot: input,
        });

        // Trim old predictions
        while self.predictions.len() > self.max_buffer_size {
            self.predictions.pop_front();
        }
    }

    /// Reconcile with authoritative server state.
    ///
    /// Compares server's position at `server_tick` against our prediction.
    /// Returns `Some((correction_x, correction_y, correction_z))` if correction needed.
    /// Returns `None` if prediction was accurate (within threshold).
    pub fn reconcile(&mut self, server_tick: u64, server_position: [f32; 3]) -> Option<[f32; 3]> {
        // Find our prediction at the server's tick
        let pred_idx = self.predictions.iter().position(|p| p.tick == server_tick);

        let pred_idx = pred_idx?;

        let predicted = &self.predictions[pred_idx];

        // Compute error
        let dx = server_position[0] - predicted.position[0];
        let dy = server_position[1] - predicted.position[1];
        let dz = server_position[2] - predicted.position[2];
        let error_sq = dx * dx + dy * dy + dz * dz;

        // Discard old predictions up to this tick
        while self
            .predictions
            .front()
            .is_some_and(|p| p.tick <= server_tick)
        {
            self.predictions.pop_front();
        }

        if error_sq < self.correction_threshold * self.correction_threshold {
            return None; // Prediction was accurate
        }

        // Apply correction smoothly
        let correction = [dx, dy, dz];
        self.pending_correction[0] += correction[0];
        self.pending_correction[1] += correction[1];
        self.pending_correction[2] += correction[2];

        Some(correction)
    }

    /// Get the smoothed correction to apply this frame, then decay the pending correction.
    /// Call once per frame and add the result to the entity's position.
    pub fn consume_correction(&mut self) -> [f32; 3] {
        let factor = 1.0 - self.smoothing;
        let correction = [
            self.pending_correction[0] * factor,
            self.pending_correction[1] * factor,
            self.pending_correction[2] * factor,
        ];
        self.pending_correction[0] -= correction[0];
        self.pending_correction[1] -= correction[1];
        self.pending_correction[2] -= correction[2];

        // Clear tiny residuals
        if self.pending_correction[0].abs()
            + self.pending_correction[1].abs()
            + self.pending_correction[2].abs()
            < 1e-6
        {
            self.pending_correction = [0.0; 3];
        }

        correction
    }

    /// Number of predictions currently buffered.
    pub fn buffer_len(&self) -> usize {
        self.predictions.len()
    }

    /// Get inputs recorded after `since_tick` for replay during reconciliation.
    pub fn inputs_since(&self, since_tick: u64) -> Vec<&[u8]> {
        self.predictions
            .iter()
            .filter(|p| p.tick > since_tick)
            .map(|p| p.input_snapshot.as_slice())
            .collect()
    }
}

impl Default for ClientPrediction {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply pending prediction corrections to all entities with `ClientPrediction` + `LocalTransform`.
///
/// Call this once per frame in the gameplay loop. Each entity with a `ClientPrediction`
/// component has its correction consumed and applied to its position.
pub fn apply_prediction_system(world: &mut euca_ecs::World) {
    use euca_ecs::{Entity, Query};

    let entities_with_prediction: Vec<Entity> = {
        let query = Query::<Entity>::new(world);
        query
            .iter()
            .filter(|e| world.get::<ClientPrediction>(*e).is_some())
            .collect()
    };

    for entity in entities_with_prediction {
        let correction = {
            let pred = match world.get_mut::<ClientPrediction>(entity) {
                Some(p) => p,
                None => continue,
            };
            pred.consume_correction()
        };

        // Skip zero corrections
        if correction[0].abs() + correction[1].abs() + correction[2].abs() < 1e-7 {
            continue;
        }

        if let Some(lt) = world.get_mut::<euca_scene::LocalTransform>(entity) {
            lt.0.translation.x += correction[0];
            lt.0.translation.y += correction[1];
            lt.0.translation.z += correction[2];
        }
    }
}

/// Record a prediction for the given entity at the current tick.
///
/// Captures the entity's current position and the provided input snapshot.
pub fn record_prediction_for_entity(
    world: &mut euca_ecs::World,
    entity: euca_ecs::Entity,
    tick: u64,
    input_snapshot: Vec<u8>,
) {
    let position = world
        .get::<euca_scene::LocalTransform>(entity)
        .map(|lt| [lt.0.translation.x, lt.0.translation.y, lt.0.translation.z])
        .unwrap_or([0.0; 3]);

    if let Some(pred) = world.get_mut::<ClientPrediction>(entity) {
        pred.record_prediction(tick, position, input_snapshot);
    }
}

/// Reconcile a server state update for the given entity.
///
/// If the server position diverges from our prediction at `server_tick`,
/// the correction is accumulated for smooth application via `apply_prediction_system`.
pub fn reconcile_entity(
    world: &mut euca_ecs::World,
    entity: euca_ecs::Entity,
    server_tick: u64,
    server_position: [f32; 3],
) -> Option<[f32; 3]> {
    let pred = world.get_mut::<ClientPrediction>(entity)?;
    pred.reconcile(server_tick, server_position)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accurate_prediction_no_correction() {
        let mut pred = ClientPrediction::new();
        pred.record_prediction(1, [10.0, 0.0, 0.0], vec![]);
        pred.record_prediction(2, [11.0, 0.0, 0.0], vec![]);

        // Server confirms tick 1 at exactly our predicted position
        let correction = pred.reconcile(1, [10.0, 0.0, 0.0]);
        assert!(correction.is_none());
    }

    #[test]
    fn diverged_prediction_returns_correction() {
        let mut pred = ClientPrediction::new();
        pred.record_prediction(1, [10.0, 0.0, 0.0], vec![]);

        // Server says we're at 12.0, we predicted 10.0 — error of 2.0
        let correction = pred.reconcile(1, [12.0, 0.0, 0.0]);
        assert!(correction.is_some());
        let c = correction.unwrap();
        assert!((c[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn smooth_correction_decays() {
        let mut pred = ClientPrediction::new();
        pred.pending_correction = [1.0, 0.0, 0.0];
        pred.smoothing = 0.5;

        let c1 = pred.consume_correction();
        assert!((c1[0] - 0.5).abs() < 1e-5); // 50% of 1.0

        let c2 = pred.consume_correction();
        assert!((c2[0] - 0.25).abs() < 1e-5); // 50% of 0.5
    }

    #[test]
    fn old_predictions_trimmed() {
        let mut pred = ClientPrediction::new();
        for tick in 0..10 {
            pred.record_prediction(tick, [tick as f32, 0.0, 0.0], vec![]);
        }

        // Reconcile at tick 5 — predictions 0-5 should be discarded
        let _ = pred.reconcile(5, [5.0, 0.0, 0.0]);
        assert_eq!(pred.buffer_len(), 4); // ticks 6,7,8,9 remain
    }

    #[test]
    fn apply_prediction_system_corrects_position() {
        let mut world = euca_ecs::World::new();
        let e = world.spawn(euca_scene::LocalTransform(
            euca_math::Transform::from_translation(euca_math::Vec3::new(0.0, 0.0, 0.0)),
        ));
        let mut pred = ClientPrediction::new();
        pred.pending_correction = [2.0, 0.0, 0.0];
        pred.smoothing = 0.0; // Snap correction (factor = 1.0)
        world.insert(e, pred);

        apply_prediction_system(&mut world);

        let lt = world.get::<euca_scene::LocalTransform>(e).unwrap();
        assert!((lt.0.translation.x - 2.0).abs() < 1e-5);
    }

    #[test]
    fn record_and_reconcile_entity() {
        let mut world = euca_ecs::World::new();
        let e = world.spawn(euca_scene::LocalTransform(
            euca_math::Transform::from_translation(euca_math::Vec3::new(5.0, 0.0, 0.0)),
        ));
        world.insert(e, ClientPrediction::new());

        record_prediction_for_entity(&mut world, e, 1, vec![]);

        // Server says we're at 7.0, we predicted 5.0
        let correction = reconcile_entity(&mut world, e, 1, [7.0, 0.0, 0.0]);
        assert!(correction.is_some());
        assert!((correction.unwrap()[0] - 2.0).abs() < 1e-5);
    }
}
