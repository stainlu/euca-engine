//! Animation montages — interruptable one-shot animations that play over the state machine.
//!
//! Montages are used for actions like attacks, reloads, and emotes that temporarily
//! override the state machine's output.

/// Definition of a montage (reusable template).
#[derive(Clone, Debug)]
pub struct MontageDefinition {
    /// Human-readable name.
    pub name: String,
    /// Index into the animation library's clip list.
    pub clip_index: usize,
    /// Playback speed multiplier.
    pub speed: f32,
    /// Duration of the blend-in from the current pose (seconds).
    pub blend_in: f32,
    /// Duration of the blend-out back to the state machine (seconds).
    pub blend_out: f32,
    /// Whether this montage can be interrupted by another montage.
    pub interruptible: bool,
}

/// Runtime state of an active montage on an entity.
#[derive(Clone, Debug)]
pub struct ActiveMontage {
    /// The montage definition being played.
    pub definition: MontageDefinition,
    /// Current playback time within the montage clip (seconds).
    pub time: f32,
    /// The duration of the clip (cached from the animation library).
    pub clip_duration: f32,
    /// Current phase of the montage lifecycle.
    pub phase: MontagePhase,
}

/// Lifecycle phases of a montage.
#[derive(Clone, Debug, PartialEq)]
pub enum MontagePhase {
    /// Blending in from the underlying pose.
    BlendingIn,
    /// Fully active — montage has full control.
    Playing,
    /// Blending out back to the underlying pose.
    BlendingOut,
    /// Montage is finished and should be removed.
    Finished,
}

impl ActiveMontage {
    /// Create a new active montage from a definition.
    pub fn new(definition: MontageDefinition, clip_duration: f32) -> Self {
        let phase = if definition.blend_in > 0.0 {
            MontagePhase::BlendingIn
        } else {
            MontagePhase::Playing
        };
        Self {
            definition,
            time: 0.0,
            clip_duration,
            phase,
        }
    }

    /// Advance the montage by `dt` seconds and update the phase.
    pub fn advance(&mut self, dt: f32) {
        self.time += dt * self.definition.speed;
        self.update_phase();
    }

    /// Update the phase based on current time.
    fn update_phase(&mut self) {
        if self.phase == MontagePhase::Finished {
            return;
        }

        let blend_out_start = self.clip_duration - self.definition.blend_out;

        if self.time >= self.clip_duration {
            self.phase = MontagePhase::Finished;
        } else if self.time >= blend_out_start && self.definition.blend_out > 0.0 {
            self.phase = MontagePhase::BlendingOut;
        } else if self.time >= self.definition.blend_in {
            self.phase = MontagePhase::Playing;
        }
        // Otherwise stays in BlendingIn.
    }

    /// Returns the blend weight of the montage over the state machine.
    ///
    /// - During blend-in: ramps from 0.0 to 1.0.
    /// - During playing: 1.0 (full override).
    /// - During blend-out: ramps from 1.0 to 0.0.
    /// - Finished: 0.0.
    pub fn blend_weight(&self) -> f32 {
        match self.phase {
            MontagePhase::BlendingIn => {
                if self.definition.blend_in <= 0.0 {
                    1.0
                } else {
                    (self.time / self.definition.blend_in).clamp(0.0, 1.0)
                }
            }
            MontagePhase::Playing => 1.0,
            MontagePhase::BlendingOut => {
                let blend_out_start = self.clip_duration - self.definition.blend_out;
                let elapsed = self.time - blend_out_start;
                if self.definition.blend_out <= 0.0 {
                    0.0
                } else {
                    1.0 - (elapsed / self.definition.blend_out).clamp(0.0, 1.0)
                }
            }
            MontagePhase::Finished => 0.0,
        }
    }

    /// Returns true if the montage is finished and should be removed.
    pub fn is_finished(&self) -> bool {
        self.phase == MontagePhase::Finished
    }

    /// Returns true if this montage can be interrupted by another montage.
    pub fn is_interruptible(&self) -> bool {
        self.definition.interruptible
    }
}

/// ECS component: active montage slot for an entity.
///
/// An entity can have at most one active montage at a time. When a montage
/// finishes or is interrupted, this component should be removed or replaced.
#[derive(Clone, Debug)]
pub struct MontagePlayer {
    pub active: Option<ActiveMontage>,
}

impl MontagePlayer {
    pub fn new() -> Self {
        Self { active: None }
    }

    /// Start playing a montage. Returns false if a non-interruptible montage is active.
    pub fn play(&mut self, definition: MontageDefinition, clip_duration: f32) -> bool {
        if let Some(ref current) = self.active
            && !current.is_interruptible()
            && !current.is_finished()
        {
            return false;
        }
        self.active = Some(ActiveMontage::new(definition, clip_duration));
        true
    }

    /// Stop the current montage immediately.
    pub fn stop(&mut self) {
        self.active = None;
    }

    /// Advance the active montage. Clears it automatically when finished.
    pub fn advance(&mut self, dt: f32) {
        if let Some(ref mut montage) = self.active {
            montage.advance(dt);
            if montage.is_finished() {
                self.active = None;
            }
        }
    }

    /// Returns the montage blend weight (0.0 if no montage is active).
    pub fn blend_weight(&self) -> f32 {
        self.active.as_ref().map_or(0.0, |m| m.blend_weight())
    }
}

impl Default for MontagePlayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attack_montage() -> MontageDefinition {
        MontageDefinition {
            name: "attack".into(),
            clip_index: 5,
            speed: 1.0,
            blend_in: 0.1,
            blend_out: 0.2,
            interruptible: true,
        }
    }

    #[test]
    fn montage_lifecycle() {
        let def = attack_montage();
        let mut montage = ActiveMontage::new(def, 1.0);

        // Starts in BlendingIn.
        assert_eq!(montage.phase, MontagePhase::BlendingIn);
        assert!(montage.blend_weight() < 1.0);

        // Advance past blend_in (0.1).
        montage.advance(0.15);
        assert_eq!(montage.phase, MontagePhase::Playing);
        assert!((montage.blend_weight() - 1.0).abs() < 0.01);

        // Advance to blend_out region (1.0 - 0.2 = 0.8).
        montage.advance(0.7);
        assert_eq!(montage.phase, MontagePhase::BlendingOut);
        assert!(montage.blend_weight() < 1.0);

        // Advance past end.
        montage.advance(0.5);
        assert_eq!(montage.phase, MontagePhase::Finished);
        assert!((montage.blend_weight() - 0.0).abs() < 0.01);
    }

    #[test]
    fn player_auto_clears_finished_montage() {
        let mut player = MontagePlayer::new();
        player.play(attack_montage(), 0.5);
        assert!(player.active.is_some());

        // Advance past the clip.
        player.advance(1.0);
        assert!(player.active.is_none());
    }

    #[test]
    fn non_interruptible_montage_blocks_new_play() {
        let mut non_interruptible = attack_montage();
        non_interruptible.interruptible = false;

        let mut player = MontagePlayer::new();
        assert!(player.play(non_interruptible, 1.0));

        // Try to interrupt — should fail.
        assert!(!player.play(attack_montage(), 1.0));
    }

    #[test]
    fn interruptible_montage_allows_replacement() {
        let mut player = MontagePlayer::new();
        player.play(attack_montage(), 1.0);

        let new_def = MontageDefinition {
            name: "dodge".into(),
            clip_index: 6,
            speed: 1.5,
            blend_in: 0.05,
            blend_out: 0.1,
            interruptible: true,
        };
        assert!(player.play(new_def, 0.8));
        assert_eq!(player.active.as_ref().unwrap().definition.name, "dodge");
    }

    #[test]
    fn zero_blend_in_skips_to_playing() {
        let def = MontageDefinition {
            name: "instant".into(),
            clip_index: 0,
            speed: 1.0,
            blend_in: 0.0,
            blend_out: 0.0,
            interruptible: true,
        };
        let montage = ActiveMontage::new(def, 1.0);
        assert_eq!(montage.phase, MontagePhase::Playing);
        assert!((montage.blend_weight() - 1.0).abs() < 0.01);
    }

    #[test]
    fn stop_clears_active_montage() {
        let mut player = MontagePlayer::new();
        player.play(attack_montage(), 1.0);
        assert!(player.active.is_some());
        player.stop();
        assert!(player.active.is_none());
    }
}
