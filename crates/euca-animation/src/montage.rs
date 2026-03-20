//! Animation montages: interruptible one-shot animations that play on top
//! of the state machine (attacks, reloads, emotes).
//!
//! A montage overrides the state machine's output for a specific set of bones
//! (or all bones) for a limited duration, with blend-in and blend-out.

/// A montage definition: which clip to play, how to blend, which bones to affect.
#[derive(Clone, Debug)]
pub struct AnimationMontage {
    /// Index into the clip library.
    pub clip_index: usize,
    /// Playback speed.
    pub speed: f32,
    /// Duration of blend-in (seconds).
    pub blend_in: f32,
    /// Duration of blend-out (seconds).
    pub blend_out: f32,
    /// Which bones this montage affects. `None` = all bones.
    pub bone_mask: Option<Vec<usize>>,
}

/// Runtime state for an active montage instance.
#[derive(Clone, Debug)]
pub struct ActiveMontage {
    /// The montage definition.
    pub montage: AnimationMontage,
    /// Current playback time.
    pub time: f32,
    /// Total clip duration (cached from the clip library).
    pub clip_duration: f32,
    /// Whether this montage has been interrupted (begins blend-out immediately).
    pub interrupted: bool,
}

impl ActiveMontage {
    /// Start a new montage.
    pub fn new(montage: AnimationMontage, clip_duration: f32) -> Self {
        Self {
            montage,
            time: 0.0,
            clip_duration,
            interrupted: false,
        }
    }

    /// Advance the montage by `dt` seconds. Returns `true` when fully finished.
    pub fn advance(&mut self, dt: f32) -> bool {
        self.time += dt * self.montage.speed;
        self.is_finished()
    }

    /// Whether the montage has fully completed (including blend-out).
    pub fn is_finished(&self) -> bool {
        self.time >= self.clip_duration
    }

    /// The current blend weight of the montage (0.0 to 1.0).
    ///
    /// Ramps up during blend-in, holds at 1.0, ramps down during blend-out.
    pub fn weight(&self) -> f32 {
        if self.is_finished() {
            return 0.0;
        }

        let blend_in = self.montage.blend_in;
        let blend_out = self.montage.blend_out;

        // Blend-in phase
        let w_in = if blend_in > 0.0 && self.time < blend_in {
            self.time / blend_in
        } else {
            1.0
        };

        // Blend-out phase
        let blend_out_start = self.clip_duration - blend_out;
        let w_out = if blend_out > 0.0 && self.time > blend_out_start {
            let remaining = self.clip_duration - self.time;
            (remaining / blend_out).max(0.0)
        } else {
            1.0
        };

        w_in * w_out
    }

    /// Interrupt the montage (begin blend-out from current position).
    pub fn interrupt(&mut self) {
        if !self.interrupted {
            self.interrupted = true;
            // Move the clip_duration so blend-out happens from now
            self.clip_duration = self.time + self.montage.blend_out;
        }
    }

    /// Whether this montage affects a specific bone.
    pub fn affects_bone(&self, bone_index: usize) -> bool {
        match &self.montage.bone_mask {
            Some(mask) => mask.contains(&bone_index),
            None => true,
        }
    }
}

/// ECS component: manages the montage stack for an entity.
///
/// Only one montage plays at a time. Playing a new montage interrupts
/// the current one.
#[derive(Clone, Debug, Default)]
pub struct MontagePlayer {
    active: Option<ActiveMontage>,
}

impl MontagePlayer {
    pub fn new() -> Self {
        Self { active: None }
    }

    /// Play a montage. Interrupts any currently-playing montage.
    pub fn play(&mut self, montage: AnimationMontage, clip_duration: f32) {
        if let Some(ref mut current) = self.active {
            current.interrupt();
        }
        self.active = Some(ActiveMontage::new(montage, clip_duration));
    }

    /// Stop the current montage (with blend-out).
    pub fn stop(&mut self) {
        if let Some(ref mut current) = self.active {
            current.interrupt();
        }
    }

    /// Whether a montage is currently active (playing or blending out).
    pub fn is_playing(&self) -> bool {
        self.active.is_some()
    }

    /// Get the active montage (if any).
    pub fn active(&self) -> Option<&ActiveMontage> {
        self.active.as_ref()
    }

    /// Get a mutable reference to the active montage.
    pub fn active_mut(&mut self) -> Option<&mut ActiveMontage> {
        self.active.as_mut()
    }

    /// Advance the montage player. Removes finished montages.
    pub fn advance(&mut self, dt: f32) {
        if let Some(ref mut montage) = self.active {
            if montage.advance(dt) {
                self.active = None;
            }
        }
    }

    /// Get the current montage weight (0.0 if no montage active).
    pub fn weight(&self) -> f32 {
        self.active.as_ref().map_or(0.0, |m| m.weight())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_montage() -> AnimationMontage {
        AnimationMontage {
            clip_index: 5,
            speed: 1.0,
            blend_in: 0.1,
            blend_out: 0.1,
            bone_mask: None,
        }
    }

    #[test]
    fn montage_lifecycle() {
        let mut player = MontagePlayer::new();
        assert!(!player.is_playing());

        player.play(test_montage(), 1.0);
        assert!(player.is_playing());

        // Advance through blend-in
        player.advance(0.05);
        assert!(player.weight() > 0.0);
        assert!(player.weight() < 1.0);

        // Advance to full weight
        player.advance(0.1);
        assert!((player.weight() - 1.0).abs() < 1e-3);

        // Advance to blend-out region
        player.advance(0.8);
        assert!(player.weight() < 1.0);
        assert!(player.weight() > 0.0);

        // Advance past end
        player.advance(0.5);
        assert!(!player.is_playing());
    }

    #[test]
    fn montage_interrupt() {
        let mut player = MontagePlayer::new();
        player.play(test_montage(), 2.0);
        player.advance(0.5);

        player.stop();
        assert!(player.is_playing());

        // Advance through blend-out
        player.advance(0.2);
        assert!(!player.is_playing());
    }

    #[test]
    fn montage_replaces_previous() {
        let mut player = MontagePlayer::new();
        player.play(test_montage(), 1.0);
        player.advance(0.5);

        let mut new_montage = test_montage();
        new_montage.clip_index = 10;
        player.play(new_montage, 1.0);

        assert_eq!(player.active().unwrap().montage.clip_index, 10);
        assert!((player.active().unwrap().time).abs() < 1e-5);
    }

    #[test]
    fn bone_mask() {
        let montage = ActiveMontage::new(
            AnimationMontage {
                clip_index: 0,
                speed: 1.0,
                blend_in: 0.0,
                blend_out: 0.0,
                bone_mask: Some(vec![0, 1, 5]),
            },
            1.0,
        );

        assert!(montage.affects_bone(0));
        assert!(montage.affects_bone(1));
        assert!(!montage.affects_bone(2));
        assert!(montage.affects_bone(5));
    }

    #[test]
    fn no_bone_mask_affects_all() {
        let montage = ActiveMontage::new(test_montage(), 1.0);
        assert!(montage.affects_bone(0));
        assert!(montage.affects_bone(99));
    }

    #[test]
    fn blend_in_ramp() {
        let montage = ActiveMontage {
            montage: AnimationMontage {
                clip_index: 0,
                speed: 1.0,
                blend_in: 0.2,
                blend_out: 0.0,
                bone_mask: None,
            },
            time: 0.1,
            clip_duration: 1.0,
            interrupted: false,
        };
        assert!((montage.weight() - 0.5).abs() < 1e-5);
    }
}
