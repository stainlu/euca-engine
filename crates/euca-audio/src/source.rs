//! Audio source components, mixing buses, and settings.

use crate::engine::AudioClipHandle;
use kira::sound::static_sound::StaticSoundHandle;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Audio Bus ──

/// Logical mixing bus that audio sources are routed through.
///
/// Each bus has an independent volume that multiplies with the source volume
/// and distance attenuation. All buses feed into `Master`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AudioBus {
    /// Master output — all other buses feed into this.
    Master,
    /// Background music.
    Music,
    /// Sound effects (default for new sources).
    #[default]
    Sfx,
    /// Voice / dialogue.
    Voice,
    /// UI sounds (click, hover, etc.).
    Ui,
}

impl AudioBus {
    /// All bus variants in declaration order (excluding Master).
    pub(crate) const SUB_BUSES: [AudioBus; 4] = [
        AudioBus::Music,
        AudioBus::Sfx,
        AudioBus::Voice,
        AudioBus::Ui,
    ];
}

/// Per-bus volume levels (world resource).
///
/// Final volume = `source.volume * distance_attenuation * bus_volume * master_volume`.
pub struct AudioBusSettings {
    volumes: HashMap<AudioBus, f32>,
}

impl Default for AudioBusSettings {
    fn default() -> Self {
        let mut volumes = HashMap::new();
        volumes.insert(AudioBus::Master, 1.0);
        volumes.insert(AudioBus::Music, 1.0);
        volumes.insert(AudioBus::Sfx, 1.0);
        volumes.insert(AudioBus::Voice, 1.0);
        volumes.insert(AudioBus::Ui, 1.0);
        Self { volumes }
    }
}

impl AudioBusSettings {
    /// Get the volume for a bus (0.0 - 1.0).
    pub fn volume(&self, bus: AudioBus) -> f32 {
        self.volumes.get(&bus).copied().unwrap_or(1.0)
    }

    /// Set the volume for a bus (clamped to 0.0 - 1.0).
    pub fn set_volume(&mut self, bus: AudioBus, volume: f32) {
        self.volumes.insert(bus, volume.clamp(0.0, 1.0));
    }
}

// ── Audio Settings ──

/// Global audio configuration (world resource).
pub struct AudioSettings {
    /// Maximum number of sounds playing simultaneously.
    pub max_concurrent_sounds: usize,
    /// Volume multiplier when a sound is occluded (0.0 - 1.0).
    pub occlusion_factor: f32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            max_concurrent_sounds: 32,
            occlusion_factor: 0.3,
        }
    }
}

// ── Components ──

/// Marks the entity whose position defines "where the listener is" (typically the camera).
#[derive(Clone, Debug, Default)]
pub struct AudioListener;

/// An active sound source attached to an entity.
pub struct AudioSource {
    /// Which clip to play.
    pub clip: AudioClipHandle,
    /// Mixing bus this source is routed to.
    pub bus: AudioBus,
    /// Base volume (0.0 - 1.0).
    pub volume: f32,
    /// Whether to loop.
    pub looping: bool,
    /// If true, volume attenuates with distance from AudioListener.
    pub spatial: bool,
    /// Max distance at which the sound is audible (spatial only).
    pub max_distance: f32,
    /// Whether the source should be playing.
    pub playing: bool,
    /// Priority: 0 = highest, 255 = lowest.
    /// When at the concurrency limit the lowest-priority, most-distant sound is evicted.
    pub priority: u8,
    /// Duration (seconds) over which volume ramps from 0 to target on play start.
    pub fade_in: f32,
    /// Duration (seconds) over which volume ramps to 0 when stopping.
    pub fade_out: f32,
    /// Internal: kira playback handle (None if not yet started).
    pub(crate) handle: Option<StaticSoundHandle>,
    /// Internal: elapsed time since playback started (for fade-in).
    pub(crate) fade_elapsed: f32,
}

impl AudioSource {
    /// Create a non-spatial (global) audio source.
    pub fn global(clip: AudioClipHandle) -> Self {
        Self {
            clip,
            bus: AudioBus::Sfx,
            volume: 1.0,
            looping: false,
            spatial: false,
            max_distance: 50.0,
            playing: true,
            priority: 128,
            fade_in: 0.0,
            fade_out: 0.0,
            handle: None,
            fade_elapsed: 0.0,
        }
    }

    /// Create a spatial audio source with a max audible distance.
    pub fn spatial(clip: AudioClipHandle, max_distance: f32) -> Self {
        Self {
            clip,
            bus: AudioBus::Sfx,
            volume: 1.0,
            looping: false,
            spatial: true,
            max_distance,
            playing: true,
            priority: 128,
            fade_in: 0.0,
            fade_out: 0.0,
            handle: None,
            fade_elapsed: 0.0,
        }
    }

    /// Set the mixing bus (builder pattern).
    pub fn with_bus(mut self, bus: AudioBus) -> Self {
        self.bus = bus;
        self
    }

    /// Set volume (builder pattern).
    pub fn with_volume(mut self, volume: f32) -> Self {
        self.volume = volume;
        self
    }

    /// Set looping (builder pattern).
    pub fn with_looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    /// Set priority (builder pattern). 0 = highest, 255 = lowest.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set fade-in duration in seconds (builder pattern).
    pub fn with_fade_in(mut self, seconds: f32) -> Self {
        self.fade_in = seconds;
        self
    }

    /// Set fade-out duration in seconds (builder pattern).
    pub fn with_fade_out(mut self, seconds: f32) -> Self {
        self.fade_out = seconds;
        self
    }

    /// Whether this source currently has an active kira playback handle.
    pub fn is_active(&self) -> bool {
        self.handle.is_some()
    }
}

/// Occlusion component: attach to an entity with [`AudioSource`] to enable
/// occlusion-based volume attenuation.
///
/// The `occluded` flag should be set by an external system (e.g., a physics
/// raycast from source to listener). When `occluded` is `true`, the audio
/// system multiplies the source volume by [`AudioSettings::occlusion_factor`].
///
/// This decouples audio from physics — any occlusion strategy (raycasts,
/// portal-based, baked) can drive the flag.
#[derive(Clone, Debug, Default)]
pub struct AudioOcclusion {
    /// Whether the path from this source to the listener is currently blocked.
    pub occluded: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_source_builder() {
        let src = AudioSource::spatial(AudioClipHandle(0), 30.0)
            .with_volume(0.5)
            .with_looping(true)
            .with_bus(AudioBus::Music)
            .with_priority(10)
            .with_fade_in(0.5)
            .with_fade_out(1.0);
        assert_eq!(src.volume, 0.5);
        assert!(src.looping);
        assert!(src.spatial);
        assert_eq!(src.max_distance, 30.0);
        assert_eq!(src.bus, AudioBus::Music);
        assert_eq!(src.priority, 10);
        assert!((src.fade_in - 0.5).abs() < f32::EPSILON);
        assert!((src.fade_out - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn audio_source_global_defaults() {
        let src = AudioSource::global(AudioClipHandle(1));
        assert!(!src.spatial);
        assert!(src.playing);
        assert_eq!(src.bus, AudioBus::Sfx);
        assert_eq!(src.priority, 128);
        assert!((src.fade_in).abs() < f32::EPSILON);
        assert!((src.fade_out).abs() < f32::EPSILON);
    }

    #[test]
    fn audio_bus_default_is_sfx() {
        assert_eq!(AudioBus::default(), AudioBus::Sfx);
    }

    #[test]
    fn bus_settings_default_all_one() {
        let settings = AudioBusSettings::default();
        assert!((settings.volume(AudioBus::Master) - 1.0).abs() < f32::EPSILON);
        assert!((settings.volume(AudioBus::Music) - 1.0).abs() < f32::EPSILON);
        assert!((settings.volume(AudioBus::Sfx) - 1.0).abs() < f32::EPSILON);
        assert!((settings.volume(AudioBus::Voice) - 1.0).abs() < f32::EPSILON);
        assert!((settings.volume(AudioBus::Ui) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn bus_settings_set_clamps() {
        let mut settings = AudioBusSettings::default();
        settings.set_volume(AudioBus::Music, 1.5);
        assert!((settings.volume(AudioBus::Music) - 1.0).abs() < f32::EPSILON);
        settings.set_volume(AudioBus::Music, -0.5);
        assert!((settings.volume(AudioBus::Music)).abs() < f32::EPSILON);
        settings.set_volume(AudioBus::Music, 0.75);
        assert!((settings.volume(AudioBus::Music) - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn audio_settings_defaults() {
        let settings = AudioSettings::default();
        assert_eq!(settings.max_concurrent_sounds, 32);
        assert!((settings.occlusion_factor - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn audio_occlusion_defaults_not_occluded() {
        let occ = AudioOcclusion::default();
        assert!(!occ.occluded);
    }

    #[test]
    fn audio_occlusion_set_occluded() {
        let occ = AudioOcclusion { occluded: true };
        assert!(occ.occluded);
    }
}
