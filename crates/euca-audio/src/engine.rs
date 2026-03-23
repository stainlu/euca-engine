//! Core audio engine — clip loading, playback, handle pooling, and bus track management.

use crate::source::AudioBus;
use kira::Tween;
use kira::sound::PlaybackState;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle, StaticSoundSettings};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Clip Handle ──

/// Opaque handle to a loaded audio clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AudioClipHandle(pub u32);

// ── AudioEngine (World Resource) ──

/// Core audio engine — wraps kira's AudioManager, stores loaded clips,
/// manages per-bus sub-tracks, and pools playback handles.
pub struct AudioEngine {
    pub(crate) manager: kira::AudioManager,
    clips: HashMap<u32, StaticSoundData>,
    next_id: u32,
    /// Per-bus kira sub-tracks. Sounds are routed to the track for their bus.
    pub(crate) bus_tracks: HashMap<AudioBus, kira::track::TrackHandle>,
    /// Pool of completed handles per clip, available for reuse.
    handle_pool: HashMap<u32, Vec<StaticSoundHandle>>,
}

impl AudioEngine {
    /// Create a new audio engine with default settings.
    ///
    /// Creates a kira sub-track per [`AudioBus`] variant so that bus volume
    /// can be controlled independently at the mixer level.
    pub fn new() -> Result<Self, String> {
        let mut manager = kira::AudioManager::new(kira::AudioManagerSettings::default())
            .map_err(|e| format!("Failed to initialize audio: {e}"))?;

        let mut bus_tracks = HashMap::new();
        for bus in AudioBus::SUB_BUSES {
            let track = manager
                .add_sub_track(kira::track::TrackBuilder::new())
                .map_err(|e| format!("Failed to create bus track {bus:?}: {e}"))?;
            bus_tracks.insert(bus, track);
        }

        Ok(Self {
            manager,
            clips: HashMap::new(),
            next_id: 0,
            bus_tracks,
            handle_pool: HashMap::new(),
        })
    }

    /// Load an audio file from disk. Returns a handle for spawning AudioSources.
    pub fn load(&mut self, path: &str) -> Result<AudioClipHandle, String> {
        let data = StaticSoundData::from_file(path)
            .map_err(|e| format!("Failed to load audio '{path}': {e}"))?;
        let id = self.next_id;
        self.next_id += 1;
        self.clips.insert(id, data);
        log::info!("Loaded audio clip: {path} (handle: {id})");
        Ok(AudioClipHandle(id))
    }

    /// Play a clip on a specific bus. Reuses pooled handles when available.
    /// Returns a playback handle.
    pub fn play(
        &mut self,
        clip: AudioClipHandle,
        volume: f32,
        looping: bool,
        bus: AudioBus,
    ) -> Result<StaticSoundHandle, String> {
        // Try to reclaim a finished handle from the pool.
        if let Some(pool) = self.handle_pool.get_mut(&clip.0) {
            pool.retain(|h| h.state() != PlaybackState::Stopped);
        }

        let data = self
            .clips
            .get(&clip.0)
            .ok_or_else(|| format!("Audio clip {} not found", clip.0))?
            .clone();

        let settings = if looping {
            StaticSoundSettings::new().loop_region(..)
        } else {
            StaticSoundSettings::new()
        };

        let data = data.with_settings(settings);

        // Route to the bus track (or main track for Master).
        let mut handle = if bus == AudioBus::Master {
            self.manager
                .play(data)
                .map_err(|e| format!("Failed to play audio: {e}"))?
        } else {
            let track = self
                .bus_tracks
                .get_mut(&bus)
                .ok_or_else(|| format!("Bus track {bus:?} not found"))?;
            track
                .play(data)
                .map_err(|e| format!("Failed to play audio on bus {bus:?}: {e}"))?
        };

        handle.set_volume(volume, Tween::default());

        Ok(handle)
    }

    /// Unload a clip, freeing its sound data and any pooled playback handles.
    ///
    /// After this call the [`AudioClipHandle`] is invalidated — attempting to
    /// play it will return an error.
    pub fn unload_clip(&mut self, handle: AudioClipHandle) {
        if self.clips.remove(&handle.0).is_some() {
            self.handle_pool.remove(&handle.0);
            log::info!("Unloaded audio clip (handle: {})", handle.0);
        } else {
            log::warn!(
                "Attempted to unload unknown audio clip (handle: {})",
                handle.0
            );
        }
    }

    /// Return a handle to the pool for potential reuse.
    pub fn return_to_pool(&mut self, clip: AudioClipHandle, handle: StaticSoundHandle) {
        self.handle_pool.entry(clip.0).or_default().push(handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_clip_handle_equality() {
        let a = AudioClipHandle(0);
        let b = AudioClipHandle(0);
        let c = AudioClipHandle(1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
