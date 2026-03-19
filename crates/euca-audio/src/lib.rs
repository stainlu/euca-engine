//! Spatial audio for EucaEngine — components, resources, and systems.
//!
//! Uses [`kira`] for high-quality audio playback.
//!
//! # Quick start
//! ```ignore
//! // 1. Insert AudioEngine as world resource
//! world.insert_resource(AudioEngine::new().unwrap());
//!
//! // 2. Load a sound
//! let clip = engine.load("assets/explosion.ogg").unwrap();
//!
//! // 3. Spawn an entity with AudioSource
//! let e = world.spawn(AudioSource::spatial(clip, 20.0));
//! world.insert(e, LocalTransform(Transform::from_translation(pos)));
//!
//! // 4. Run audio_update_system each tick
//! audio_update_system(&world);
//! ```

use euca_ecs::World;
use euca_math::Vec3;
use euca_scene::GlobalTransform;
use kira::Tween;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle, StaticSoundSettings};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Clip Handle ──

/// Opaque handle to a loaded audio clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AudioClipHandle(pub u32);

// ── AudioEngine (World Resource) ──

/// Core audio engine — wraps kira's AudioManager and stores loaded clips.
pub struct AudioEngine {
    manager: kira::AudioManager,
    clips: HashMap<u32, StaticSoundData>,
    next_id: u32,
}

impl AudioEngine {
    /// Create a new audio engine with default settings.
    pub fn new() -> Result<Self, String> {
        let manager = kira::AudioManager::new(kira::AudioManagerSettings::default())
            .map_err(|e| format!("Failed to initialize audio: {e}"))?;
        Ok(Self {
            manager,
            clips: HashMap::new(),
            next_id: 0,
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

    /// Play a clip with given settings. Returns a playback handle.
    pub fn play(
        &mut self,
        clip: AudioClipHandle,
        volume: f32,
        looping: bool,
    ) -> Result<StaticSoundHandle, String> {
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

        let handle = self
            .manager
            .play(data)
            .map_err(|e| format!("Failed to play audio: {e}"))?;

        // Set initial volume
        let mut handle = handle;
        handle.set_volume(volume, Tween::default());

        Ok(handle)
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
    /// Internal: kira playback handle (None if not yet started).
    handle: Option<StaticSoundHandle>,
}

impl AudioSource {
    /// Create a non-spatial (global) audio source.
    pub fn global(clip: AudioClipHandle) -> Self {
        Self {
            clip,
            volume: 1.0,
            looping: false,
            spatial: false,
            max_distance: 50.0,
            playing: true,
            handle: None,
        }
    }

    /// Create a spatial audio source with a max audible distance.
    pub fn spatial(clip: AudioClipHandle, max_distance: f32) -> Self {
        Self {
            clip,
            volume: 1.0,
            looping: false,
            spatial: true,
            max_distance,
            playing: true,
            handle: None,
        }
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
}

// ── System ──

/// Each tick: starts playback for new sources, updates spatial volumes.
pub fn audio_update_system_mut(world: &mut World) {
    // Get listener position
    let listener_pos = {
        let query = euca_ecs::Query::<(&AudioListener, &GlobalTransform)>::new(world);
        query.iter().next().map(|(_, gt)| gt.0.translation)
    };
    let listener_pos = listener_pos.unwrap_or(Vec3::ZERO);

    // Collect source data
    struct SourceData {
        entity: euca_ecs::Entity,
        playing: bool,
        has_handle: bool,
        volume: f32,
        max_distance: f32,
        spatial: bool,
        clip: AudioClipHandle,
        pos: Option<Vec3>,
    }

    let sources: Vec<SourceData> = {
        let query = euca_ecs::Query::<(euca_ecs::Entity, &AudioSource)>::new(world);
        query
            .iter()
            .map(|(e, src)| {
                let pos = world.get::<GlobalTransform>(e).map(|gt| gt.0.translation);
                SourceData {
                    entity: e,
                    playing: src.playing,
                    has_handle: src.handle.is_some(),
                    volume: src.volume,
                    max_distance: src.max_distance,
                    spatial: src.spatial,
                    clip: src.clip,
                    pos,
                }
            })
            .collect()
    };

    for s in sources {
        if s.playing && !s.has_handle {
            // Start playback
            let handle = world
                .resource_mut::<AudioEngine>()
                .and_then(|eng| eng.play(s.clip, s.volume, false).ok());
            if let Some(handle) = handle
                && let Some(src) = world.get_mut::<AudioSource>(s.entity)
            {
                src.handle = Some(handle);
            }
        }

        if !s.playing {
            // Stop playback
            if s.has_handle
                && let Some(src) = world.get_mut::<AudioSource>(s.entity)
            {
                if let Some(ref mut h) = src.handle {
                    h.stop(Tween::default());
                }
                src.handle = None;
            }
            continue;
        }

        // Update spatial volume
        if s.spatial
            && let Some(src_pos) = s.pos
        {
            let distance = (src_pos - listener_pos).length();
            let attenuation = if distance >= s.max_distance {
                0.0
            } else {
                let t = 1.0 - (distance / s.max_distance);
                t * t // quadratic falloff
            };
            let final_volume = s.volume * attenuation;
            if let Some(src) = world.get_mut::<AudioSource>(s.entity)
                && let Some(ref mut h) = src.handle
            {
                h.set_volume(final_volume, Tween::default());
            }
        }
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

    #[test]
    fn audio_source_builder() {
        let src = AudioSource::spatial(AudioClipHandle(0), 30.0)
            .with_volume(0.5)
            .with_looping(true);
        assert_eq!(src.volume, 0.5);
        assert!(src.looping);
        assert!(src.spatial);
        assert_eq!(src.max_distance, 30.0);
    }
}
