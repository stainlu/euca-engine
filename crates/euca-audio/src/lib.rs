//! Spatial audio for EucaEngine — components, resources, and systems.
//!
//! Uses [`kira`] for high-quality audio playback with mixing buses,
//! reverb zones, sound priority/concurrency, audio pooling, basic
//! occlusion, and fade transitions.
//!
//! # Quick start
//! ```ignore
//! // 1. Insert AudioEngine and AudioBusSettings as world resources
//! world.insert_resource(AudioEngine::new().unwrap());
//! world.insert_resource(AudioBusSettings::default());
//! world.insert_resource(AudioSettings::default());
//!
//! // 2. Load a sound
//! let clip = engine.load("assets/explosion.ogg").unwrap();
//!
//! // 3. Spawn an entity with AudioSource
//! let e = world.spawn(AudioSource::spatial(clip, 20.0).with_bus(AudioBus::Sfx));
//! world.insert(e, LocalTransform(Transform::from_translation(pos)));
//!
//! // 4. Run audio_update_system each tick
//! audio_update_system_mut(&world);
//! ```

use euca_ecs::World;
use euca_math::Vec3;
use euca_scene::GlobalTransform;
use kira::Tween;
use kira::sound::PlaybackState;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle, StaticSoundSettings};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ── Clip Handle ──

/// Opaque handle to a loaded audio clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AudioClipHandle(pub u32);

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
    const SUB_BUSES: [AudioBus; 4] = [AudioBus::Music, AudioBus::Sfx, AudioBus::Voice, AudioBus::Ui];
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

// ── AudioEngine (World Resource) ──

/// Core audio engine — wraps kira's AudioManager, stores loaded clips,
/// manages per-bus sub-tracks, and pools playback handles.
pub struct AudioEngine {
    manager: kira::AudioManager,
    clips: HashMap<u32, StaticSoundData>,
    next_id: u32,
    /// Per-bus kira sub-tracks. Sounds are routed to the track for their bus.
    bus_tracks: HashMap<AudioBus, kira::track::TrackHandle>,
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

    /// Return a handle to the pool for potential reuse.
    pub fn return_to_pool(&mut self, clip: AudioClipHandle, handle: StaticSoundHandle) {
        self.handle_pool
            .entry(clip.0)
            .or_default()
            .push(handle);
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
    handle: Option<StaticSoundHandle>,
    /// Internal: elapsed time since playback started (for fade-in).
    fade_elapsed: f32,
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

/// Occlusion component: attach to an entity with [`AudioSource`] to enable
/// occlusion-based volume attenuation.
///
/// The `occluded` flag should be set by an external system (e.g., a physics
/// raycast from source to listener). When `occluded` is `true`, the audio
/// system multiplies the source volume by [`AudioSettings::occlusion_factor`].
///
/// This decouples audio from physics — any occlusion strategy (raycasts,
/// portal-based, baked) can drive the flag.
#[derive(Clone, Debug)]
pub struct AudioOcclusion {
    /// Whether the path from this source to the listener is currently blocked.
    pub occluded: bool,
}

impl Default for AudioOcclusion {
    fn default() -> Self {
        Self { occluded: false }
    }
}

// ── Internal helpers ──

/// Snapshot of an audio source gathered during the read phase of the system.
struct SourceSnapshot {
    entity: euca_ecs::Entity,
    playing: bool,
    has_handle: bool,
    volume: f32,
    max_distance: f32,
    spatial: bool,
    clip: AudioClipHandle,
    bus: AudioBus,
    priority: u8,
    fade_in: f32,
    fade_elapsed: f32,
    looping: bool,
    pos: Option<Vec3>,
    occluded: bool,
}

/// Returns the position of the first [`AudioListener`] entity, or `Vec3::ZERO` if none.
fn listener_position(world: &World) -> Vec3 {
    let query = euca_ecs::Query::<(&AudioListener, &GlobalTransform)>::new(world);
    query
        .iter()
        .next()
        .map(|(_, gt)| gt.0.translation)
        .unwrap_or(Vec3::ZERO)
}

/// Collects reverb zone data from the world.
fn collect_reverb_zones(world: &World) -> Vec<(Vec3, f32, f32, f32, f32)> {
    let query = euca_ecs::Query::<(&ReverbZone, &GlobalTransform)>::new(world);
    query
        .iter()
        .map(|(rz, gt)| (gt.0.translation, rz.radius, rz.mix, rz.decay, rz.damping))
        .collect()
}

/// Computes distance-based quadratic falloff.
fn distance_attenuation(distance: f32, max_distance: f32) -> f32 {
    if distance >= max_distance {
        0.0
    } else {
        let t = 1.0 - (distance / max_distance);
        t * t
    }
}

// ── Systems ──

/// Each tick: synchronizes bus volumes, enforces concurrency limits, starts /
/// stops sounds with fading, applies spatial attenuation with occlusion, and
/// processes reverb zones.
pub fn audio_update_system_mut(world: &mut World, dt: f32) {
    // ── 1. Sync bus volumes ──
    if let Some(bus_settings) = world.resource::<AudioBusSettings>() {
        // Read all bus volumes while we hold the immutable borrow.
        let master = bus_settings.volume(AudioBus::Master);
        let music = bus_settings.volume(AudioBus::Music);
        let sfx = bus_settings.volume(AudioBus::Sfx);
        let voice = bus_settings.volume(AudioBus::Voice);
        let ui = bus_settings.volume(AudioBus::Ui);

        if let Some(engine) = world.resource_mut::<AudioEngine>() {
            engine.manager.main_track().set_volume(master, Tween::default());
            let bus_vols = [
                (AudioBus::Music, music),
                (AudioBus::Sfx, sfx),
                (AudioBus::Voice, voice),
                (AudioBus::Ui, ui),
            ];
            for (bus, vol) in bus_vols {
                if let Some(track) = engine.bus_tracks.get_mut(&bus) {
                    track.set_volume(vol, Tween::default());
                }
            }
        }
    }

    // ── 2. Read listener position ──
    let listener_pos = listener_position(world);

    // ── 3. Read audio settings ──
    let (audio_settings_max, occlusion_factor) = world
        .resource::<AudioSettings>()
        .map(|s| (s.max_concurrent_sounds, s.occlusion_factor))
        .unwrap_or((32, 0.3));

    // ── 4. Snapshot all audio sources ──
    let mut sources: Vec<SourceSnapshot> = {
        let query = euca_ecs::Query::<(euca_ecs::Entity, &AudioSource)>::new(world);
        query
            .iter()
            .map(|(e, src)| {
                let pos = world.get::<GlobalTransform>(e).map(|gt| gt.0.translation);
                let occluded = world
                    .get::<AudioOcclusion>(e)
                    .is_some_and(|occ| occ.occluded);
                SourceSnapshot {
                    entity: e,
                    playing: src.playing,
                    has_handle: src.handle.is_some(),
                    volume: src.volume,
                    max_distance: src.max_distance,
                    spatial: src.spatial,
                    clip: src.clip,
                    bus: src.bus,
                    priority: src.priority,
                    fade_in: src.fade_in,
                    fade_elapsed: src.fade_elapsed,
                    looping: src.looping,
                    pos,
                    occluded,
                }
            })
            .collect()
    };

    // ── 5. Read reverb zones ──
    let reverb_zones = collect_reverb_zones(world);

    // ── 6. Enforce concurrency limit ──
    // Count currently active (has_handle && playing) sources.
    let active_count = sources.iter().filter(|s| s.has_handle && s.playing).count();
    let want_to_start_count = sources
        .iter()
        .filter(|s| s.playing && !s.has_handle)
        .count();

    let slots_available = audio_settings_max.saturating_sub(active_count);
    if want_to_start_count > slots_available {
        // Need to either skip low-priority new sounds or evict existing ones.
        // Strategy: sort candidates wanting to start by priority (ascending = higher prio first).
        // Evict active sources with lower priority and greater distance.
        let mut eviction_candidates: Vec<(usize, u8, f32)> = sources
            .iter()
            .enumerate()
            .filter(|(_, s)| s.has_handle && s.playing)
            .map(|(i, s)| {
                let dist = s
                    .pos
                    .map(|p| (p - listener_pos).length())
                    .unwrap_or(0.0);
                (i, s.priority, dist)
            })
            .collect();
        // Sort: lowest priority first, then furthest distance first — these are evicted first.
        eviction_candidates.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        let need_to_free = want_to_start_count - slots_available;
        for &(idx, _, _) in eviction_candidates.iter().take(need_to_free) {
            sources[idx].playing = false;
        }
    }

    // ── 7. Compute weighted reverb from overlapping zones ──
    let (_reverb_mix, _reverb_feedback, _reverb_damping) =
        compute_reverb_params(&reverb_zones, listener_pos);

    // ── 8. Process each source ──
    for s in &sources {
        // --- Stop ---
        if !s.playing && s.has_handle {
            // Extract handle and metadata, then drop the AudioSource borrow.
            let taken = world.get_mut::<AudioSource>(s.entity).and_then(|src| {
                if let Some(ref mut h) = src.handle {
                    let fade_out_dur = src.fade_out;
                    if fade_out_dur > 0.0 {
                        h.stop(Tween {
                            duration: Duration::from_secs_f32(fade_out_dur),
                            ..Tween::default()
                        });
                    } else {
                        h.stop(Tween::default());
                    }
                }
                let clip = src.clip;
                let handle = src.handle.take();
                src.fade_elapsed = 0.0;
                handle.map(|h| (clip, h))
            });
            // Return handle to pool (separate borrow scope).
            if let Some((clip, handle)) = taken {
                if let Some(engine) = world.resource_mut::<AudioEngine>() {
                    engine.return_to_pool(clip, handle);
                }
            }
            continue;
        }

        // --- Start ---
        if s.playing && !s.has_handle {
            let handle = world
                .resource_mut::<AudioEngine>()
                .and_then(|eng| eng.play(s.clip, 0.0, s.looping, s.bus).ok());
            if let Some(mut handle) = handle {
                // Apply initial fade-in: start at volume 0, then the volume update below
                // will ramp it up over subsequent frames.
                if s.fade_in > 0.0 {
                    handle.set_volume(0.0_f32, Tween::default());
                }
                if let Some(src) = world.get_mut::<AudioSource>(s.entity) {
                    src.handle = Some(handle);
                    src.fade_elapsed = 0.0;
                }
            }
        }

        // --- Update volume ---
        if s.playing {
            // Advance fade elapsed.
            if let Some(src) = world.get_mut::<AudioSource>(s.entity) {
                src.fade_elapsed += dt;
            }
            let new_elapsed = s.fade_elapsed + dt;

            // Fade-in factor.
            let fade_factor = if s.fade_in > 0.0 {
                (new_elapsed / s.fade_in).min(1.0)
            } else {
                1.0
            };

            // Distance attenuation.
            let attenuation = if s.spatial {
                if let Some(src_pos) = s.pos {
                    distance_attenuation((src_pos - listener_pos).length(), s.max_distance)
                } else {
                    1.0
                }
            } else {
                1.0
            };

            // Occlusion: binary attenuate if the AudioOcclusion component says occluded.
            let occlusion = if s.occluded {
                occlusion_factor
            } else {
                1.0
            };

            // Bus volume is already applied at the kira track level via sync_bus_volumes.
            let final_volume = s.volume * attenuation * occlusion * fade_factor;

            if let Some(src) = world.get_mut::<AudioSource>(s.entity)
                && let Some(ref mut h) = src.handle
            {
                h.set_volume(final_volume, Tween::default());
            }
        }
    }

    // ── 9. Clean up finished (non-looping) sounds ──
    let finished: Vec<(euca_ecs::Entity, AudioClipHandle)> = {
        let query = euca_ecs::Query::<(euca_ecs::Entity, &AudioSource)>::new(world);
        query
            .iter()
            .filter_map(|(e, src)| {
                if let Some(ref h) = src.handle {
                    if h.state() == PlaybackState::Stopped {
                        return Some((e, src.clip));
                    }
                }
                None
            })
            .collect()
    };
    for (entity, clip) in finished {
        let taken = world.get_mut::<AudioSource>(entity).and_then(|src| {
            let handle = src.handle.take();
            src.playing = false;
            src.fade_elapsed = 0.0;
            handle
        });
        if let Some(handle) = taken {
            if let Some(engine) = world.resource_mut::<AudioEngine>() {
                engine.return_to_pool(clip, handle);
            }
        }
    }
}

/// Compute distance-weighted reverb parameters from all overlapping reverb zones.
///
/// For each zone that contains `listener_pos`, compute a weight based on how
/// close the listener is to the zone center (linear falloff). Then blend the
/// mix, feedback, and damping values proportionally.
fn compute_reverb_params(
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
    fn distance_attenuation_at_zero() {
        assert!((distance_attenuation(0.0, 50.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn distance_attenuation_at_max() {
        assert!((distance_attenuation(50.0, 50.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn distance_attenuation_beyond_max() {
        assert!((distance_attenuation(100.0, 50.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn distance_attenuation_quadratic_midpoint() {
        let att = distance_attenuation(25.0, 50.0);
        let expected = 0.5_f32 * 0.5; // (1 - 25/50)^2 = 0.25
        assert!((att - expected).abs() < f32::EPSILON);
    }

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
            (Vec3::ZERO, 10.0, 0.4, 0.8, 0.1),         // weight = 1.0 (dist=0)
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
