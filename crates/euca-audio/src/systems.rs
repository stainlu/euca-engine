//! Audio update system — bus synchronization, concurrency, spatial attenuation,
//! fading, occlusion, and reverb processing.

use crate::engine::{AudioClipHandle, AudioEngine};
use crate::reverb::{collect_reverb_zones, compute_reverb_params, listener_position};
use crate::source::{AudioBus, AudioBusSettings, AudioOcclusion, AudioSettings, AudioSource};
use euca_ecs::World;
use euca_math::Vec3;
use euca_scene::GlobalTransform;
use kira::Tween;
use kira::sound::PlaybackState;
use std::time::Duration;

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

/// Computes distance-based quadratic falloff.
fn distance_attenuation(distance: f32, max_distance: f32) -> f32 {
    if distance >= max_distance {
        0.0
    } else {
        let t = 1.0 - (distance / max_distance);
        t * t
    }
}

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
            engine
                .manager
                .main_track()
                .set_volume(master, Tween::default());
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
                let dist = s.pos.map(|p| (p - listener_pos).length()).unwrap_or(0.0);
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
            if let Some((clip, handle)) = taken
                && let Some(engine) = world.resource_mut::<AudioEngine>()
            {
                engine.return_to_pool(clip, handle);
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
            let occlusion = if s.occluded { occlusion_factor } else { 1.0 };

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
                if let Some(ref h) = src.handle
                    && h.state() == PlaybackState::Stopped
                {
                    return Some((e, src.clip));
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
        if let Some(handle) = taken
            && let Some(engine) = world.resource_mut::<AudioEngine>()
        {
            engine.return_to_pool(clip, handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
