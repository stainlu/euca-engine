//! Spatial audio for EucaEngine — components, resources, and systems.
//!
//! Uses [`kira`] for high-quality audio playback with mixing buses,
//! reverb zones, sound priority/concurrency, audio pooling, basic
//! occlusion, and fade transitions.
//!
//! # Quick start
//! ```ignore
//! // 1. Insert AudioEngine and AudioBusSettings as world resources
//! world.insert_resource(euca_audio::shared_engine(AudioEngine::new().unwrap()));
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
//!
//! # Forking semantics
//!
//! [`AudioEngine`] wraps `kira::AudioManager` which holds OS audio driver
//! handles that cannot be deep-cloned. It is therefore stored as a
//! [`Shared<AudioEngine>`] (alias for `Arc<Mutex<AudioEngine>>`) so that
//! [`World::clone`](euca_ecs::World::clone) forks share a single audio
//! driver. Audio mutations on a fork are visible to the parent.

mod engine;
mod reverb;
mod source;
mod systems;

use std::sync::{Arc, Mutex};

pub use engine::{AudioClipHandle, AudioEngine};
pub use reverb::{ReverbZone, query_reverb_for_listener};
pub use source::{
    AudioBus, AudioBusSettings, AudioListener, AudioOcclusion, AudioSettings, AudioSource,
};
pub use systems::audio_update_system_mut;

/// Shared, lockable handle to an [`AudioEngine`]. Registered as a world
/// resource so that it satisfies the `Clone` bound on resources.
pub type Shared<T> = Arc<Mutex<T>>;

/// Wrap an [`AudioEngine`] in a [`Shared`] handle for insertion as a
/// world resource.
pub fn shared_engine(engine: AudioEngine) -> Shared<AudioEngine> {
    Arc::new(Mutex::new(engine))
}
