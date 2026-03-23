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

mod engine;
mod reverb;
mod source;
mod systems;

pub use engine::{AudioClipHandle, AudioEngine};
pub use reverb::{ReverbZone, query_reverb_for_listener};
pub use source::{
    AudioBus, AudioBusSettings, AudioListener, AudioOcclusion, AudioSettings, AudioSource,
};
pub use systems::audio_update_system_mut;
