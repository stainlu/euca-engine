# euca-audio

Spatial audio system powered by kira, with bus mixing, reverb zones, and occlusion.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `AudioEngine` resource wrapping kira for high-quality playback with handle pooling
- `AudioSource` component with spatial attenuation (quadratic falloff), looping, and priority
- `AudioBus` mixing (Master, Music, SFX, Voice, UI) with independent volume control
- `ReverbZone` component for distance-weighted reverb blending
- `AudioOcclusion` flag for physics-driven volume attenuation
- Fade-in / fade-out transitions on play and stop
- Concurrency limiting with priority-based eviction
- `AudioListener` component for listener positioning

## Usage

```rust
use euca_audio::*;

let mut engine = AudioEngine::new().unwrap();
let clip = engine.load("assets/explosion.ogg").unwrap();

let source = AudioSource::spatial(clip, 20.0)
    .with_bus(AudioBus::Sfx)
    .with_volume(0.8);
```

## License

MIT
