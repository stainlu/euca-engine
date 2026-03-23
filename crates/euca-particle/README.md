# euca-particle

CPU particle emitters with configurable shapes, color interpolation, and renderer-agnostic output.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `ParticleEmitter` component with per-emitter particle pools (not per-entity)
- `EmitterConfig` with rate, lifetime, speed/size ranges, color start/end, gravity
- `EmitterShape` variants: Point, Sphere, Cone
- Texture atlas support with animated frame cycling
- Blend modes (AlphaBlend, Additive) for rendering
- `emit_particles_system` and `particle_update_system` for the simulation loop
- `collect_particle_data` and `ParticleRenderBatch` for renderer-agnostic draw output
- Pool cap enforcement via `max_particles`

## Usage

```rust
use euca_particle::*;

let emitter = ParticleEmitter::new(EmitterConfig {
    rate: 50.0,
    particle_lifetime: 2.0,
    speed_range: [2.0, 5.0],
    shape: EmitterShape::Cone { angle: 30.0 },
    ..Default::default()
});
```

## License

MIT
