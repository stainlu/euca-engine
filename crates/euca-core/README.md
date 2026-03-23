# euca-core

Application lifecycle, plugin system, time resource, and frame profiler.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `App` for engine initialization and main loop orchestration
- `Plugin` trait for modular feature registration
- `Time` resource tracking delta time, elapsed time, and frame count
- `Profiler` with scoped `ProfileSection` for per-frame performance measurement
- Re-exports `winit` for downstream window/event handling

## Usage

```rust
use euca_core::*;

let mut app = App::new();
app.add_plugin(MyGamePlugin);
app.run();
```

## License

MIT
