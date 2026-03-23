# euca-agent

HTTP API server for external engine control: entity management, simulation, camera, HUD, and authentication.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `AgentServer` built on axum with 72+ REST endpoints for full engine control
- `AgentBridge` connecting HTTP requests to the ECS world
- `SharedWorld` for thread-safe world access from async handlers
- nit-based authentication (`AgentId`, `Owner`, `Persistent` markers)
- `EngineControl` for simulation play/pause/step/reset
- `CameraOverride` for programmatic camera positioning
- `ScreenshotChannel` for viewport capture
- HUD canvas system for text, bars, and shapes
- Level loading via `load_level_into_world`

## Usage

```rust
use euca_agent::*;

let server = AgentServer::new(shared_world, 3917);
server.start().await;

// Then from any HTTP client:
// POST /entity/create { "mesh": "cube", "position": [0, 2, 0] }
// POST /sim/play
```

## License

MIT
