# euca-game

Standalone game runner with project configuration and arena game mode.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `ProjectConfig` loaded from `.eucaproject.json` for window size, title, and game settings
- `WindowConfig` for resolution and display options
- Arena game mode module for quick prototyping
- Re-exports core engine crates (`euca_ecs`, `euca_math`, `euca_render`, `euca_physics`, `euca_scene`, `euca_net`)

## Usage

```rust
use euca_game::*;

let config = ProjectConfig::load(PROJECT_FILE_NAME).unwrap();
// Initialize window, renderer, and game loop from config
```

## License

MIT
