# Euca Engine

An ECS-first, agent-native game engine built in Rust.

## Core Pillars

1. **ECS Architecture** — Custom archetype-based Entity-Component-System, optimized for large-scale simulation
2. **Agent-Native** — AI agents control the engine via the `euca` CLI, backed by HTTP REST on port 3917. Authentication via [nit](https://github.com/newtype-ai/nit) Ed25519 identity. See [SKILL.md](SKILL.md) for the full interface.
3. **Rust** — Ownership for safety, proc macros for reflection, zero-cost abstractions

## Quick Start

```bash
# Run the physics demo (PBR cubes falling under gravity)
cargo run -p euca-render --example physics_demo

# Run the editor
cargo run -p euca-editor --example editor

# Run the headless server (for AI agents)
cargo run -p euca-agent --example headless_server

# Use the CLI tool (while editor is running — server starts on port 3917)
cargo run -p euca-cli -- status
cargo run -p euca-cli -- observe
cargo run -p euca-cli -- modify 1 --transform 3,2,0
cargo run -p euca-cli -- screenshot
cargo run -p euca-cli -- play

# Load a glTF model
cargo run -p euca-asset --example gltf_viewer -- path/to/model.glb

# Run all tests
cargo test --workspace
```

## Crate Map

| Crate | Purpose |
|-------|---------|
| `euca-ecs` | Custom archetype-based ECS (Entity, Component, World, Query, Schedule, Change Detection, Snapshots, par_for_each) |
| `euca-math` | Custom SIMD-ready math — Vec2/3/4, Quat, Mat4, Transform, AABB (zero external deps) |
| `euca-reflect` | `#[derive(Reflect)]` proc macro for runtime type info |
| `euca-scene` | Transform hierarchy (LocalTransform, GlobalTransform, Parent/Children BFS propagation) |
| `euca-core` | App lifecycle, Plugin trait, Time resource, winit event loop |
| `euca-render` | wgpu PBR renderer: Cook-Torrance BRDF, textures (albedo/UV/procedural), shadow mapping (2048px, PCF), procedural sky, GPU instancing (16K instances), HDR post-processing (bloom, ACES tone mapping, vignette) |
| `euca-physics` | Custom AABB/sphere collision, raycasting, gravity (zero external deps) |
| `euca-asset` | glTF 2.0 model loading (meshes + PBR materials) |
| `euca-input` | InputState, ActionMap, InputSnapshot (humans + AI agents) |
| `euca-net` | Raw UDP networking: PacketHeader, GameServer, GameClient, state replication protocol |
| `euca-agent` | HTTP API server + nit auth for AI agents (axum + tokio + ed25519-dalek) |
| `euca-editor` | egui editor: 3D viewport, hierarchy panel, inspector, play/pause/stop, transform gizmos, undo/redo, scene save/load (JSON), entity creation (+Empty/Cube/Sphere), grid overlay, keyboard shortcuts (Delete/F/Ctrl+Z/Ctrl+Y) |
| `euca-cli` | CLI for AI agents: observe, spawn, modify, despawn, step, play, pause, screenshot, auth |

## Agent Interface

The engine runs as a simulation server. AI agents connect via HTTP or CLI:

```bash
# Start headless server
cargo run -p euca-agent --example headless_server

# From another terminal (or from Claude Code / any AI agent):
curl -X POST http://localhost:8080/observe    # Get world state
curl -X POST http://localhost:8080/step -d '{"ticks": 10}'  # Advance simulation
curl -X POST http://localhost:8080/spawn -d '{"position": [0,5,0]}'  # Create entity
curl http://localhost:8080/schema              # List components & actions
```

## Architecture

```
External AI Agents (Claude Code, RL agents, etc.)
        | CLI / HTTP / WebSocket
        v
+-- Agent Interface (euca-agent) ------+
|  observe / act / step / save / reset |
+--------------------------------------+
        |
+-- Engine Core ----+
| ECS Runtime       |  Render (wgpu)    Physics (custom)
| World > Archetype |  PBR + Forward+   AABB + Sphere
| Query + Schedule  |  Material + Light  Gravity + Raycast
+-------------------+
        |
+-- Editor (egui) --+
| Hierarchy panel    |
| Inspector panel    |
| Play/Pause/Step    |
+--------------------+
```

## Published Crates

The following crates are available on [crates.io](https://crates.io):

| Crate | Description |
|-------|-------------|
| [`euca-math`](https://crates.io/crates/euca-math) | SIMD-ready Vec2/3/4, Quat, Mat4, Transform, AABB (zero deps) |
| [`euca-ecs`](https://crates.io/crates/euca-ecs) | Archetype-based ECS with Query, Schedule, Change Detection, Snapshots |

```bash
cargo add euca-math
cargo add euca-ecs
```

## License

MIT — see [LICENSE](LICENSE)
