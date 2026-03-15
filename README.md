# Euca Engine

An ECS-first, agent-native game engine built in Rust.

## Core Pillars

1. **ECS Architecture** — Custom archetype-based Entity-Component-System, optimized for large-scale simulation
2. **Agent-Native** — External AI agents access the engine via CLI tools and HTTP API (no internal AI systems)
3. **Rust** — Ownership for safety, proc macros for reflection, zero-cost abstractions

## Quick Start

```bash
# Run the physics demo (PBR cubes falling under gravity)
cargo run -p euca-render --example physics_demo

# Run the editor
cargo run -p euca-editor --example editor

# Run the headless server (for AI agents)
cargo run -p euca-agent --example headless_server

# Use the CLI tool (while server is running)
cargo run -p euca-cli -- status
cargo run -p euca-cli -- observe
cargo run -p euca-cli -- step --ticks 10

# Load a glTF model
cargo run -p euca-asset --example gltf_viewer -- path/to/model.glb

# Run all tests
cargo test --workspace
```

## Crate Map

| Crate | Purpose |
|-------|---------|
| `euca-ecs` | Custom archetype-based ECS (Entity, Component, World, Query, Schedule) |
| `euca-math` | Math types wrapping glam (Vec2/3/4, Quat, Mat4, Transform, AABB) |
| `euca-reflect` | `#[derive(Reflect)]` proc macro for runtime type info |
| `euca-scene` | Transform hierarchy (LocalTransform, GlobalTransform, Parent/Children) |
| `euca-core` | App lifecycle, Plugin trait, Time resource, winit event loop |
| `euca-render` | wgpu PBR renderer (Cook-Torrance BRDF, materials, lights, meshes) |
| `euca-physics` | Rapier3D integration (RigidBody, Collider, physics step system) |
| `euca-asset` | glTF 2.0 model loading (meshes + PBR materials) |
| `euca-agent` | HTTP API server for external AI agents (axum + tokio) |
| `euca-editor` | egui-based visual editor (hierarchy, inspector, play/pause) |
| `euca-cli` | CLI tool for AI agents (`euca observe`, `euca step`, etc.) |

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
| ECS Runtime       |  Render (wgpu)    Physics (rapier3d)
| World > Archetype |  PBR + Forward+   RigidBody + Collider
| Query + Schedule  |  Material + Light  Gravity + Collision
+-------------------+
        |
+-- Editor (egui) --+
| Hierarchy panel    |
| Inspector panel    |
| Play/Pause/Step    |
+--------------------+
```

## License

MIT — see [LICENSE](LICENSE)
