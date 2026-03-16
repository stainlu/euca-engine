# Euca Engine — Full Playbook

Everything you can do with the engine, step by step.

## Prerequisites

- **Rust 1.85+** (edition 2024) — install via [rustup.rs](https://rustup.rs)
- **GPU** with Vulkan, Metal (macOS), or D3D12 (Windows) support
- **Git** for cloning

```bash
git clone https://github.com/stainlu/euca-engine.git
cd euca-engine
```

## Build & Test

```bash
# Build everything
cargo build --workspace

# Run all 86 tests
cargo test --workspace

# Check code quality
cargo clippy --workspace
cargo fmt --all -- --check
```

---

## 1. Hello Cubes — Your First Window

Three spinning PBR cubes with an orbiting camera.

```bash
cargo run -p euca-render --example hello_cubes
```

**What you'll see:** Red, green, and blue cubes spinning at different speeds, lit by a directional light.

**Controls:** Escape to quit.

---

## 2. Physics Demo — Objects Falling Under Gravity

Cubes and a sphere fall onto a ground plane with PBR materials.

```bash
cargo run -p euca-render --example physics_demo
```

**What you'll see:** 4 colored cubes + 1 gold sphere drop from above, bounce on a gray ground plane. Orbiting camera.

**What's happening:**
- Rapier3D physics steps every frame
- Rigid body positions write back to ECS LocalTransform
- Transform propagation computes GlobalTransform
- PBR renderer reads GlobalTransform + Material for each entity

---

## 3. glTF Viewer — Load 3D Models

Load and render any .glb/.gltf model file.

```bash
# First, download a test model
curl -sL "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Box/glTF-Binary/Box.glb" -o box.glb

# View it
cargo run -p euca-asset --example gltf_viewer -- box.glb
```

**What you'll see:** The loaded model rendered with PBR lighting on a gray ground plane, orbiting camera.

**Supports:** Any glTF 2.0 / .glb file with PBR metallic-roughness materials.

---

## 4. Texture Demo — Textures, Shadows & Sky

Demonstrates texture mapping, shadow mapping, and the procedural sky.

```bash
cargo run -p euca-render --example texture_demo
```

**What you'll see:** Textured objects lit by a directional light with real-time shadows, against a procedural gradient sky. GPU instancing is used for efficient rendering.

**What's happening:**
- Albedo texture maps sampled in the PBR shader
- Shadow mapping from directional light
- Procedural sky background (gradient, not a cubemap)
- HDR post-processing: bloom, ACES tone mapping, vignette
- GPU instancing for batched draw calls

---

## 5. Editor — Visual Scene Editor

The visual editor with 3D viewport, entity hierarchy, component inspector, and full content-creation tools.

```bash
cargo run -p euca-editor --example editor
```

**What you'll see:** Three-panel layout:
- **Left:** Entity hierarchy — click an entity to select it
- **Right:** Inspector — shows all components on the selected entity (transform, physics body, collider)
- **Top toolbar:** Play / Pause / Step / Stop buttons, entity creation buttons (+ Empty, + Cube, + Sphere), Save / Load buttons, entity count + tick counter
- **3D viewport:** PBR rendering with shadows, procedural sky, ground grid overlay

**Features:**
- **PBR rendering** with shadows and procedural sky in the 3D viewport
- **Grid overlay** on the ground plane for spatial reference
- **Transform gizmos** — red (X), green (Y), blue (Z) axis handles on the selected entity; click and drag to move
- **Click-to-select** with raycasting in the 3D viewport
- **Selection outline** — orange highlight around selected entity
- **Entity creation** — toolbar buttons to spawn Empty, Cube, or Sphere entities
- **Scene save/load** — Save and Load buttons in the toolbar, persists to `scene.json`
- **Undo/redo** — full command-based transaction system (Ctrl+Z / Ctrl+Y)
- **Play/Pause/Step/Stop** simulation controls

**How to use:**
1. Click an entity in the left panel or click directly in the 3D viewport to select
2. Use the transform gizmo handles to move the entity along X/Y/Z axes
3. Click **+ Cube** or **+ Sphere** to spawn new entities
4. Arrange your scene, then click **Save** to persist to `scene.json`
5. Click **Load** to restore a previously saved scene
6. Click **Play** — physics starts, entities fall under gravity
7. Click **Pause** — simulation freezes
8. Click **Step** — advance exactly one physics tick
9. Use **Ctrl+Z** to undo and **Ctrl+Y** to redo any action

---

## 6. Headless Server — For AI Agents

Run the engine without a window. AI agents control the simulation via HTTP API and CLI.

### Start the server

```bash
cargo run -p euca-agent --example headless_server
```

Server starts on `http://localhost:8080`. You'll see:
```
Spawned 3 entities
Euca Engine running in headless mode
Euca Agent Server listening on http://127.0.0.1:8080
```

### Use the CLI (in another terminal)

```bash
# Check engine status
cargo run -p euca-cli -- status

# See all entities and their positions
cargo run -p euca-cli -- observe

# Advance simulation by 10 ticks
cargo run -p euca-cli -- step --ticks 10

# Observe again — positions have changed
cargo run -p euca-cli -- observe

# Spawn a new entity at position (50, 0, 0)
cargo run -p euca-cli -- spawn --position "50,0,0"

# See available components and actions
cargo run -p euca-cli -- schema

# Reset the world (despawn all entities)
cargo run -p euca-cli -- reset
```

### Use curl directly

```bash
# Status
curl http://localhost:8080/

# Observe world state
curl -X POST http://localhost:8080/observe

# Step 100 ticks
curl -X POST http://localhost:8080/step -H 'Content-Type: application/json' -d '{"ticks": 100}'

# Spawn entity
curl -X POST http://localhost:8080/spawn -H 'Content-Type: application/json' -d '{"position": [0, 10, 0]}'

# Despawn entity (by ID and generation)
curl -X POST http://localhost:8080/despawn -H 'Content-Type: application/json' -d '{"entity_id": 0, "entity_generation": 0}'

# Reset world
curl -X POST http://localhost:8080/reset -H 'Content-Type: application/json' -d '{}'

# List components and actions
curl http://localhost:8080/schema
```

### AI Agent Integration

Any external AI agent (Claude Code, RL agent, custom bot) can control the engine:

```
1. Start headless server (cargo run -p euca-agent --example headless_server)
2. Agent observes:  POST /observe → JSON with entity positions
3. Agent decides:   (agent's own logic)
4. Agent acts:      POST /spawn, /despawn, or future /act endpoint
5. Agent steps:     POST /step → advance simulation
6. Repeat from 2
```

---

## API Reference

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/` | GET | — | Engine status (name, version, entity count, tick) |
| `/observe` | POST | `{}` | World state snapshot (all entities + positions) |
| `/step` | POST | `{"ticks": N}` | Advance simulation N ticks (default: 1) |
| `/spawn` | POST | `{"position": [x,y,z]}` | Create new entity (returns ID) |
| `/despawn` | POST | `{"entity_id": N, "entity_generation": N}` | Remove entity |
| `/reset` | POST | `{}` | Despawn all entities |
| `/schema` | GET | — | List registered components and actions |

---

## Project Structure

```
eucaengine/
├── crates/
│   ├── euca-ecs/          # Custom archetype-based ECS (51 tests)
│   ├── euca-math/         # Custom SIMD-ready math (26 tests, zero deps)
│   ├── euca-reflect/      # #[derive(Reflect)] proc macro
│   ├── euca-scene/        # Transform hierarchy + BFS propagation (3 tests)
│   ├── euca-core/         # App lifecycle, Plugin, Time (1 test)
│   ├── euca-render/       # wgpu PBR renderer (8 tests)
│   ├── euca-physics/      # Custom collision + raycasting (12 tests, zero deps)
│   ├── euca-asset/        # glTF 2.0 model loading (1 test)
│   ├── euca-input/        # InputState, ActionMap, InputSnapshot (4 tests)
│   ├── euca-net/          # Raw UDP networking (11 tests, zero async deps)
│   ├── euca-agent/        # HTTP API server for AI agents
│   └── euca-editor/       # egui editor with 3D viewport (4 tests)
├── tools/
│   └── euca-cli/          # CLI tool (euca observe, step, spawn, etc.)
├── examples/
│   ├── hello_cubes.rs     # Spinning PBR cubes
│   ├── physics_demo.rs    # Cubes falling with gravity
│   ├── texture_demo.rs    # Textures, shadows, procedural sky
│   ├── gltf_viewer.rs     # Load and render .glb files
│   ├── headless_server.rs # AI agent HTTP server
│   └── editor.rs          # Visual scene editor
├── DESIGN.md              # System design document
├── README.md              # Quick start
└── GUIDE.md               # This file
```

---

## Keyboard Shortcuts

| Key | Action | Where |
|-----|--------|-------|
| Escape | Quit | All examples |
| Delete | Despawn selected entity | Editor |
| F | Focus camera on selected entity | Editor |
| Ctrl+Z | Undo | Editor |
| Ctrl+Y | Redo | Editor |
| Right mouse + drag | Orbit camera | Editor / examples |
| Middle mouse + drag | Pan camera | Editor / examples |
| Scroll wheel | Zoom in/out | Editor / examples |
| Left click | Select entity / interact with gizmo | Editor |
| Play button | Start simulation | Editor |
| Pause button | Freeze simulation | Editor |
| Step button | Advance one tick | Editor |
| Stop button | Stop simulation | Editor |
