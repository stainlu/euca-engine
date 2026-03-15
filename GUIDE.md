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

## 4. Editor — Visual Scene Inspector

The visual editor with entity hierarchy, component inspector, and simulation controls.

```bash
cargo run -p euca-editor --example editor
```

**What you'll see:** Three-panel layout:
- **Left:** Entity hierarchy — click an entity to select it
- **Right:** Inspector — shows all components on the selected entity (transform, physics body, collider)
- **Top toolbar:** Play / Pause / Step buttons + entity count + tick counter

**How to use:**
1. Click an entity in the left panel (e.g., "Entity 0v0")
2. Inspector shows its LocalTransform position, PhysicsBody type, PhysicsCollider shape
3. Click **Play** — physics starts, entities fall under gravity
4. Click **Pause** — simulation freezes
5. Click **Step** — advance exactly one physics tick
6. Watch the position values change in the inspector as entities move

---

## 5. Headless Server — For AI Agents

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
│   ├── euca-ecs/          # Custom archetype-based ECS (43 tests)
│   ├── euca-math/         # Math types: Vec2/3/4, Quat, Mat4, Transform (23 tests)
│   ├── euca-reflect/      # #[derive(Reflect)] proc macro
│   ├── euca-scene/        # Transform hierarchy + propagation (3 tests)
│   ├── euca-core/         # App lifecycle, Plugin, Time (1 test)
│   ├── euca-render/       # wgpu PBR renderer (8 tests)
│   ├── euca-physics/      # Rapier3D integration (3 tests)
│   ├── euca-asset/        # glTF 2.0 model loading (1 test)
│   ├── euca-agent/        # HTTP API server for AI agents
│   └── euca-editor/       # egui visual editor (4 tests)
├── tools/
│   └── euca-cli/          # CLI tool (euca observe, step, spawn, etc.)
├── examples/
│   ├── hello_cubes.rs     # Spinning PBR cubes
│   ├── physics_demo.rs    # Cubes falling with gravity
│   ├── gltf_viewer.rs     # Load and render .glb files
│   ├── headless_server.rs # AI agent HTTP server
│   └── editor.rs          # Visual editor
├── DESIGN.md              # System design document
├── README.md              # Quick start
└── GUIDE.md               # This file
```

---

## Keyboard Shortcuts

| Key | Action | Where |
|-----|--------|-------|
| Escape | Quit | All examples |
| Play button | Start simulation | Editor |
| Pause button | Freeze simulation | Editor |
| Step button | Advance one tick | Editor |
