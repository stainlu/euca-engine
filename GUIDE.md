# Euca Engine — Getting Started Guide

Everything you need to go from zero to running demos, understanding the
architecture, and building your own game.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Rust** | 1.89+ (Edition 2024) -- install via [rustup.rs](https://rustup.rs) |
| **GPU** | Vulkan, Metal (macOS), or D3D12 (Windows) support |
| **Git** | For cloning the repository |

**Platform support:**

| Platform | Renderer | Status |
|----------|----------|--------|
| macOS (Apple Silicon) | wgpu (default) or native Metal (`--features metal-native`) | Primary |
| macOS (Intel) | wgpu | Supported |
| Linux | wgpu (Vulkan) | Supported -- install `libasound2-dev` for audio |
| Windows | wgpu (D3D12 / Vulkan) | Supported |
| Web (WASM) | wgpu (WebGPU / WebGL2) | Supported via `wasm-pack` |

**Optional tools:**

- `wasm-pack` -- for WASM builds (`cargo install wasm-pack`)
- `curl` -- for testing the headless agent server

---

## Quick Start

```bash
# Clone and enter the repo
git clone https://github.com/stainlu/euca-engine.git
cd euca-engine

# Build the entire workspace
cargo build --workspace

# Run three spinning PBR cubes
cargo run -p euca-render --example hello_cubes
```

You should see a window with red, green, and blue cubes spinning under a
directional light, with an orbiting camera. Press **Escape** to quit.

### Build and test

```bash
# Run all tests
cargo test --workspace

# Check code quality
cargo clippy --workspace
cargo fmt --all -- --check
```

---

## Walkthrough: hello_cubes Explained

The `hello_cubes` example (`examples/hello_cubes.rs`) demonstrates every
foundational concept in Euca Engine. Here is the core setup, taken directly
from the example:

### 1. Create the ECS World and insert resources

```rust
use euca_core::Time;
use euca_ecs::World;
use euca_math::Vec3;
use euca_render::{AmbientLight, Camera};

let mut world = World::new();
world.insert_resource(Time::new());
world.insert_resource(Camera::new(Vec3::new(3.0, 3.0, 3.0), Vec3::ZERO));
world.insert_resource(AmbientLight::default());
```

`World` is the ECS container that holds every entity and resource. `Time`,
`Camera`, and `AmbientLight` are **resources** -- singleton values accessed by
systems.

### 2. Upload meshes and materials to the GPU

```rust
use euca_render::{Material, Mesh, Renderer, GpuContext};

let cube_mesh = renderer.upload_mesh(gpu, &Mesh::cube());
let red_mat = renderer.upload_material(gpu, &Material::red_plastic());
let green_mat = renderer.upload_material(gpu, &Material::green());
let blue_mat = renderer.upload_material(gpu, &Material::blue_plastic());
```

`Mesh::cube()` creates unit-cube vertex data. `Material::red_plastic()` and
friends produce PBR materials with preset albedo, metallic, and roughness
values. The `upload_*` methods transfer data to the GPU and return lightweight
handles.

### 3. Spawn entities with components

```rust
use euca_math::Transform;
use euca_render::{MaterialRef, MeshRenderer};
use euca_scene::{GlobalTransform, LocalTransform};

let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
world.insert(e, GlobalTransform::default());
world.insert(e, MeshRenderer { mesh: cube_mesh });
world.insert(e, MaterialRef { handle: red_mat });
```

Each entity is a bundle of components. `LocalTransform` positions it in the
scene. `GlobalTransform` is computed by the transform propagation system.
`MeshRenderer` and `MaterialRef` tell the renderer what to draw.

### 4. Update loop: rotate cubes, propagate transforms, render

```rust
use euca_ecs::{Entity, Query};
use euca_math::Quat;
use euca_render::{DirectionalLight, DrawCommand};

// Spin each cube around Y
let query = Query::<(Entity, &Spin)>::new(&world);
for (entity, spin) in query.iter() {
    if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
        lt.0.rotation = Quat::from_axis_angle(
            Vec3::new(0.0, 1.0, 0.0),
            elapsed * spin.speed,
        );
    }
}

// Propagate LocalTransform -> GlobalTransform
euca_scene::transform_propagation_system(&mut world);

// Collect draw commands and render
let draw_commands: Vec<DrawCommand> = {
    let query = Query::<(&GlobalTransform, &MeshRenderer, &MaterialRef)>::new(&world);
    query.iter().map(|(gt, mr, mat)| DrawCommand {
        mesh: mr.mesh,
        material: mat.handle,
        model_matrix: gt.0.to_matrix(),
        aabb: None,
    }).collect()
};
renderer.draw(gpu, &camera, &light, &ambient, &draw_commands);
```

### 5. Event loop (winit)

The example implements `winit::application::ApplicationHandler` to handle
window creation, resize, keyboard input (Escape to quit), and the
`RedrawRequested` event that drives each frame.

```rust
fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = HelloCubesApp::new();
    event_loop.run_app(&mut app).unwrap();
}
```

---

## Adding Physics

The `physics_demo` example (`examples/physics_demo.rs`) extends hello_cubes
with gravity, collision, and rigid bodies.

```bash
cargo run -p euca-render --example physics_demo
```

**What you'll see:** Colored cubes and a gold sphere drop from above, bounce
on a gray ground plane, with an orbiting camera.

### Key additions over hello_cubes

**1. Physics resources and components:**

```rust
use euca_physics::{Collider, PhysicsBody, PhysicsConfig, Velocity};

world.insert_resource(PhysicsConfig::new());

// Static ground plane
world.insert(ground, PhysicsBody::fixed());
world.insert(ground, Collider::aabb(10.0, 0.01, 10.0));

// Dynamic falling cube
world.insert(e, PhysicsBody::dynamic());
world.insert(e, Velocity::default());
world.insert(e, Collider::aabb(0.5, 0.5, 0.5).with_restitution(0.5));

// Dynamic sphere with bouncier restitution
world.insert(s, Collider::sphere(0.5).with_restitution(0.7));
```

**2. Physics step in the update loop:**

```rust
use euca_physics::physics_step_system;

// Step physics (applies gravity, resolves collisions, updates positions)
physics_step_system(&mut world);

// Propagate LocalTransform -> GlobalTransform (physics writes to LocalTransform)
euca_scene::transform_propagation_system(&mut world);
```

The physics system reads `PhysicsBody`, `Velocity`, and `Collider` components,
integrates forces (gravity), detects collisions, and writes updated positions
back to `LocalTransform`. The transform propagation system then computes
`GlobalTransform` for rendering.

---

## Multiplayer Basics

The engine includes a UDP networking layer with client-side prediction and
server reconciliation. Two examples demonstrate the full flow.

### Start the server

```bash
# Terminal 1: start the authoritative game server
cargo run -p euca-game --example server
```

The server runs at 60 ticks/sec on port 7777, with a physics world containing
a ground plane.

### Connect a client

```bash
# Terminal 2: connect a client (multiple clients supported)
cargo run -p euca-game --example client
```

The client renders cubes for each connected player. Your player is green;
other players are red. Use **WASD** to move.

### How it works

**Server** (`examples/server.rs`):

```rust
use euca_net::{GameServer, UdpTransport, ClientMessage, ServerMessage};

// Bind UDP socket
let transport = UdpTransport::bind("0.0.0.0:7777".parse().unwrap())?;

// Each tick: receive input, step physics, broadcast state
loop {
    state.receive_packets();      // Deserialize ClientMessage from each addr
    state.tick();                  // physics_step_system + transform propagation
    state.broadcast_state();       // Send StateDelta with all entity positions
}
```

**Client** (`examples/client.rs`):

```rust
use euca_net::{
    GameClient, ClientPrediction, reconcile_entity,
    record_prediction_for_entity, apply_prediction_system,
};

// Each frame:
// 1. Receive authoritative state from server
self.receive_packets();
// 2. Send local input to server
self.send_input();
// 3. Record and apply client-side prediction
record_prediction_for_entity(&mut world, entity, tick, input_snapshot);
apply_prediction_system(&mut world);
// 4. Render from the ECS world
self.render();
```

The server is authoritative: it runs physics and broadcasts `StateDelta`
messages. Clients predict movement locally for responsiveness, then reconcile
against the server state when it arrives.

---

## Running the MOBA Demo

The DotA-style MOBA client is a full gameplay demonstration with heroes,
abilities, items, creep waves, and a top-down camera.

```bash
# Build (first time may take a moment)
cargo build -p euca-game --example dota_client

# Run
cargo run -p euca-game --example dota_client
```

The demo loads the map from `levels/dota.json` and sets up a full MOBA
gameplay loop: click-to-move, QWER abilities, shop access, creep waves, and
day/night cycles.

**Systems involved:** `euca-gameplay` (heroes, items, combat, economy, AI),
`euca-physics` (collisions), `euca-render` (PBR rendering), `euca-nav`
(pathfinding), `euca-terrain` (map rendering).

---

## Running on Metal

On macOS with Apple Silicon, you can use the native Metal backend for maximum
GPU performance instead of the default wgpu backend.

```bash
# Run any example with native Metal
cargo run -p euca-game --example gpu_benchmark --features metal-native --release

# Metal-specific examples (require the metal-backend feature on euca-rhi)
cargo run -p euca-render --example metal_cubes --features euca-rhi/metal-backend
cargo run -p euca-render --example metal_stress --features euca-rhi/metal-backend
cargo run -p euca-render --example metal_mesh_stress --features euca-rhi/metal-backend
cargo run -p euca-render --example metal_fx_upscale --features euca-rhi/metal-backend
cargo run -p euca-render --example metal_combined --features euca-rhi/metal-backend
```

The `metal-native` feature on `euca-game` (or `euca-rhi/metal-backend` on
`euca-render`) switches the renderer from wgpu to native Metal via
`objc2-metal`, unlocking mesh shaders, MetalFX upscaling, and indirect command
buffers.

---

## Building for WASM

The engine can run in the browser via WebGPU. A ready-made demo lives in
`games/web-demo/`.

```bash
# Install wasm-pack if you haven't
cargo install wasm-pack

# Build the WASM package
cd games/web-demo
wasm-pack build --target web --out-dir pkg

# Serve locally (any static file server works)
python3 -m http.server 8080
```

Open `http://localhost:8080/` in a WebGPU-capable browser (Chrome 113+,
Edge 113+, or Firefox Nightly with `dom.webgpu.enabled`).

### How it works

`games/web-demo/src/lib.rs` implements the `WebApp` trait from `euca-web`:

```rust
use euca_web::{WebApp, euca_core::Time, euca_render::Camera};
use euca_web::euca_ecs::{Entity, Query, World};
use euca_web::euca_render::{GpuContext, Material, Mesh, Renderer};
use euca_web::euca_scene::{GlobalTransform, LocalTransform};

pub struct SpinningCubes;

impl WebApp for SpinningCubes {
    fn init(&mut self, world: &mut World, renderer: &mut Renderer, gpu: &GpuContext) {
        let cube = renderer.upload_mesh(gpu, &Mesh::cube());
        let red = renderer.upload_material(gpu, &Material::red_plastic());
        // ... spawn entities with LocalTransform, MeshRenderer, MaterialRef
    }

    fn update(&mut self, world: &mut World, _dt: f32) {
        // Spin cubes, orbit camera
    }
}

// WASM entry point
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() {
    euca_web::run_web_app::<SpinningCubes>();
}
```

The `euca-web` crate handles the browser event loop, canvas setup, and WebGPU
initialization. You implement `WebApp::init` and `WebApp::update`.

---

## Benchmarking

The `gpu_benchmark` example measures frame times with statistical rigor:
percentiles, standard deviation, min/max.

```bash
# Default: 1000 cubes, 60 warmup frames, 600 measured frames
cargo run -p euca-game --example gpu_benchmark --release

# Custom configuration via environment variables
BENCH_ENTITIES=5000 BENCH_FRAMES=1000 \
    cargo run -p euca-game --example gpu_benchmark --release

# Compare Metal vs wgpu
cargo run -p euca-game --example gpu_benchmark --release
cargo run -p euca-game --example gpu_benchmark --features metal-native --release

# Save results to CSV
BENCH_CSV=benchmark_results.csv \
    cargo run -p euca-game --example gpu_benchmark --release
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_ENTITIES` | 1000 | Number of cubes to spawn |
| `BENCH_WARMUP` | 60 | Warmup frames to discard |
| `BENCH_FRAMES` | 600 | Frames to measure |
| `BENCH_CSV` | (stdout) | Output CSV file path |

The benchmark spawns entities in a 3D grid, runs a deterministic camera orbit,
and reports avg/median/min/max/stdev/p1/p99 frame times in milliseconds and
FPS.

---

## All Examples

| # | Example | Command | Description |
|---|---------|---------|-------------|
| 1 | hello_cubes | `cargo run -p euca-render --example hello_cubes` | Three spinning PBR cubes |
| 2 | physics_demo | `cargo run -p euca-render --example physics_demo` | Cubes and sphere falling with gravity |
| 3 | texture_demo | `cargo run -p euca-render --example texture_demo` | Textures, shadows, procedural sky |
| 4 | gltf_viewer | `cargo run -p euca-asset --example gltf_viewer -- box.glb` | Load and render .glb/.gltf models |
| 5 | editor | `cargo run -p euca-editor --example editor` | Visual scene editor |
| 6 | headless_server | `cargo run -p euca-agent --example headless_server` | AI agent HTTP server |
| 7 | server | `cargo run -p euca-game --example server` | Multiplayer authoritative server |
| 8 | client | `cargo run -p euca-game --example client` | Multiplayer client |
| 9 | dota_client | `cargo run -p euca-game --example dota_client` | DotA-style MOBA demo |
| 10 | gpu_benchmark | `cargo run -p euca-game --example gpu_benchmark --release` | GPU performance benchmark |
| 11 | metal_cubes | `cargo run -p euca-render --example metal_cubes --features euca-rhi/metal-backend` | Native Metal cubes |
| 12 | stress_test | `cargo run -p euca-render --example stress_test` | Entity stress test |
| 13 | tiled_level | `cargo run -p euca-game --example tiled_level` | Tiled map level loading |

---

## glTF Viewer

Load and render any .glb/.gltf model file.

```bash
# Download a test model
curl -sL "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Box/glTF-Binary/Box.glb" -o box.glb

# View it
cargo run -p euca-asset --example gltf_viewer -- box.glb
```

Supports any glTF 2.0 / .glb file with PBR metallic-roughness materials.

---

## Texture Demo

Demonstrates texture mapping, shadow mapping, and the procedural sky.

```bash
cargo run -p euca-render --example texture_demo
```

**What's happening:**
- Albedo texture maps sampled in the PBR shader
- Shadow mapping from a directional light
- Procedural sky background (gradient, not a cubemap)
- HDR post-processing: bloom, ACES tone mapping, vignette
- GPU instancing for batched draw calls

---

## Editor

The visual editor with 3D viewport, entity hierarchy, component inspector,
and full content-creation tools.

```bash
cargo run -p euca-editor --example editor
```

**Layout:**
- **Left panel:** Entity hierarchy -- click to select
- **Right panel:** Inspector -- transform, physics body, collider components
- **Top toolbar:** Play/Pause/Step/Stop, entity creation (+ Empty, + Cube, + Sphere), Save/Load, entity count and tick counter
- **3D viewport:** PBR rendering with shadows, procedural sky, ground grid

**Features:**
- PBR rendering with shadows and procedural sky
- Grid overlay for spatial reference
- Transform gizmos -- red (X), green (Y), blue (Z) axis handles; drag to move
- Click-to-select with raycasting
- Selection outline (orange highlight)
- Entity creation via toolbar buttons
- Scene save/load (persists to `scene.json`)
- Undo/redo -- full command-based transaction system (Ctrl+Z / Ctrl+Y)
- Play/Pause/Step/Stop simulation controls

---

## Headless Server (AI Agents)

Run the engine without a window. AI agents control the simulation via HTTP API
and CLI.

### Start the server

```bash
cargo run -p euca-agent --example headless_server
```

Server starts on `http://localhost:8080`.

### Use the CLI (in another terminal)

```bash
cargo run -p euca-cli -- status              # Engine status
cargo run -p euca-cli -- observe             # All entities and positions
cargo run -p euca-cli -- step --ticks 10     # Advance 10 ticks
cargo run -p euca-cli -- spawn --position "50,0,0"  # Spawn entity
cargo run -p euca-cli -- schema              # Available components/actions
cargo run -p euca-cli -- reset               # Despawn all entities
```

### Use curl directly

```bash
curl http://localhost:8080/                                  # Status
curl -X POST http://localhost:8080/observe                   # World state
curl -X POST http://localhost:8080/step \
    -H 'Content-Type: application/json' -d '{"ticks": 100}' # Step 100 ticks
curl -X POST http://localhost:8080/spawn \
    -H 'Content-Type: application/json' \
    -d '{"position": [0, 10, 0]}'                            # Spawn entity
curl -X POST http://localhost:8080/reset \
    -H 'Content-Type: application/json' -d '{}'              # Reset world
curl http://localhost:8080/schema                            # Components/actions
```

### AI Agent Integration

Any external AI agent (Claude Code, RL agent, custom bot) can control the
engine:

```
1. Start headless server  (cargo run -p euca-agent --example headless_server)
2. Agent observes:         POST /observe -> JSON with entity positions
3. Agent decides:          (agent's own logic)
4. Agent acts:             POST /spawn, /despawn, or future /act endpoint
5. Agent steps:            POST /step -> advance simulation
6. Repeat from 2
```

### API Reference

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/` | GET | -- | Engine status (name, version, entity count, tick) |
| `/observe` | POST | `{}` | World state snapshot (all entities + positions) |
| `/step` | POST | `{"ticks": N}` | Advance simulation N ticks (default: 1) |
| `/spawn` | POST | `{"position": [x,y,z]}` | Create new entity (returns ID) |
| `/despawn` | POST | `{"entity_id": N, "entity_generation": N}` | Remove entity |
| `/reset` | POST | `{}` | Despawn all entities |
| `/schema` | GET | -- | List registered components and actions |

---

## Project Structure

```
eucaengine/
├── crates/
│   ├── euca-ecs/          # Custom archetype-based ECS
│   ├── euca-math/         # Custom SIMD-ready math (zero deps)
│   ├── euca-reflect/      # #[derive(Reflect)] proc macro
│   ├── euca-scene/        # Transform hierarchy + BFS propagation
│   ├── euca-core/         # App lifecycle, Plugin, Time
│   ├── euca-rhi/          # RenderDevice trait (wgpu + native Metal)
│   ├── euca-render/       # PBR renderer over RenderDevice
│   ├── euca-physics/      # Custom collision + raycasting (zero deps)
│   ├── euca-asset/        # glTF 2.0 model loading
│   ├── euca-input/        # InputState, ActionMap, InputSnapshot
│   ├── euca-net/          # Raw UDP networking (zero async deps)
│   ├── euca-agent/        # HTTP API server for AI agents
│   ├── euca-editor/       # egui editor with 3D viewport
│   ├── euca-gameplay/     # Health, combat, economy, abilities, AI
│   ├── euca-audio/        # Spatial audio, bus mixing, reverb
│   ├── euca-animation/    # Skeletal animation, state machines
│   ├── euca-particle/     # CPU particle emitters
│   ├── euca-terrain/      # Heightmap terrain, chunk LOD
│   ├── euca-nav/          # Grid navmesh, A* pathfinding
│   ├── euca-ai/           # Behavior trees, blackboard
│   ├── euca-ui/           # Runtime UI framework
│   ├── euca-script/       # Lua scripting (mlua)
│   ├── euca-services/     # Matchmaking, game servers
│   └── euca-web/          # WASM entry point for browsers
├── tools/
│   ├── euca-cli/          # CLI tool (euca observe, step, spawn, etc.)
│   └── euca-cook/         # Asset cooking pipeline
├── games/
│   ├── poker/             # Poker game
│   ├── poker-web/         # Poker web client
│   └── web-demo/          # WASM spinning cubes demo
├── services/
│   ├── matchmaking/       # Matchmaking service
│   ├── poker-server/      # Poker game server
│   └── level-mcp/         # Level MCP service
├── examples/              # All runnable examples
├── levels/                # Level data files
├── benches/               # ECS benchmarks
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
| WASD | Move player | Multiplayer client |
| QWER | Abilities | MOBA demo |
