# Euca Engine — V2 Roadmap

## What we built (V1 recap)

**16 crates, 121+ tests, MIT license, custom everything on the critical path.**

| Layer | What | Custom? |
|-------|------|---------|
| ECS | Archetype storage, generational entities, queries, schedule, change detection, par_for_each, snapshots | Yes |
| Math | Vec2/3/4, Quat, Mat4, Transform, AABB — SIMD-ready, zero deps | Yes |
| Physics | AABB/sphere collision, raycasting, gravity, push-out resolution | Yes |
| Networking | Raw UDP transport, PacketHeader (seq/ack/ack_bits), GameServer/Client, bincode protocol | Yes |
| Input | InputState, ActionMap, timestamped InputSnapshot | Yes |
| Rendering | wgpu PBR (Cook-Torrance BRDF), materials, lights, meshes | wgpu dep |
| Editor | 3D viewport, click-to-select with raycasting, selection outline, editable transforms, orbit/pan/zoom camera, FPS counter, play/pause/step/stop | egui dep |
| Agent API | HTTP server (axum), CLI tool, headless mode | axum dep |
| Multiplayer | Working server + client demo over raw UDP | Proven |

## What V1 proved
- The architecture works: ECS → Physics → Networking → Rendering
- Multiplayer works: two clients see each other over UDP
- The Ghostty philosophy works: custom math/physics/networking with zero heavy deps
- Agent-native design works: headless server + HTTP API + CLI

## What V1 didn't prove
- Can we build a real game on this engine?
- Can an AI agent actually play a game through the API?
- Does the engine scale to 100+ entities with networking?
- Is the editor usable for content creation?
- Can someone outside of us use this engine?

## V2 Goals

**Theme: From engine to game platform.**

### Milestone 1: "Playable Game" (highest priority)
Build the simplest possible competitive multiplayer game that both humans and AI agents can play. This forces every engine system to work together under real conditions.

**Game concept:** Top-down arena — 2-4 players, move with WASD, shoot projectiles, last player standing wins. Simple enough to build in days, complex enough to test everything.

**What this requires:**
- Game logic system (health, damage, respawn)
- Projectile system (spawn, move, collision → damage)
- Game state (score, round management, win condition)
- HUD overlay (health bar, score)
- Better camera (top-down fixed or following)
- Sound effects (optional, can add later)

### Milestone 2: "AI Agent Plays"
Connect an AI agent to the game server as a player. Prove the core differentiator.

**What this requires:**
- Agent observes game state via HTTP API (positions, health, projectiles)
- Agent sends actions (move direction, shoot direction) via HTTP API
- Agent plays the same game as humans, through different interface
- Benchmark: AI agent vs human, same rules

### Milestone 3: "Others Can Use It"
Make the engine usable by people who aren't us.

**What this requires:**
- Clean public API with documentation
- Getting started tutorial
- Publish euca-ecs and euca-math to crates.io
- Example game as reference implementation
- Contribution guidelines

### Milestone 4: "Production Quality"
Polish for real game development.

**What this requires:**
- Texture support in renderer
- Shadow mapping
- Audio (spatial sound)
- Scene save/load
- Editor: transform gizmos, undo/redo, content browser
- Performance profiling and optimization
- Mobile deployment (Android/iOS)

## Priority order
1. **Playable Game** — proves the engine works for real
2. **AI Agent Plays** — proves the differentiator
3. **Others Can Use It** — grows the community
4. **Production Quality** — polish for shipping

## V2 Execution Plan

### Phase A: Rendering Quality
Make it look like a real game engine, not a tech demo.

1. **Texture support** — albedo maps on materials, UV sampling in PBR shader
2. **Shadow mapping** — cascaded shadow maps for directional light
3. **Skybox** — cubemap background instead of solid color
4. **GPU instancing** — draw 1000+ objects with one draw call per mesh type
5. **Post-processing** — bloom, FXAA, configurable tone mapping

### Phase B: Editor Maturity
Make it usable for content creation, not just inspection.

1. **Transform gizmos** — visible translate/rotate/scale handles on selected entity
2. **Grid overlay** — reference grid on the ground
3. **Wireframe mode** — toggle view mode
4. **Undo/redo** — command-based transaction system
5. **Keyboard shortcuts** — Del (delete), Ctrl+D (duplicate), Ctrl+Z/Y (undo/redo), F (focus)
6. **Entity creation** — right-click context menu to spawn entities
7. **Scene save/load** — serialize world to file

### Phase C: Publish to crates.io
Get the community using our building blocks.

1. **euca-math** — standalone SIMD-ready math crate (zero deps, useful to anyone)
2. **euca-ecs** — standalone archetype ECS with change detection + parallel iteration
3. **Clean public API** — remove `pub(crate)`, add proper doc comments
4. **README per crate** — getting started, examples, API reference
5. **Changelog** — semantic versioning, CHANGELOG.md
6. **CI for crates** — publish workflow on tag

### Phase D: Game-Ready Features
Fill gaps needed before any real game can be built.

1. **Audio** — spatial sound via cpal/kira
2. **Animation** — skeletal animation from glTF
3. **Particle effects** — basic particle system for projectiles, explosions
4. **AI agent plays** — connect agent to multiplayer server as a player
5. **Client prediction** — smooth movement without waiting for server response
6. **Interest management** — only send nearby entities to each client

### Order
A → B → C → D (but can overlap — rendering and editor are independent)

### Success criteria
- Phase A done: engine renders a scene that looks professional (textures, shadows, skybox)
- Phase B done: can create a simple level entirely in the editor (spawn objects, arrange, save)
- Phase C done: `cargo add euca-ecs euca-math` works, someone builds something with it
- Phase D done: a real multiplayer game can be built and played by humans + AI agents
