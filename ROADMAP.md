# Euca Engine — V2 Roadmap

> **Status (March 2026):** Phases A, B, C (partial), and E are complete. All 11 CRITICAL architectural issues from the UE5 comparison review are resolved: mutable queries, parallel scheduling, capsule colliders, spatial hash broadphase, CCD, iterative constraint solver, transform dirty flags, multi-world agent pool, entity ownership, and reflection integration. Phase D (Game-Ready Features) is the next focus.

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

## What V1 didn't prove (updated)
- Can we build a real game on this engine?
- Can an AI agent actually play a game through the API?
- Does the engine scale to 100+ entities with networking?
- ~~Is the editor usable for content creation?~~ **YES** — transform gizmos, entity creation, scene save/load, undo/redo now working
- ~~Can someone outside of us use this engine?~~ **PARTIALLY** — euca-math and euca-ecs published to crates.io; full engine packaging still TODO

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

### Phase A: Rendering Quality ✅
Make it look like a real game engine, not a tech demo.

1. ✅ **Texture support** — albedo maps on materials, UV sampling in PBR shader
2. ✅ **Shadow mapping** — shadow maps for directional light
3. ✅ **Skybox** — procedural gradient sky (not cubemap)
4. ✅ **GPU instancing** — draw 1000+ objects with one draw call per mesh type
5. ✅ **Post-processing** — bloom, ACES tone mapping, vignette

### Phase B: Editor Maturity ✅
Make it usable for content creation, not just inspection.

1. ✅ **Transform gizmos** — visible translate handles (red=X, green=Y, blue=Z) on selected entity, click+drag
2. ✅ **Grid overlay** — reference grid on the ground plane
3. **Wireframe mode** — toggle view mode (not yet implemented)
4. ✅ **Undo/redo** — command-based transaction system (Ctrl+Z / Ctrl+Y)
5. ✅ **Keyboard shortcuts** — Delete (despawn), F (focus), Ctrl+Z/Y (undo/redo)
6. ✅ **Entity creation** — toolbar buttons to spawn Empty, Cube, or Sphere
7. ✅ **Scene save/load** — serialize world to JSON file (Save/Load buttons)

### Phase C: Publish to crates.io (partial) ✅
Get the community using our building blocks.

1. ✅ **euca-math v0.1.0** — published to crates.io, standalone SIMD-ready math crate (zero deps)
2. ✅ **euca-ecs v0.1.0** — published to crates.io, standalone archetype ECS with change detection + parallel iteration
3. ✅ **Clean public API** — doc comments added for published crates
4. **README per crate** — getting started, examples, API reference (TODO)
5. **Changelog** — semantic versioning, CHANGELOG.md (TODO)
6. **CI for crates** — publish workflow on tag (TODO)

### Phase D: Game-Ready Features
Fill gaps needed before any real game can be built.

1. **Audio** — spatial sound via cpal/kira
2. **Animation** — skeletal animation from glTF
3. **Particle effects** — basic particle system for projectiles, explosions
4. **AI agent plays** — connect agent to multiplayer server as a player
5. **Client prediction** — smooth movement without waiting for server response
6. **Interest management** — only send nearby entities to each client

### Phase E: Architecture Hardening ✅
All 11 CRITICAL issues from the UE5 comparison review resolved on 2026-03-17.

1. ✅ **Mutable queries** — `Query<&mut T>`, tuple expansion to 8, aliasing validation
2. ✅ **System access tracking** — SystemAccess, UnsafeWorldCell, Res/ResMut, IntoSystem\<Marker\>
3. ✅ **Parallel scheduling** — Greedy batch algorithm, std::thread::scope execution
4. ✅ **Reflection integration** — Reflect on 7 components, generic inspector display
5. ✅ **Broadphase** — Spatial hash grid (O(n²) → O(n * neighbors))
6. ✅ **CCD** — Sweep-test fast bodies against statics
7. ✅ **Capsule collider** — All collision pairs + raycast
8. ✅ **Constraint solver** — 4-iteration position-based, stable stacking
9. ✅ **Multi-world pool** — RwLock\<WorldPool\>, independent environments
10. ✅ **Entity ownership** — Owner component, permission checks
11. ✅ **Transform dirty flags** — Tick-based, O(N) → O(moved)

### Order
A → B → C → D → E (all complete except D partial, C partial)

### Success criteria
- ✅ Phase A done: engine renders a scene that looks professional (textures, shadows, procedural sky, HDR post-processing)
- ✅ Phase B done: can create a simple level entirely in the editor (spawn objects, arrange with gizmos, save/load, undo/redo)
- Phase C partial: `cargo add euca-ecs euca-math` works; README per crate, changelog, and CI still TODO
- Phase D in progress: game features (audio, animation, particles, client prediction still TODO)
- ✅ Phase E done: all 11 CRITICAL issues from UE5 comparison resolved
