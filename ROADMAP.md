# Euca Engine — V2 Roadmap

> **Status (March 2026):** All phases (A–G) are complete. 24 crates, 850+ tests. AI agents can build and observe games via 30 CLI command groups / 75+ HTTP endpoints. Next: prove end-to-end (playable game, AI agent plays).

## What we built (V1 recap)

**24 crates, 850+ tests, MIT license, custom everything on the critical path.**

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
- ~~Can we build a real game on this engine?~~ **YES** — MOBA demo with combat, teams, abilities, economy, AI, waves built entirely from CLI commands
- Can an AI agent actually play a game through the API? **(API complete, end-to-end proof pending)**
- ~~Does the engine scale to 100+ entities with networking?~~ **YES** — 10,000-entity stress test at 75 FPS (physics + rendering), 50K render-only at 50 FPS. Performance scaling: dynamic buffers, rayon scheduler, broadphase caching + large-body bypass, island solver, CCD lazy construction, per-axis AABB, bindless materials. Interest culling + bandwidth budgeting in place.
- ~~Is the editor usable for content creation?~~ **YES** — transform gizmos, entity creation, scene save/load, undo/redo, terrain brushes, content browser
- ~~Can someone outside of us use this engine?~~ **PARTIALLY** — euca-math and euca-ecs published to crates.io; 24 per-crate READMEs; full engine packaging still TODO

## V2 Goals

**Theme: From engine to game platform.**

### Milestone 1: "Playable Game" ✅
Build the simplest possible competitive multiplayer game that both humans and AI agents can play.

**Status:** MOBA demo built entirely from CLI commands — heroes, minions, towers, waves, combat, economy, abilities, respawn, scoring. All systems work together. `scripts/moba.sh` runs the full game.

**What was built:**
- ✅ Game logic system (health, damage, respawn, teams)
- ✅ Projectile system (spawn, move, collision → damage, configurable radius)
- ✅ Game state (match phases, score, win condition)
- ✅ HUD overlay (health bars, score, gold, level)
- ✅ Camera (top-down, follow, preset views)
- ✅ Audio (spatial sound, bus mixing, reverb)

### Milestone 2: "AI Agent Plays" — IN PROGRESS
Connect an AI agent to the game server as a player. Prove the core differentiator.

**Status:** API surface complete (observe, spawn, step, all game commands). `agent_client.rs` example exists. End-to-end proof with Claude Code / OpenClaw pending.

**What exists:**
- ✅ Agent observes game state via HTTP API (positions, health, projectiles, game state)
- ✅ Agent sends actions (move, shoot, use ability, damage) via HTTP API
- ✅ Agent plays through CLI/HTTP — same game, different interface
- Benchmark: AI agent vs human — **not yet tested**

### Milestone 3: "Others Can Use It" — MOSTLY DONE
Make the engine usable by people who aren't us.

**What exists:**
- ✅ Clean public API with doc comments across all 24 crates
- Getting started tutorial — TODO
- ✅ Published euca-ecs and euca-math to crates.io
- ✅ Example game (MOBA demo) + 11 runnable examples
- Contribution guidelines — TODO

### Milestone 4: "Production Quality" ✅
Polish for real game development.

**What was built:**
- ✅ Texture support (albedo, normal, metallic-roughness, AO, emissive from glTF)
- ✅ Shadow mapping (cascaded shadow maps)
- ✅ Audio (kira: spatial, buses, reverb, occlusion, priority)
- ✅ Scene save/load (JSON, auto-save, hot reload)
- ✅ Editor (transform gizmos, undo/redo, content browser, terrain brushes, multi-select, copy/paste)
- ✅ Performance profiling (frame profiler, per-system timings, criterion benchmarks)
- Mobile deployment (Android/iOS) — TODO

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

### Phase C: Publish to crates.io ✅
Get the community using our building blocks.

1. ✅ **euca-math v0.1.0** — published to crates.io, standalone SIMD-ready math crate (zero deps)
2. ✅ **euca-ecs v0.1.0** — published to crates.io, standalone archetype ECS with change detection + parallel iteration
3. ✅ **Clean public API** — doc comments added across all 24 crates (v0.8.0)
4. ✅ **README per crate** — all 24 crates have README.md with description, features, usage examples (v0.7.0)
5. ✅ **Changelog** — CHANGELOG.md with semantic versioning, updated every release
6. **CI for crates** — publish workflow on tag (TODO)

### Phase D: Game-Ready Features ✅
Fill gaps needed before any real game can be built.

1. ✅ **Audio** — euca-audio: spatial sound via kira, bus mixing (Master/Music/SFX/Voice/UI), reverb zones, occlusion, priority
2. ✅ **Animation** — euca-animation: skeletal animation from glTF, state machines, blend spaces, IK (two-bone + FABRIK), root motion, montages
3. ✅ **Particle effects** — euca-particle: CPU particle system with emission shapes, color interpolation, billboard render. GPU compute particles in euca-render.
4. **AI agent plays** — API surface complete, end-to-end proof pending
5. ✅ **Client prediction** — euca-net: ClientPrediction with entity reconciliation, smooth correction, configurable smoothing factor
6. ✅ **Interest management** — euca-net: relevance-based entity filtering with GlobalTransform position lookups, bandwidth budgeting

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
A → B → C → D → E → F → G (all complete)

### Success criteria
- ✅ Phase A done: engine renders a scene that looks professional (textures, shadows, procedural sky, HDR post-processing)
- ✅ Phase B done: can create a simple level entirely in the editor (spawn objects, arrange with gizmos, save/load, undo/redo)
- ✅ Phase C done: `cargo add euca-ecs euca-math` works; READMEs, doc comments, and CHANGELOG complete across all 24 crates
- ✅ Phase D done: audio (kira), animation (state machines + IK), particles (CPU + GPU), client prediction, interest management all implemented
- ✅ Phase E done: all 11 CRITICAL issues from UE5 comparison resolved
- ✅ Phase F done: agent-native interface (SharedWorld unification, rich CLI, screenshot, play/pause, nit auth, SKILL.md)
- ✅ Phase G done: game logic layer (euca-gameplay: health, damage, teams, triggers, projectiles, AI, rules, game state, HUD, data tables — 95 tests)

---

## Phase F: Agent-Native Interface (COMPLETE)

**Goal:** Make AI agents (Claude Code, OpenClaw) first-class users of the engine.

**Decision:** CLI as primary interface (not MCP). Same pattern as `git`, `gh`, `cargo`.

**What was built:**
1. ✅ **SharedWorld unification** — Editor and HTTP server share the same ECS world via `Arc<RwLock<WorldPool>>`
2. ✅ **Rich CLI** — `euca` with 12 commands: status, observe, spawn, modify, despawn, step, play, pause, screenshot, reset, schema, auth
3. ✅ **Screenshot capture** — `euca screenshot` renders scene to offscreen texture, encodes PNG, returns path. Agent reads the image to verify visual state.
4. ✅ **Play/pause control** — `EngineControl` resource with `Arc<AtomicBool>`, shared between editor toolbar and HTTP handler
5. ✅ **nit authentication** — Ed25519 signature verification, session tokens, `euca auth login`
6. ✅ **SKILL.md** — Full CLI reference for agent consumption

---

## Phase G: Game Logic Layer (COMPLETE)

**Goal:** Enable agents to build complete games, not just 3D scenes.

**Design principle:** Library of composable ECS primitives, not a framework. Rules ARE entities.

**What was built:**

New crate `euca-gameplay` (39 tests):
1. ✅ **Health + Damage** — Health component, DamageEvent, DeathEvent, apply_damage_system, death_check_system
2. ✅ **Teams + Respawn** — Team component, SpawnPoint, RespawnTimer, respawn_system
3. ✅ **Projectiles** — Projectile component, collision detection, projectile_system
4. ✅ **Trigger Zones** — TriggerZone with Damage/Heal/Teleport actions, trigger_system
5. ✅ **AI Behavior** — AiGoal (Idle/Patrol/Chase/Flee), ai_system sets Velocity
6. ✅ **Game State** — GameState resource, MatchConfig, phase transitions, score tracking
7. ✅ **Data Tables** — JSON-loaded game configuration
8. ✅ **Data-Driven Rules** — OnDeathRule, TimerRule, HealthBelowRule with GameAction execution. Agents define "when X, do Y" without code.
9. ✅ **HUD** — Text, bars, rectangles via egui, controlled by CLI
10. ✅ **Full CLI integration** — entity damage/heal, game create/state, trigger/projectile/ai/rule/ui commands
