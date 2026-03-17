# Euca Engine — First Principles Review

> Written 2026-03-15. Brutally honest assessment of what we built, what we got wrong, what we took for granted, and what the real roadmap should be.

## Core Value (Restated)

**The most modern & efficient ECS engine for building multiplayer online games that is friendly to external AI agents.**

Three claims. Let's evaluate each honestly.

---

## Claim 1: "Most modern & efficient ECS"

### What we got right
- **Archetype-based storage (SoA)** — This is the proven approach (Bevy, flecs). Cache-friendly iteration, O(1) entity location lookup.
- **Generational entity IDs** — Industry standard. Prevents dangling references.
- **Type-safe queries** — Compile-time checked. No runtime type errors.
- **43 unit tests** — The ECS core is the most tested part of the engine. Solid.

### What we took for granted (and shouldn't have)

**1. The schedule is sequential.**
We claim "optimal for large-scale simulation" but run every system one after another on a single thread. A 64-core server wastes 63 cores. This directly contradicts "most efficient."

**Fix:** Implement access-pattern analysis. If System A reads `Position` and System B reads `Velocity`, they can run in parallel. Bevy does this. We need it.

**2. No change detection.**
Every system iterates every entity every frame, even if nothing changed. For a game with 10,000 entities where only 50 moved, we're wasting 99.5% of work.

**Fix:** Tick-based change tracking. Each component slot stores the tick it was last written. Queries filter `Changed<T>` by comparing against the system's last-run tick. This is how Bevy does it, and it's the right approach.

**3. No archetype graph.**
When you add/remove a component, we do a linear search through `archetype_index` (HashMap). Bevy and flecs cache "edges" between archetypes — if you've ever added `Velocity` to an entity with `[Position]`, the transition to `[Position, Velocity]` is cached and subsequent identical transitions are O(1).

**Fix:** Add archetype transition caching. One HashMap per archetype storing `ComponentId → ArchetypeId` edges.

**4. Query composition limited to 3 components.**
Games often query 4-6 components. Our tuple impls only go to 3. This is an artificial limit.

**Fix:** Macro-generate tuple impls up to 8 or 10. Or use a proc macro.

### Should we rebuild the ECS from scratch?
**No.** The foundation is correct. The archetype storage, entity allocation, and query system are all sound. We need to add features (parallelism, change detection, archetype graph), not replace the core.

---

## Claim 2: "Building multiplayer online games"

### What we have
**Nothing.** There is zero networking code. This is the single biggest gap between our core value and our implementation. We have a single-player engine claiming to be for multiplayer.

### First principles: What does a multiplayer game engine actually need?

Think about what happens when 100 players are in the same game world:

```
Server (headless, no GPU):
  1. Receive player inputs from network
  2. Validate inputs (anti-cheat)
  3. Apply inputs to ECS world
  4. Step simulation (physics, game logic)
  5. Compute state delta (what changed this tick)
  6. Send deltas to relevant clients (interest management)

Client (has GPU):
  1. Predict locally (don't wait for server)
  2. Receive server state
  3. Reconcile prediction vs server truth
  4. Render
```

**Implication: The server is the primary runtime, and it runs HEADLESS.**

This means our headless mode (`euca-agent --example headless_server`) is actually closer to the right architecture than our windowed examples. The renderer is a client concern, not a server concern.

**What we need to build:**
1. **Transport layer** — QUIC via `quinn` (reliable + unreliable channels, built-in encryption)
2. **State replication** — Component-level delta sync. Only send what changed (needs change detection from ECS).
3. **Interest management** — Don't send all entities to all clients. Only nearby/relevant ones.
4. **Client prediction** — Run simulation locally, reconcile when server state arrives
5. **Input protocol** — Timestamped inputs from clients to server
6. **Authority model** — Server owns truth. Clients are untrusted.

### Should we use an existing networking library?
**No existing Rust game networking library fits our ECS architecture well.** Libraries like `naia` are opinionated about their own ECS. We should build the networking layer ourselves, using `quinn` for transport and leveraging our ECS's change detection for efficient delta compression.

---

## Claim 3: "Friendly to external AI agents"

### What we got right
- **CLI tool** — AI agents like Claude Code interact via terminal commands. This is genuinely novel. No other engine does this.
- **HTTP API** — Standard REST interface. Any language can talk to it.
- **Headless mode** — Server runs without GPU. Essential for AI training at scale.

### What we took for granted

**1. The Mutex bottleneck.**
Our `SharedWorld` wraps everything in `Arc<Mutex<>>`. When an RL training loop calls `POST /step` 10,000 times per second, the Mutex serializes everything. An RL agent doing self-play with 100 parallel environments gets zero parallelism.

**First principles fix:** Each environment should be an independent `World` in its own thread. The agent interface should manage a pool of worlds, not share one.

```
Current:   1 World behind 1 Mutex, N agents compete for lock
Should be: N Worlds, each in own thread, agent picks which to interact with
```

**2. HTTP overhead for training.**
HTTP has ~100μs overhead per request. For RL training doing `step → observe → act → step` at 10kHz, HTTP adds 400μs per loop iteration — that's 40% overhead if a tick takes 1ms.

**First principles fix:** Add a shared-memory or Unix socket protocol for local agents. Keep HTTP for remote/debugging. Something like:

```
Agent ←→ Engine communication:
  - HTTP (remote, debugging, human-friendly): /observe, /step, /act
  - Shared memory (local RL training, maximum speed): mmap'd ring buffer
  - In-process (embedding): direct function call to World::step()
```

**3. No observation filtering.**
`POST /observe` returns ALL entities. An RL agent observing a 10,000-entity world gets a massive JSON blob, most of which is irrelevant. Real RL environments give agents a small, fixed-size observation.

**Fix:** Spatial observation queries, component filtering, tensor output format.

### Should we rebuild the agent interface?
**Partially.** Keep HTTP + CLI for dev/debugging. Add an in-process embedding API for RL training speed. The HTTP layer is correct, just insufficient for high-performance training.

---

## Dependencies: Keep vs Rebuild

| Dependency | Current Role | Keep? | Reasoning |
|-----------|-------------|-------|-----------|
| **glam** | Math (Vec3, Quat, Mat4) | **Keep but drop the wrapper** | Our Vec3/Quat/Mat4 types are `#[repr(transparent)]` wrappers around glam with Deref. This adds zero value and complicates the API. Just re-export glam types directly. |
| **wgpu** | GPU rendering | **Keep** | GPU abstraction is hard to get right. wgpu handles Vulkan/Metal/D3D12/WebGPU. Not worth rebuilding. |
| **rapier3d** | Physics | **Consider replacing** | Rapier brings `nalgebra` (adds 30+ seconds to compile time). Multiplayer online games often need simple collision (AABB, sphere-sphere, raycasting) not full rigid body simulation. A custom minimal physics system could be 10x faster to compile and more suitable for network games. |
| **egui** | Editor UI | **Keep for now** | Good enough for prototype. Long-term, a web-based editor (like Bevy's planned approach) would be more powerful. |
| **axum + tokio** | HTTP server | **Keep but add alternatives** | Axum is great for HTTP. But add in-process API for embedded use (RL training). |
| **gltf** | Asset loading | **Keep** | Standard format parser. Not worth rebuilding. |
| **serde** | Serialization | **Keep** | Industry standard. |
| **winit** | Windowing | **Keep** | Commodity. |

### The rapier3d question deserves more thought:

**Rapier pros:** Full rigid body sim, joint constraints, continuous collision detection, SIMD solver
**Rapier cons:** 30s compile time, nalgebra dependency, full GJK/EPA narrowphase (overkill for simple games)

**For multiplayer online games, what physics do you actually need?**
- AABB collision detection
- Sphere/capsule overlap tests
- Raycasting (for shooting, line of sight)
- Simple gravity + character controllers
- NOT: Stacking cubes, ragdolls, cloth simulation

A custom minimal physics system (AABB broadphase, capsule narrowphase, raycast) would compile in 2 seconds and cover 80% of multiplayer game needs. The other 20% (vehicle physics, destruction) can use Rapier as an optional plugin.

---

## Architectural Blind Spots

### 1. We're render-first when we should be simulation-first
Our examples all open windows and render. The primary runtime should be headless simulation. Rendering is a client-side visualization layer.

**Implication:** The engine entry point should be `World::new() + Schedule::run_loop()`, not `App::run_windowed()`.

### 2. No embeddability
You can't `use euca_ecs::World` in a Python script or import it into a Jupyter notebook. For AI research, this is critical. The engine should be usable as a library, not just as a standalone binary.

**Fix:** Publish `euca-ecs` and `euca-physics` as standalone crates on crates.io. They should work independently without windowing or rendering.

### 3. No scripting
Game logic is written in Rust and compiled. This means:
- Designers can't iterate without recompiling
- Modding is impossible
- AI agents can't define new behaviors at runtime

**Possible fixes:** Lua scripting (via `mlua`), WASM modules, or Rhai. Or accept that Rust + hot-reload (cargo-watch) is the workflow.

### 4. The reflect system is unused
We built `#[derive(Reflect)]` but nothing uses it. The editor hardcodes component inspection. Serialization doesn't use it. The agent API doesn't use it.

**Fix:** Either integrate Reflect into the editor and serialization, or delete it. Dead code is worse than no code.

### 5. No benchmarks
We claim "efficient" but have zero benchmarks. How many entities can we tick per second? How does our query performance compare to Bevy? Without numbers, "efficient" is just marketing.

**Fix:** Add criterion benchmarks for: entity spawn/despawn, query iteration, archetype migration, physics step, headless tick rate.

---

## Revised Roadmap (First Principles)

Reordered by what actually matters for the core value.

### Tier 1: Foundation (must-do, blocks everything else)
1. **Parallel system scheduling** — Without this, "efficient" is a lie
2. **Change detection** — Without this, networking delta sync is impossible
3. **Fix physics body leak** — Memory leak in long-running servers is fatal
4. **Benchmarks** — Can't optimize what you can't measure

### Tier 2: The Differentiator (what makes Euca unique)
5. **Networking / state replication** — This is the core value. Build it.
6. **Agent API: in-process embedding** — Let RL researchers call `world.step()` directly from Rust/Python
7. **Multi-world support** — Run 100 simulation instances in parallel for AI training
8. **World serialization** — Save/load for checkpointing, replay, RL episodes

### Tier 3: Make it usable
9. **Drop glam wrappers** — Simplify the math API, reduce abstraction tax
10. **Minimal custom physics** — Replace Rapier with AABB/capsule/raycast for 10x compile speed
11. **Input abstraction** — InputState resource, action mapping
12. **Texture support** — Albedo maps in the renderer

### Tier 4: Polish
13. **Shadow mapping** — Visual quality
14. **GPU instancing** — Rendering performance
15. **Editor improvements** — Editable transforms, entity creation
16. **Audio** — Spatial sound
17. **Plugin system** — Modular extension points

### What NOT to do (yet)
- Web/WASM port (premature)
- Custom GPU backend (wgpu is fine)
- Full retained-mode editor UI (egui is fine for now)
- Skeletal animation (not needed for initial multiplayer games)
- Visual scripting (Rust is the scripting language for now)

---

## The Real Question

The real question isn't "what features to add next." It's:

**What game are you going to make with this engine?**

An engine without a game is an academic exercise. The best game engines (Unreal, Unity, Godot) were all built to ship specific games. The engine's design should be driven by the game's needs.

If the answer is "a multiplayer online game where AI agents compete with humans" — then networking and agent API hardening are the only things that matter right now. Everything else (shadows, editor gizmos, skeletal animation) can wait until there's an actual game that needs them.

Build the game. Let the game tell you what the engine needs.

---

## Update: 2026-03-16

Since this review was written, significant progress was made across all fronts:

**Rendering (Phase A complete):** Textures, 2048px shadow mapping with PCF, procedural sky with sun glow, GPU instancing via storage buffers (16K instances), HDR pipeline with bloom + ACES tone mapping + vignette. The engine now renders professional-quality scenes.

**Editor (Phase B complete):** Transform gizmos (3-axis drag), undo/redo system (Ctrl+Z/Y), entity creation (+ Cube/Sphere/Empty), scene save/load (JSON), grid overlay, keyboard shortcuts (Delete, F focus). The editor is now usable for content creation.

**crates.io (Phase C complete):** `euca-math` v0.1.0 and `euca-ecs` v0.1.0 published with full doc comments. Anyone can `cargo add euca-math euca-ecs`.

**The critique above still stands.** Shadows and gizmos were built (contrary to the "can wait" advice), but the core point remains: the engine needs a game to drive its next evolution. Phase D should focus on making a real game playable by both humans and AI agents.

---

## Update: 2026-03-17 — Full UE5 Comparison Review

Strict review of all 16 crates against Unreal Engine 5. 79 issues found.

### CRITICAL (11 — blocks production)

| # | System | Issue |
|---|--------|-------|
| 1 | ECS | No mutable queries — `Query` only supports `&T`, not `&mut T` |
| 2 | ECS | No system parameter extraction — systems take raw `&mut World`, can't auto-parallelize |
| 3 | ECS | No parallel system scheduling — sequential loop wastes all cores but one |
| 4 | ECS | Reflection system exists but unused — can't generically serialize/replicate/inspect |
| 5 | Physics | O(n²) broadphase — 1000 bodies = 500K checks/frame |
| 6 | Physics | No continuous collision detection — fast objects tunnel through geometry |
| 7 | Physics | Only AABB + Sphere — no capsule, convex hull, triangle mesh |
| 8 | Physics | No constraint solver — push-out causes jitter with 3+ stacked bodies |
| 9 | Agent | Mutex bottleneck on SharedWorld — zero parallelism for RL training |
| 10 | Agent | No entity ownership — any client can despawn any entity |
| 11 | Scene | No dirty flags on transforms — recalculates ALL globals every frame |

### HIGH (22)

**ECS:** Entity-level change detection only (#12), no query caching (#13), no sparse sets (#14), 61 unsafe blocks without SAFETY comments (#15), no system ordering deps (#16), no lifecycle phases (#17), event loop blocks forever (#18)

**Rendering:** Forward-only (#19), no normal maps (#20), only 1 directional light (#21), no frustum culling/LOD (#22), no mip-maps (#23), no texture compression (#24), single 2048 shadow map (#25), no tangents in vertex format (#26)

**Physics:** Angular velocity unused (#27), no joints (#28), no sleeping (#29), friction model incorrect (#30), no timestep accumulation (#31)

**Networking:** No packet retransmission (#32), no client prediction (#33)

### MEDIUM (28)

ECS: no multi-world (#34), commands applied immediately (#35), events 2-frame only (#36), no plugin deps (#37), snapshot captures transforms only (#38), no determinism (#39), no resource lifecycle (#40)

Rendering: no AA (#41), no exposure control (#42), perspective only (#43), no color grading (#44), no skeletal animation (#45), no transparency sorting (#46), hardcoded post-FX (#47), GPU init panics (#48), hardcoded vsync (#49), MAX_INSTANCES unchecked (#50), shaders as strings (#51)

Physics: restitution average→multiply (#52), no triggers (#53), no collision layers (#54), hot loop allocations (#55)

Asset: glTF discards textures (#56), no caching (#57), no async load (#58)

Input: no gamepad (#59), no contexts (#60), no buffering (#61)

Agent: fake socket ports (#62), no spatial filtering (#63), no rate limiting (#64)

Editor: selection without generation (#65), translate-only gizmo (#66), hardcoded inspector (#67), 3 undo types (#68)

Infra: no profiling (#69), no benchmarks (#70), no input validation (#71)

### LOW (8)

No SIMD in math (#72), no swizzling (#73), magic slerp threshold (#74), no GPU debug markers (#75), no CI coverage (#76), single-platform CI (#77), no nightly testing (#78), no scripting (#79)

### What works well

- Archetype ECS fundamentals (correct SoA, generational IDs, type-safe queries)
- PBR rendering (Cook-Torrance BRDF, shadow PCF, HDR bloom, ACES)
- Hardware survey (GPU enumeration, vendor detection, diagnostics)
- Math library (correct, zero-dep)
- Agent HTTP API (novel approach)
- Code organization (16 crates, clear boundaries)
- 145 tests, CI green

### Verdict

~30% production-ready. Strong prototype with sound foundations. Next priorities: mutable queries, parallel scheduling, transform dirty flags, physics broadphase, agent mutex elimination.
