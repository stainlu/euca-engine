# Euca Engine

### An ECS-first, agent-native game engine in Rust

24 crates · 340+ tests · MIT licensed · Rust 1.89 (Edition 2024)

---

# The Problem

Traditional engines (Unreal, Unity) were designed in the OOP era:

- **God objects** — UWorld owns everything: actors, components, physics, rendering, GC
- **Deep inheritance** — AActor → APawn → ACharacter → AMyHero (rigid, hard to compose)
- **Circular dependencies** — 7+ circular module deps in UE5 core
- **200+ runtime modules** — monolithic, all-or-nothing
- **Not agent-friendly** — designed for humans clicking in an editor, not AI agents sending commands

What if we started from scratch with modern principles?

---

# Three Pillars

### 1. ECS-First
Data-oriented design. Composition over inheritance.
Entities are IDs. Components are data. Systems are logic.
No god objects. No inheritance hierarchies.

### 2. Agent-Native
AI agents are first-class citizens.
They observe, act, and build games via CLI and HTTP — no Rust needed.

### 3. Modular
24 independent crates with a strict acyclic dependency graph.
Pick what you need. No framework lock-in.

---

# Architecture

```
AI Agents (Claude Code, scripts, RL agents)
    │ CLI (euca) / HTTP REST (port 3917)
    ▼
┌─ Agent Interface ────────────────────────────┐
│  50+ endpoints: spawn, observe, combat,      │
│  rules, economy, abilities, diagnose         │
├──────────────────────────────────────────────┤
│  Gameplay: Health, Teams, Combat, Economy,   │
│  Leveling, Abilities, Rules, Triggers, AI    │
├──────────────────────────────────────────────┤
│  Domain: Animation, AI, Terrain, UI, Script  │
├──────────────────────────────────────────────┤
│  Core: ECS, Render, Physics, Audio, Nav,     │
│  Scene, Particles, Assets, Input, Net        │
├──────────────────────────────────────────────┤
│  Foundation: Math (SIMD), Reflect (proc mac) │
└──────────────────────────────────────────────┘
    ▲
    │ same API
    Editor (egui: viewport, inspector, gizmos)
```

Strict acyclic DAG. No crate depends on a crate above it.

---

# Why Custom ECS

**Not Bevy. Not flecs. Built from scratch.**

| Feature | Why it matters |
|---------|---------------|
| Archetype SoA storage | Components in contiguous columns → cache-friendly iteration |
| Generational entity IDs | Reusable slots, stale handles fail automatically → no GC |
| Type-safe queries | `Query<(&Position, &mut Velocity), Without<Static>>` — compile-time safety |
| Change detection | Per-entity tick tracking → skip unchanged work |
| Parallel batching | Systems with non-conflicting accesses run in parallel |
| Command pattern | Deferred spawn/despawn → deterministic simulation |
| Sparse sets | Rare components stored separately → no archetype explosion |

**Why not Bevy?** Full control over memory layout. No coupling to Bevy's runtime.
Optimized for agent access patterns (observe entire world, step N ticks).

---

# Why Agent-Native

**Traditional workflow:**
Human writes code → engine compiles → game runs

**Euca workflow:**
AI agent observes world → sends commands → engine simulates

```bash
# An AI agent builds a MOBA — no Rust code
euca entity create --mesh cube --position 0,2,0 \
    --health 100 --team 1 --combat
euca rule create --when death --do-action "score source +1"
euca rule create --when timer --interval 30 \
    --do-action "spawn minion"
euca sim play
```

The engine is the runtime. Agents are the developers.
50+ CLI endpoints. Data-driven rules. Zero game code.

---

# Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Rendering | Forward+ via wgpu | Simpler, handles transparency, MSAA works. wgpu = cross-platform (Vulkan/Metal/D3D12/WebGPU) |
| Physics | Custom (no Rapier) | Zero heavy deps, spatial hash broadphase, CCD, constraint solver — predictable behavior |
| Math | Custom SIMD (SSE2/NEON) | Zero external deps for the critical path |
| Simulation | Command pattern | Systems don't mutate world directly → deterministic, reproducible |
| Events | Double-buffered | Events persist 2 frames → systems are decoupled, no ordering dependency |
| Editor | egui (immediate-mode) | Native Rust, uses same API as agents — editor is just another client |
| Platform | Desktop-first, web-ready | WASM's 4GB cap + single-thread = not ready for large sim. But wgpu keeps the door open |

---

# Zero-Dependency Foundation

The critical path has **zero external dependencies**:

```
euca-math     → custom Vec/Quat/Mat4/AABB, SIMD intrinsics
euca-ecs      → custom archetype storage, generational IDs
euca-physics  → custom collision, raycasting, solver
euca-net      → raw UDP, bincode protocol
```

**Why?** Game engines live or die on their hot path.
External deps mean external perf characteristics, breaking changes, and bloat.
We control the memory layout, the allocation patterns, the iteration order.

---

# What's Built

| System | Highlights |
|--------|-----------|
| **ECS** | Archetype storage, parallel scheduling, change detection, sparse sets (69 tests) |
| **Rendering** | PBR (Cook-Torrance), 3-cascade shadows, normal maps, HDR bloom, SSAO, GPU instancing, LOD |
| **Physics** | AABB/sphere/capsule, spatial hash, CCD, joints, constraint solver, sleeping (23 tests) |
| **Gameplay** | Health/damage, auto-combat, teams, economy, leveling, abilities (Q/W/E/R), data-driven rules (39 tests) |
| **Animation** | State machines, blend spaces, root motion, montages, IK (FABRIK + two-bone) (50 tests) |
| **AI** | Behavior trees, blackboard, patrol/chase/flee (23 tests) |
| **Editor** | 3D viewport, hierarchy, inspector, gizmos, undo/redo, scene save/load |
| **Scripting** | Lua (hot reload, sandboxing, ECS bridge) |
| **Audio** | Spatial sound (kira), bus mixing, reverb zones, occlusion |
| **Terrain** | Heightmap, chunk LOD, 4-layer splatting, brush editing (30 tests) |

---

# Unreal vs Euca

| | Unreal Engine 5 | Euca Engine |
|-|------------------|-------------|
| **Object model** | Actor-Component (OOP inheritance) | Archetype ECS (composition) |
| **Memory** | GC + reflection for lifetime | Generational IDs, no GC |
| **Modules** | 200+ runtime modules, circular deps | 24 crates, strict acyclic DAG |
| **World** | God object UWorld | Decomposed: World + Schedule + Resources |
| **Scripting** | Blueprint VM | Lua + data-driven rules |
| **AI integration** | Built-in (tightly coupled) | Agent-native (loosely coupled via CLI/HTTP) |
| **Build** | 40+ min full rebuild | `cargo build` in seconds |
| **Language** | C++ (1979) | Rust (2015) — ownership, no data races |

Not a replacement for Unreal. A different philosophy:
**What if the engine was designed for AI agents from day one?**

---

# What's Next

- **Parallel scheduling** — access-pattern analysis for automatic parallelism
- **Deferred rendering** — G-buffer path for 100+ light scenes
- **Real networking** — replication, interest management, lag compensation
- **Ship a game** — prove the architecture with a real multiplayer title

---

# Thank You

**Euca Engine** — ECS-first, agent-native, built in Rust

github.com/stainlu/euca-engine · eucaengine.com
