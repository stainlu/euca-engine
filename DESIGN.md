# Euca Engine — System Design Document

> Living document. Single source of truth for architecture, decisions, and status.

## Vision

An ECS-first, agent-native game engine in Rust. Three pillars:
1. **ECS architecture** — Archetype-based, custom-built, optimized for large-scale simulation
2. **Agent-native** — AI agents control the engine via the `euca` CLI (primary) backed by HTTP REST API. Agents authenticate with [nit](https://github.com/newtype-ai/nit) Ed25519 identity. See SKILL.md for the full agent interface.
3. **Rust** — Ownership for safety, proc macros for reflection, zero-cost abstractions

## Architecture Overview

```
External AI Agents (Claude Code, OpenClaw, etc.)
        │ euca CLI (primary) / HTTP REST (backend)
        ▼
┌─ Agent Interface ──────────────────────────────────┐
│  entity / sim / camera / game / trigger / ai /     │
│  rule / projectile / ui / screenshot / scene       │
└────────────────────────┬───────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────┐
│                    Engine Core                      │
│                                                     │
│  ┌─── ECS Runtime ──────────────────────────────┐  │
│  │  World → Archetypes → Columns (SoA)          │  │
│  │  Entity (gen indices) · Query · Schedule      │  │
│  │  Resource · Event · Command                   │  │
│  └──────────────────────────────────────────────┘  │
│                         │                           │
│  ┌──────┬───────┬───────┼───────┬──────┬────────┐  │
│  │Scene │Render │Physics│ Audio │ Net  │ Input  │  │
│  └──────┴───────┴───────┴───────┴──────┴────────┘  │
└─────────────────────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │ Editor  │ (egui, also an agent client)
                    └─────────┘
```

## Crate Map (24 crates, 850+ tests)

| Crate | Purpose | Tests |
|-------|---------|-------|
| `euca-ecs` | Archetype ECS: Entity, World, Query, Schedule, Events, change detection, Changed<T> filter, query caching, ParallelSchedule | 95 |
| `euca-math` | SIMD-accelerated (SSE2/NEON) Vec2/3/4, Quat, Mat4, Transform, AABB | 39 |
| `euca-reflect` | Runtime reflection: field access, TypeRegistry, JSON serialization, `#[derive(Reflect)]` | 6 |
| `euca-scene` | Transform hierarchy, prefabs, spatial index, world streaming, level file format | 28 |
| `euca-core` | App lifecycle, Plugin trait, Time resource, frame Profiler | 9 |
| `euca-render` | Forward+ PBR, cascaded shadows, FXAA, SSAO, SSR, volumetric fog, LOD, HLOD, HZB occlusion, GPU-driven, clustered lights (256+), foliage, decals, compute, Metal hints, SmartBuffer | 171 |
| `euca-physics` | Collision layers/masks, mass, character controller, vehicle physics, CCD, spatial hash, scene queries, joints | 53 |
| `euca-asset` | glTF loading, skeletal animation, async AssetStore, hot-reload | 11 |
| `euca-agent` | HTTP API (axum), 75+ endpoints, nit auth, HUD canvas, level loading | — |
| `euca-editor` | egui: viewport, hierarchy, inspector, play/pause/stop/reset, gizmos, undo/redo, level loading | 13 |
| `euca-input` | InputState, ActionMap, gamepad, input contexts, MOBA keybindings | 8 |
| `euca-net` | UDP transport, reliable layer, property replication, delta compression, RPCs, interest culling | 39 |
| `euca-gameplay` | Health, combat (role-aware targeting, SpatialIndex), economy, abilities, rules, player control, MOBA camera, corpse cleanup | 95 |
| `euca-audio` | Spatial audio (kira): bus mixing, reverb zones, occlusion, priority | 19 |
| `euca-particle` | CPU particle emitters, billboard render data, texture atlas, blend modes | 14 |
| `euca-nav` | Grid navmesh, A* pathfinding, steering behaviors | 10 |
| `euca-animation` | Blending, state machines, blend spaces, root motion, events, montages, IK (two-bone + FABRIK) | 54 |
| `euca-ai` | Behavior trees, blackboard, decorators, composites, action/condition nodes | 23 |
| `euca-ui` | Runtime UI: anchored layout, flex, widgets, input routing, world-space UI | 27 |
| `euca-terrain` | Heightmap terrain, chunk LOD, 4-layer splatting, physics colliders, brush editing | 30 |
| `euca-script` | Lua scripting (mlua): hot reload, sandboxing, ECS bridge, event handlers | 25 |
| `euca-game` | Standalone game runner | 4 |
| `euca-cli` | CLI tool: 30 command groups, level load/save, `euca discover --json` | 0 |

## Dependency DAG

```
euca-reflect-derive (proc-macro, no deps)
       │
euca-reflect (re-exports derive, TypeRegistry, JSON serialization)
       │
euca-math (serde, SIMD SSE2/NEON — zero external math deps)
       │
euca-ecs (euca-reflect, euca-math, serde)
       │
       ├── euca-scene (euca-ecs, euca-math)
       ├── euca-core (euca-ecs, euca-math, winit)
       ├── euca-render (euca-ecs, euca-math, euca-scene, wgpu — SSAO, LOD, compute, materials)
       ├── euca-physics (euca-ecs, euca-math — collision layers, mass, scene queries)
       ├── euca-animation (euca-ecs, euca-math, euca-scene — blending, state machines)
       ├── euca-ai (euca-ecs, euca-math — behavior trees, blackboard)
       ├── euca-terrain (euca-ecs, euca-math, euca-physics — heightmap, LOD, splatting)
       ├── euca-ui (euca-ecs, euca-math, euca-scene, euca-input — layout, widgets)
       ├── euca-script (euca-ecs, euca-math, mlua — Lua scripting, hot reload)
       ├── euca-agent (euca-ecs, euca-scene, tokio, axum, serde)
       └── euca-editor (euca-ecs, euca-render, euca-scene, euca-agent, egui)
```

## Key Design Decisions

### 1. ECS over OOP
**Decision**: Archetype-based ECS (like Bevy/flecs), not Actor-Component-inheritance (like Unreal).
**Why**: Cache-friendly iteration for large-scale sim, natural Rust ownership, trivial serialization for AI agent observation, no GC needed.
**Trade-off**: Hierarchical relationships (scene graph) are less natural — handled via Parent/Children components.

### 2. Dual AI: Agent-native + built-in NPC AI
**Decision**: External AI agents control the engine via CLI/HTTP. Built-in behavior trees (euca-ai) and navmesh (euca-nav) provide NPC AI at engine speed.
**Why**: External agents (Claude Code) need HTTP/CLI for observation and control. But NPCs need sub-millisecond AI at 60 FPS — HTTP round-trips are too slow. Both coexist: agents design the game, BTs run the NPCs.
**Trade-off**: Two AI paradigms to maintain.

### 3. Custom ECS over Bevy ECS
**Decision**: Build ECS from scratch, study Bevy/flecs/EnTT for reference.
**Why**: Full control over memory layout, query system, and archetype storage. Can optimize for large-scale sim + AI agent access patterns. No coupling to Bevy's runtime.
**Trade-off**: More implementation effort. Must validate correctness and performance independently.

### 4. Enhanced Forward+ as primary rendering path
**Decision**: Forward+ with clustered light culling (256+ lights) as primary. Deferred available as opt-in.
**Why**: Apple Silicon TBDR does deferred in hardware — our own G-buffer adds overhead. Forward handles transparency and MSAA natively. Clustered culling makes forward competitive with deferred for light count.
**Trade-off**: Visibility buffer (Nanite-style) is the future but requires mesh shaders wgpu doesn't support yet.

### 5. wgpu over raw Vulkan/Metal
**Decision**: Use wgpu for GPU abstraction.
**Why**: Cross-platform (Vulkan, Metal, D3D12, WebGPU), safe Rust API, well-maintained. Avoids writing platform-specific GPU code.
**Trade-off**: Can't access bleeding-edge GPU features (mesh shaders, work graphs) until wgpu adds them.

### 6. egui for Editor UI
**Decision**: egui (immediate-mode) rather than a retained-mode UI toolkit.
**Why**: Native Rust, integrates with wgpu, minimal boilerplate. Editor is a client of the ECS world via the same API as AI agents.
**Trade-off**: Less polished look than Qt/GTK. Limited layout capabilities compared to web-based editors.

### 7. Desktop-first, web-ready
**Decision**: Target desktop + mobile natively. Design abstractions for future web (WASM) support.
**Why**: WASM's single-threaded default and 4GB memory cap are deal-breakers for large-scale simulation. But wgpu already supports WebGPU, so the door stays open.
**Trade-off**: No browser deployment initially.

## ECS Internals

### Entity: Generational Index
```
Entity { index: u32, generation: u32 }
```
- Index is a reusable slot in a dense array
- Generation increments on slot reuse → stale handles fail validation
- Free list tracks available slots (LIFO for cache warmth)

### Archetype Storage (SoA)
- Each unique set of component types defines an archetype
- Components stored as contiguous columns (one `Vec<u8>` per component type per archetype)
- Adding/removing components moves entities between archetypes
- Archetype lookup: `HashMap<Vec<ComponentId>, ArchetypeId>`

### Query
- Type-safe: `Query<(&Position, &Velocity), Without<Static>>`
- Iterates only matching archetypes (skips non-matching)
- Filters: `With<T>`, `Without<T>`
- Can fetch `Entity` alongside components

### Schedule
- Sequential system execution (deterministic ordering)
- Each system is `Box<dyn FnMut(&mut World)>`
- Events swapped each tick (double-buffered, persist 2 frames)
- Commands: deferred spawn/despawn/insert/remove, applied between systems

## Rendering Pipeline

### Enhanced Forward+ Architecture
```
Pass 1 — Shadow Maps (3 cascades, 2048×2048 depth array)
Pass 2 — Clustered Light Assignment (compute, 16×9×24 clusters, 256 lights)
Pass 3 — Sky + PBR Forward (HDR, 4× MSAA, Cook-Torrance BRDF)
         PBR textures: albedo, normal, metallic/roughness, AO, emissive
         Alpha blend/cutout transparency (back-to-front)
         GPU instancing (16K instances), HZB occlusion culling, LOD selection
Pass 4 — Post-Processing Stack (modular):
         SSAO (GTAO, needs depth prepass), FXAA (default on),
         Bloom + ACES tonemapping, color grading, volumetric fog, SSR
         Quality presets: Low/Medium/High/Ultra

Additional subsystems: foliage instancing, decals, HLOD, GPU-driven (draw indirect),
deferred G-buffer (opt-in), depth+normal prepass, compute shaders

Apple Silicon: Metal TBDR hints, unified memory SmartBuffer, 32-thread compute, NEON SIMD
```

## Agent Interface Protocol

### Observation
```json
POST /observe
{ "components": ["Position", "Health"], "filter": { "within_radius": 100 } }
→ { "tick": 1234, "entities": [{ "id": 42, "Position": [1,2,3], "Health": 100 }] }
```

### Action
```json
POST /act
{ "entity": 42, "action": "move", "params": { "direction": [1,0] } }
→ { "ok": true }
```

### Simulation Control
```
POST /step   { "ticks": 100 }     → advance simulation
POST /save   { "path": "..." }    → snapshot world state
POST /load   { "path": "..." }    → restore world state
POST /reset                        → reset to initial state
```

## Changelog

| Date | Phase | What |
|------|-------|------|
| 2026-03-15 | 1 | euca-math: Vec/Quat/Mat4/Transform/AABB (23 tests) |
| 2026-03-15 | 1 | euca-ecs: Entity/Component/Archetype/World/Query/Resource/Event/Command/System/Schedule (43 tests) |
| 2026-03-15 | 1 | euca-reflect: Reflect trait + derive macro |
| 2026-03-15 | 2 | euca-scene: Transform hierarchy + propagation (3 tests) |
| 2026-03-15 | 2 | euca-core: App builder, Plugin, Time, winit event loop |
| 2026-03-15 | 2 | euca-render: wgpu Forward+ with vertex colors, then PBR upgrade |
| 2026-03-15 | 3 | euca-agent: HTTP API server (axum), headless mode |
| 2026-03-15 | 3 | euca-cli: CLI tool for AI agents |
| 2026-03-15 | 4 | euca-physics: Rapier3D integration (3 tests) |
| 2026-03-15 | 4 | euca-render: PBR Cook-Torrance BRDF, materials, lights |
| 2026-03-15 | 4 | euca-asset: glTF 2.0 model loading |
| 2026-03-15 | 5 | euca-editor: egui panels (hierarchy, inspector, play/pause) |
| 2026-03-15 | CI | GitHub Actions: check, test, clippy, fmt |
| 2026-03-15 | CI | Added 9 new tests, fixed all clippy warnings, README.md |
| 2026-03-15 | 5 | Editor: 3D viewport with PBR scene + egui overlay, Stop/Reset button |
| 2026-03-15 | — | Open-sourced under MIT license |
| 2026-03-15 | — | REVIEW.md: first-principles assessment, roadmap |
| 2026-03-15 | 7 | Fix: physics body leak on despawn, transform BFS propagation |
| 2026-03-15 | 7 | ECS: tick-based change detection, World::changed_entities, World::par_for_each |
| 2026-03-15 | 7 | Schedule: system stages (add_stage, add_system_to_stage) |
| 2026-03-15 | 7 | euca-input: InputState, ActionMap, InputSnapshot for networking |
| 2026-03-15 | 7 | euca-net: GameServer, GameClient, protocol (bincode serialization) |
| 2026-03-15 | 7 | ECS: WorldSnapshot (bincode + JSON serialization) |
| 2026-03-15 | 7 | Benchmarks: criterion (spawn, query, get, despawn, par_for_each, tick) |
| 2026-03-16 | 8 | GHOSTTY REFACTOR: removed glam → custom SIMD-ready math from scratch |
| 2026-03-16 | 8 | GHOSTTY REFACTOR: removed rapier3d/nalgebra → custom AABB/sphere/raycast physics |
| 2026-03-16 | 8 | GHOSTTY REFACTOR: removed quinn/rustls/tokio(net) → raw UDP with PacketHeader |
| 2026-03-16 | A | Rendering Quality: texture support (albedo maps, UV sampling, procedural textures), shadow mapping (2048px, 3×3 PCF, depth bias), procedural sky (gradient, sun disk, atmospheric scattering), GPU instancing (storage buffers, 16K instances), HDR post-processing (Rgba16Float, 13-tap bloom, ACES tone mapping, vignette) |
| 2026-03-16 | B | Editor Maturity: grid overlay, keyboard shortcuts (Delete/F/Ctrl+Z/Ctrl+Y), entity creation (+Empty/Cube/Sphere), scene save/load (JSON SceneFile), transform gizmos (3 axis handles, click+drag translate), undo/redo (stack-based UndoHistory, typed UndoAction, drag debouncing) |
| 2026-03-16 | C | Published to crates.io: euca-math v0.1.0, euca-ecs v0.1.0 — full doc comments, metadata (description, keywords, categories, license, repository) |
| 2026-03-17 | — | Hybrid GPU strategy: HardwareSurvey at startup (enumerate all adapters, vendor detection, single Instance reuse), RenderBackend enum, metal-native feature flag, re-export wgpu from euca-render |
| 2026-03-17 | — | Full UE5 comparison review: 79 issues identified (11 CRITICAL, 22 HIGH, 28 MEDIUM, 8 LOW). Phase E roadmap added. |
| 2026-03-17 | E | CRITICAL #1: Mutable queries (`Query<&mut T>`), tuple expansion to 8, ComponentAccess tracking, aliasing validation at Query::new() |
| 2026-03-17 | E | CRITICAL #2: SystemAccess enum, validate_no_conflicts(), System::accesses(), AccessSystem wrapper, UnsafeWorldCell, Res/ResMut wrappers, IntoSystem<Marker> |
| 2026-03-17 | E | CRITICAL #3: Parallel system scheduling — greedy batch algorithm, std::thread::scope parallel execution, SystemJob Send wrapper |
| 2026-03-17 | E | CRITICAL #11: Transform dirty flags — PropagationState resource, tick-based change detection, skip unchanged subtrees O(N)→O(moved) |
| 2026-03-17 | — | Physics fixes: restitution multiply, friction geometric mean, per-entity gravity override, pre-allocated corrections Vec |
| 2026-03-17 | — | Editor: improved showcase scene (pedestal, pillars, material showcase, warm lighting) |
| 2026-03-17 | E | CRITICAL #7: Capsule collider (capsule-capsule, capsule-sphere, capsule-AABB, raycast) |
| 2026-03-17 | E | CRITICAL #5: Spatial hash broadphase (replaces O(n²), cell size 4.0, falls back for <20 bodies) |
| 2026-03-17 | E | CRITICAL #6: Continuous collision detection (sweep-test fast bodies against statics) |
| 2026-03-17 | E | CRITICAL #8: Iterative constraint solver (4-iteration position-based, stable stacking) |
| 2026-03-17 | E | CRITICAL #9: Multi-world pool (RwLock<WorldPool>, create_world(), per-world access) |
| 2026-03-17 | E | CRITICAL #10: Entity ownership (Owner component, agent_id on spawn/despawn/patch, permission checks) |
| 2026-03-17 | E | CRITICAL #4: Reflection integration — Reflect derived/implemented on 7 component types, generic reflect_component<T>() inspector, replaced hardcoded display |
| 2026-03-18 | — | HIGH #26: Tangent vectors in vertex format (32→44 bytes), computed for cube/sphere/plane/glTF |
| 2026-03-18 | — | HIGH #20: Normal map support — Material.normal_texture, TBN matrix in PBR shader, has_normal_map flag |
| 2026-03-18 | — | HIGH #21: PointLight + SpotLight types, PBR shader light loop (4 point + 2 spot), draw_with_lights() API |
| 2026-03-18 | — | HIGH #23: Automatic mipmap generation on texture upload (CPU box filter, full mip chain) |
| 2026-03-18 | — | HIGH #25: Cascaded shadow maps (3 cascades, 2D array texture, per-fragment cascade selection) |
| 2026-03-18 | — | HIGH #27: Angular velocity integration (axis-angle from angular vel → rotation) |
| 2026-03-18 | — | HIGH #29: Body sleeping/deactivation (Sleeping component, velocity threshold, wake on collision) |
| 2026-03-18 | — | HIGH #15: SAFETY documentation for archetype.rs unsafe blocks |
| 2026-03-18 | — | HIGH #16: System ordering (labels + after() deps, topological sort in Stage) |
| 2026-03-18 | — | HIGH #17: Lifecycle phases (startup/shutdown systems, App::tick(), App::shutdown()) |
| 2026-03-18 | — | HIGH #18: Non-blocking tick (App::tick() for external event loop integration) |
| 2026-03-18 | — | HIGH #31: Fixed timestep accumulation (PhysicsAccumulator, physics_step_with_dt, max_substeps) |
| 2026-03-18 | — | HIGH #32: Packet retransmission (ReliableTransport, count-based retry, ack processing) |
| 2026-03-18 | — | Quality review: reverted broken query cache, fixed solver body collection (collect once), fixed transport retry logic |
| 2026-03-18 | — | HIGH #13: Query caching with archetype generation tracking (cache invalidation on archetype creation) |
| 2026-03-18 | — | HIGH #22: Frustum culling (Frustum struct, Gribb/Hartmann plane extraction, AABB-frustum test) |
| 2026-03-18 | — | HIGH #12: Change detection (Changed<T> marker, per-entity get_change_tick, changed_entities iterator) |
| 2026-03-18 | — | Quality review: reverted broken query cache, fixed solver body collection, fixed transport retry logic |
| 2026-03-18 | E | HIGH #14: Sparse set storage (SparseSet, ComponentInfo.sparse flag, transparent routing in World) |
| 2026-03-18 | E | HIGH #28: Joint constraints (distance, ball-and-socket, revolute, integrated into constraint solver) |
| 2026-03-18 | E | HIGH #33: Client prediction (ClientPrediction, reconciliation, smooth corrections, input replay) |
| 2026-03-18 | E | HIGH #24: Compressed texture upload (upload_compressed with explicit wgpu format) |
| 2026-03-18 | E | HIGH #19: Deferred rendering infrastructure (GBuffer, RenderPath enum, coexists with forward) |
