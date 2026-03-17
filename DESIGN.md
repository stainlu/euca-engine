# Euca Engine — System Design Document

> Living document. Single source of truth for architecture, decisions, and status.

## Vision

An ECS-first, agent-native game engine in Rust. Three pillars:
1. **ECS architecture** — Archetype-based, custom-built, optimized for large-scale simulation
2. **Agent-native** — External AI agents access the engine via CLI tools + HTTP/WebSocket API
3. **Rust** — Ownership for safety, proc macros for reflection, zero-cost abstractions

## Architecture Overview

```
External AI Agents (Claude Code, RL agents, etc.)
        │ CLI / HTTP / WebSocket
        ▼
┌─ Agent Interface ──────────────────────────────────┐
│  observe / act / step / save / load / reset        │
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

## Crate Map

| Crate | Purpose | Status | Tests |
|-------|---------|--------|-------|
| `euca-ecs` | Custom ECS: Entity, Component, Archetype, World, Query (&T + &mut T), Resource, Event, Command, Schedule (parallel batching), Snapshot, Change Detection, par_for_each, SystemAccess | Done | 69 |
| `euca-math` | Custom SIMD-ready Vec2/3/4, Quat, Mat4, Transform, AABB (zero deps) | Done | 28 |
| `euca-reflect` | `#[derive(Reflect)]` proc macro for runtime type info, integrated into editor inspector | Done | 1 |
| `euca-scene` | Transform hierarchy, Parent/Children BFS propagation, dirty-flag optimization | Done | 5 |
| `euca-core` | App builder, Plugin trait, Time resource, winit event loop | Done | 1 |
| `euca-render` | wgpu PBR renderer: Cook-Torrance BRDF, textures, shadow mapping, procedural sky, GPU instancing, HDR post-processing, hardware survey | Done | 16 |
| `euca-physics` | Custom AABB/sphere/capsule collision, spatial hash broadphase, CCD, iterative solver, raycasting, gravity (zero deps) | Done | 23 |
| `euca-asset` | glTF 2.0 model loading (meshes + PBR materials) | Done | 1 |
| `euca-agent` | HTTP API server for external AI agents (axum + tokio), multi-world pool, entity ownership | Done | 3 |
| `euca-editor` | egui editor: 3D viewport, hierarchy, inspector, play/pause/stop, transform gizmos, undo/redo, scene save/load, entity creation, grid, keyboard shortcuts | Done | 11 |
| `euca-input` | InputState, ActionMap, InputSnapshot for humans + AI agents | Done | 4 |
| `euca-net` | Raw UDP networking: PacketHeader, GameServer, GameClient, protocol | Done | 11 |
| `euca-game` | Arena game: health, projectiles, shooting, elimination | Done | 4 |
| `euca-cli` | CLI tool for AI agents (`euca observe`, `euca step`, etc.) | Done | 0 |

## Dependency DAG

```
euca-reflect-derive (proc-macro, no deps)
       │
euca-reflect (re-exports derive)
       │
euca-math (serde — zero external math deps)
       │
euca-ecs (euca-reflect, euca-math, serde)
       │
       ├── euca-scene (euca-ecs, euca-math)
       │
       ├── euca-core (euca-ecs, euca-math, winit)
       │
       ├── euca-render (euca-ecs, euca-math, euca-scene, wgpu, winit)
       │
       ├── euca-agent (euca-ecs, euca-scene, tokio, axum, serde)
       │
       └── euca-editor (euca-ecs, euca-render, euca-scene, euca-agent, egui)
```

## Key Design Decisions

### 1. ECS over OOP
**Decision**: Archetype-based ECS (like Bevy/flecs), not Actor-Component-inheritance (like Unreal).
**Why**: Cache-friendly iteration for large-scale sim, natural Rust ownership, trivial serialization for AI agent observation, no GC needed.
**Trade-off**: Hierarchical relationships (scene graph) are less natural — handled via Parent/Children components.

### 2. External AI agents over internal AI systems
**Decision**: No built-in BehaviorTree/NavMesh/AI Perception. Instead, the engine exposes API endpoints and CLI tools.
**Why**: AI agents like Claude Code interact via terminals and HTTP. The engine should be a tool they use, not a platform with built-in AI. This keeps the engine lean and lets agent intelligence evolve independently.
**Trade-off**: No out-of-the-box AI for traditional game dev. Users who want NPC AI must bring their own.

### 3. Custom ECS over Bevy ECS
**Decision**: Build ECS from scratch, study Bevy/flecs/EnTT for reference.
**Why**: Full control over memory layout, query system, and archetype storage. Can optimize for large-scale sim + AI agent access patterns. No coupling to Bevy's runtime.
**Trade-off**: More implementation effort. Must validate correctness and performance independently.

### 4. Forward+ over Deferred Shading
**Decision**: Start with Forward+ (clustered forward shading).
**Why**: Simpler to implement, handles transparency naturally, MSAA works out of the box. Good enough for pragmatic high-quality rendering.
**Trade-off**: Less efficient with many lights (>100). Can add deferred path later.

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

### 4-Pass Architecture
```
Pass 1 — Shadow Map
  Directional light depth pass → 2048×2048 depth texture
  Depth bias to reduce shadow acne
  Used by PBR pass for shadow sampling (3×3 PCF soft shadows)

Pass 2 — Procedural Sky
  Full-screen quad, no depth write
  Gradient horizon→zenith, sun disk + glow, atmospheric scattering

Pass 3 — PBR Forward (HDR)
  Renders to offscreen Rgba16Float target (HDR color space)
  Cook-Torrance BRDF (GGX distribution, Smith geometry, Fresnel-Schlick)
  Texture sampling: albedo maps, UV coordinates, procedural textures (checkerboard)
  Shadow sampling from Pass 1 depth map
  GPU instancing via storage buffers — batched draws by mesh+material (up to 16K instances)

Pass 4 — Post-Processing
  Reads HDR target from Pass 3
  13-tap Gaussian bloom (bright extraction → blur → composite)
  ACES filmic tone mapping (HDR → LDR)
  Vignette
  Outputs to swapchain (final present)
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
| 2026-03-18 | — | HIGH #21: PointLight + SpotLight component types, GpuPointLight/GpuSpotLight structs, SceneUniforms extended |
