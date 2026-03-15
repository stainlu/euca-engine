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
| `euca-math` | Vec2/3/4, Quat, Mat4, Transform, AABB (wraps glam) | Done | 23 |
| `euca-reflect` | `#[derive(Reflect)]` proc macro for runtime type info | Done | 1 |
| `euca-ecs` | Custom ECS: Entity, Component, Archetype, World, Query, Resource, Event, Command, System, Schedule | Done | 43 |
| `euca-scene` | Transform hierarchy, Parent/Children, spatial index | In Progress | — |
| `euca-core` | App builder, Plugin trait, Time resource, winit event loop | In Progress | — |
| `euca-render` | wgpu renderer, Camera, Mesh, basic Forward+ pipeline | In Progress | — |
| `euca-agent` | HTTP/WebSocket/CLI interface for external AI agents | Planned | — |
| `euca-editor` | egui-based visual editor (scene viewport, inspector) | Planned | — |
| `euca-physics` | Rapier integration as ECS systems | Planned | — |
| `euca-audio` | Spatial audio via cpal | Planned | — |
| `euca-asset` | Async asset loading, hot-reload, format conversion | Planned | — |
| `euca-net` | Multiplayer state replication via QUIC (quinn) | Planned | — |
| `euca-cli` | CLI tool for AI agents (`euca observe`, `euca act`, etc.) | Planned | — |

## Dependency DAG

```
euca-reflect-derive (proc-macro, no deps)
       │
euca-reflect (re-exports derive)
       │
euca-math (glam, serde)
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

## Rendering Pipeline (Phase 2)

### Current: Basic vertex-colored meshes
```
CPU: Build vertex/index buffers → Upload to GPU → Record draw calls
GPU: Vertex shader (MVP transform) → Fragment shader (vertex color) → Present
```

### Target: Forward+ PBR
```
1. Depth prepass
2. Light culling (compute: assign lights to screen-space clusters)
3. Forward pass (PBR: albedo, metallic, roughness, normal, AO)
4. Post-processing (bloom, tone mapping, FXAA)
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
| 2026-03-15 | 1 | euca-ecs: Entity/Component/Archetype/World/Query (24 tests) |
| 2026-03-15 | 1 | euca-ecs: Resource/Event/Command/System/Schedule (19 tests) |
| 2026-03-15 | 1 | euca-reflect: Reflect trait + derive macro |
| 2026-03-15 | 2 | Started: euca-scene, euca-core, euca-render |
