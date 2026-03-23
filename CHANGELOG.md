# Changelog

## v0.9.0 (2026-03-24)

### Hard Problems — Tuples, Stress Test, Multiplayer, WASM Foundation

- **ECS query tuples expanded to 12**: `Query<(&A, &B, ..., &L)>` now works with up to 12 component types. QueryFilter also expanded to 12.
- **1000-entity stress test**: New `examples/stress_test.rs` — spawns 1000 physics entities with collision, renders at 60fps, prints FPS every 60 frames.
- **Real multiplayer proof**: Rewrote `examples/client.rs` to use proper ECS world sync. Client spawns/updates/despawns entities from server state. Integrated ClientPrediction for the local player with smooth correction.
- **WASM foundation**: Platform abstraction layers for future web export:
  - `euca-core/time.rs`: `web-time` crate for WASM-compatible `Instant`
  - `euca-ecs/schedule.rs`: Sequential fallback for `#[cfg(target_arch = "wasm32")]`

## v0.8.2 (2026-03-23)

### Medium-Severity Bug Fixes

- **Scene**: Spatial index `cell_key` clamps coordinates to `i32::MIN/2..=i32::MAX/2` to prevent overflow hash collisions at extreme positions
- **Scene**: Streaming loader wrapped in `catch_unwind` so the `ChunkLoader` resource is always re-inserted even if a load callback panics
- **Render**: TAA neighborhood clamping now operates in YCoCg color space instead of linear RGB, reducing color-shift ghosting artifacts
- **Physics**: Capsule raycast uses hemisphere normal for all cylinder hits, eliminating normal discontinuity at the cylinder-hemisphere junction
- **Physics**: Vehicle suspension `prev_compression` correctly updated each frame (verified with test)
- **Animation**: State machine suppresses transition evaluation on the frame a crossfade completes, preventing double-transitions through any-state rules
- **Animation**: Root motion `extract_root_motion` logs a warning when the root bone index is out of bounds
- **ECS**: Despawning the last entity in an archetype increments `archetype_generation` to invalidate query caches

## v0.8.1 (2026-03-23)

### Deep Logic Fixes

Critical bug fixes found during full foundation audit:

- **ECS**: Entity generation overflow → saturating add prevents use-after-free
- **ECS**: Parallel schedule panics on dependency cycles instead of silent fallback
- **Math**: Slerp handles near-parallel quaternions without NaN
- **Math**: Matrix inverse returns identity for singular matrices
- **Math**: Transform inverse guards against degenerate (zero) scale
- **Render**: TAA stores jittered VP for correct temporal reprojection
- **Render**: Shadow bias scales with scene instead of fixed constant
- **Physics**: Friction impulse sign corrected (clamp instead of max)
- **Physics**: Raycast handles axis-aligned rays (zero direction components)
- **Gameplay**: Timer rules check-then-update instead of update-then-check
- **Net**: Prediction reconciliation uses nearest-tick tolerance (±2) instead of exact match

## v0.8.0 (2026-03-23)

### Consolidation — Architecture & Code Quality Review

- Cargo.toml metadata for all 24 crates (description, keywords, categories)
- Split euca-cli/main.rs (2807 lines) into focused command modules
- Split euca-audio/lib.rs (899 lines) into engine/source/reverb/systems modules
- Replaced println!/eprintln! with proper log:: macros
- Replaced 21 wildcard re-exports in euca-agent routes with explicit exports
- Reviewed and documented all clippy suppressions
- Added comprehensive doc comments to all public APIs across 24 crates
- Version bump to 0.8.0

## v0.7.0 (2026-03-23)

### Quality Release — READMEs, Robustness, Architecture Review

- **24 per-crate READMEs**: Every crate now has a README.md with description, features, usage example. Ready for crates.io discoverability.
- **Hot-path unwrap elimination**: Replaced dangerous `.expect()`/`.unwrap()` in production render, physics, and scheduler code with graceful fallbacks (`Option` returns, `log::warn` + skip, early return guards).
  - `compute.rs`: Frustum cull bind group → returns `Option<BindGroup>`
  - `occlusion.rs`: HZB bind group → returns `Option<BindGroup>`, dispatch → logs + returns on missing pipeline
  - `vehicle.rs`: Torque curve → early return on empty samples
  - `schedule.rs`: Thread join → `log::error!` instead of propagating panic
- **Architecture audit**: Verified no circular dependencies, no dead code beyond legitimate GPU resource ownership, clean layered dependency graph.

### Infrastructure
- 740+ tests, 0 failures
- Version bump to 0.7.0

## v0.6.0 (2026-03-23)

### Wire Everything — Close All Integration Gaps
Every "declared but not functional" feature is now wired into the render/gameplay loop.

- **glTF texture upload**: `apply_texture_handles()` bridges image indices to GPU TextureHandles. glTF viewer wires textures before material upload.
- **CPU particle rendering**: `collect_particle_render_data()` called in editor render loop, billboard geometry converted to Vertex meshes and drawn.
- **GPU particle integration**: `Renderer::add_gpu_particle_system()` API, compute emit/update + billboard draw in `render_to_view()`.
- **Client prediction in gameplay schedule**: `apply_prediction_system` added as "prediction_correction" stage after steering, before event flush.
- **Terrain brush editor UI**: `terrain_panel()` with Raise/Lower/Flatten/Smooth modes, radius/strength/target_height sliders. `EditorState` tracks brush state.
- **Decal rendering**: `DecalRenderer` initialized in Renderer, `set_decal_commands()` API, drawn after opaque geometry using unit-cube projection.

### Infrastructure
- 740+ tests, 0 failures
- Version bump to 0.6.0

## v0.5.0 (2026-03-23)

### Asset Pipeline
- glTF texture image extraction — all 10 image formats converted to RGBA8
- Texture indices per mesh (albedo, normal, metallic-roughness, AO, emissive)
- `euca asset info` now reports texture count and dimensions

### GPU Compute Particles
- GpuParticleSystem: compute emit/update + instanced billboard render
- PCG hash PRNG on GPU for particle randomization
- Configurable: 100K+ particles, cone emission, gravity, lifetime color fade
- Compute + render shaders (particle_compute.wgsl, particle_render.wgsl)

### Editor UX
- **Multi-select**: Shift-click in hierarchy or viewport to add to selection
- **Content browser**: Bottom panel with built-in mesh buttons (Cube, Sphere, Plane, Cylinder, Cone)
- **Copy/paste**: Ctrl+C copies selected entities, Ctrl+V pastes with offset
- **Snap-to-grid**: G key toggles grid snapping for gizmo translate
- **Cylinder & Cone meshes**: New built-in primitive meshes
- **Multi-entity gizmo**: Transform applies to all selected entities

### Networking
- Client prediction system wired into ECS: `apply_prediction_system()`
- `record_prediction_for_entity()` and `reconcile_entity()` helper functions
- Smooth correction with configurable smoothing factor

### Infrastructure
- 740+ tests
- Version bump to 0.5.0

## v0.4.0 (2026-03-23)

### File-System-First Architecture
Inspired by the "environment IS state" paradigm — the file system is now the canonical source of truth.

- **File-watching hot reload**: Editor polls level and asset directories every ~1 second. External edits (e.g., in VSCode) are detected and the viewport reloads automatically
- **Auto-save / journaling**: Dirty tracking on entity edits (spawn, despawn, gizmo drag, inspector). Debounced auto-save to `.euca_autosave.json` after 5 seconds of inactivity. Startup recovery detection
- **Enhanced FileWatcher**: Added `watch_file()` for individual files, improved test coverage

### Composable CLI Asset Pipeline
Offline file-processing tools following the Unix philosophy — each does one thing, outputs JSON.

- `euca asset info <file>` — Show glTF metadata (mesh count, vertices, triangles, skeleton, animations)
- `euca asset optimize <file>` — Run mesh optimization (dedup, tangents, cache reorder) and report stats
- `euca asset lod <file> --levels N` — Generate LOD chain with QEM simplification, report per-level stats

### AI Agent Discoverability
Self-describing CLI — always in sync because it's generated from clap definitions.

- `euca discover` — Human-readable overview of all 30+ command groups
- `euca discover --json` — Machine-readable JSON manifest with all commands, args, types, and descriptions
- `euca discover <group>` — Detailed view of a specific group (e.g., `euca discover entity`)
- Tags each command with `requires_engine` flag (offline vs. online)

### Editor
- Inspector panel returns dirty signal for auto-save tracking
- Level entities appear in viewport on selection (carried from v0.3.0)

### Infrastructure
- 720+ tests
- Version bump to 0.4.0

## v0.3.0 (2026-03-21)

### Rendering
- Deferred rendering path (G-buffer + lighting pass)
- Screen-space reflections (SSR)
- Volumetric fog with god rays
- SSAO (GTAO) + bilateral blur
- FXAA anti-aliasing
- Post-process stack (bloom, color grading, ACES tonemapping)
- GPU-driven rendering (draw indirect)
- HZB occlusion culling
- LOD system (screen-space mesh selection)
- Depth+normal pre-pass
- Clustered light culling (256+ lights)
- Render quality presets (Low/Medium/High/Ultra)
- Material system: transparency, emissive, metallic/roughness/AO textures
- Shader extraction to .wgsl files
- Foliage system (Poisson disk instancing)
- HLOD (hierarchical LOD)
- Decal system
- Particle render data pipeline
- Compute shader infrastructure

### Physics
- Collision layers and masks
- Mass and inertia properties
- Scene queries (overlap_sphere, sweep_sphere, raycast_world)
- Collision events
- Character controller (capsule, ground detection, slopes, jumping)
- Vehicle physics (suspension, tires, engine, transmission)

### Animation
- Animation blending and state machines
- Blend spaces (1D parametric)
- Root motion extraction
- Animation events
- Montage player
- Inverse kinematics (two-bone IK, FABRIK, look-at)

### AI
- Behavior trees with blackboard
- Decorators, composites, action/condition nodes

### Gameplay
- Role-aware targeting (heroes/minions/towers have different priorities)
- Persistent target tracking (CurrentTarget component)
- March direction (units advance toward enemy base)

### Scale
- World streaming / chunk loading
- Spatial index (uniform grid queries)
- Prefab system (PrefabRegistry, spawn by name)

### Networking
- Property replication with delta compression
- RPCs (ServerRpc, ClientRpc)
- Replication priority

### Performance
- SIMD math (SSE2/NEON) for Vec3/Vec4/Mat4/Quat
- ECS query caching with generation invalidation
- Parallel system execution (ParallelSchedule)
- Frame profiler (per-system timing via CLI)
- Criterion benchmarks (ECS, physics, math)

### Apple Silicon
- Metal TBDR render hints
- Unified memory SmartBuffer
- Optimized 32-thread compute dispatch

### Scripting
- Lua scripting via mlua (hot reload, sandboxing, ECS bridge)

### UI
- Runtime UI framework (anchored layout, flex, widgets)

### Terrain
- Heightmap terrain (chunk LOD, splatting, physics, brush editing)

### Audio
- Bus mixing hierarchy (Master/Music/SFX/Voice/UI)
- Reverb zones
- Sound priority and occlusion

### Reflection
- Runtime field access (field_ref, field_mut, set_field)
- TypeRegistry for dynamic type creation
- JSON serialization

### CLI / Agent API
- 70+ HTTP endpoints
- CLI commands: terrain, prefab, material, postprocess, fog, foliage, profile
- Render quality presets via CLI

### Infrastructure
- 715+ tests
- Criterion benchmark suites
- CI: build-essential for mlua
