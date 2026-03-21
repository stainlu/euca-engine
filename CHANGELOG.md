# Changelog

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
