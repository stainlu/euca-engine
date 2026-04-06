# euca-render

RHI-abstracted PBR renderer with compile-time backend selection (wgpu cross-platform, native Metal on Apple Silicon).

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Architecture

The renderer is generic over `RenderDevice` -- a trait defined in `euca-rhi`:

- **`WgpuDevice`** -- Cross-platform backend via wgpu 27 (Vulkan, Metal-via-wgpu, DX12, WebGPU)
- **`MetalDevice`** -- Native Metal backend via `objc2-metal` for direct Apple Silicon access

All GPU resources (buffers, textures, pipelines, bind groups) flow through the trait's associated types. Backend selection is compile-time via generics with `WgpuDevice` as the default.

## Features

- RHI trait abstraction (`RenderDevice`) with wgpu and native Metal backends
- Forward+ rendering as primary path with deferred opt-in (`GBuffer`, `DeferredPipeline`)
- PBR materials (metallic-roughness workflow) with alpha modes and material handles
- Bindless material system (single bind group for all materials + textures)
- Native MSL shaders: PBR (Cook-Torrance BRDF), cascaded shadows, procedural sky
- Cascaded shadow maps with PCSS soft shadows
- SSAO, SSR, SSGI, TAA, motion blur, depth of field
- Volumetric fog (ray-marched scattering)
- GPU-driven rendering with compute frustum culling and indirect draw calls
- LOD selection, HLOD clusters, HZB occlusion culling
- GPU particle system (compute emit/update, instanced billboard render)
- Clustered light grid for 256+ lights
- Decal rendering, foliage instancing, light probes (spherical harmonics)
- Post-processing stack (bloom, color grading, ACES tone mapping, FXAA)
- Hardware survey with Apple Silicon Metal optimization hints

## Usage

```rust
use euca_render::*;

// Renderer defaults to WgpuDevice via the RHI trait
let renderer = Renderer::new(&gpu_context);
let mesh = renderer.upload_mesh(&gpu_context, &mesh_data);
let material = renderer.upload_material(&gpu_context, &material_data);
let cmd = DrawCommand { mesh, material, model_matrix, aabb: None, is_water: false };
```

## License

MIT
