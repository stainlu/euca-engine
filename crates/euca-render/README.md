# euca-render

wgpu-based PBR renderer with deferred shading, cascaded shadows, and modern post-processing.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- Deferred rendering pipeline (`GBuffer`, `DeferredPipeline`) with forward fallback
- PBR materials with alpha modes and material handles
- Clustered light grid for efficient many-light scenes
- GPU-driven rendering with indirect draw buffers and frustum culling
- Cascaded shadow maps, SSAO, SSR, TAA, volumetric fog
- LOD selection, HLOD clusters, and occlusion culling (HZB)
- GPU particle system, decal rendering, foliage instancing
- Light probes with spherical harmonics
- Post-processing stack (bloom, color grading, FXAA)
- Hardware survey and Metal render pass hints

## Usage

```rust
use euca_render::*;

let renderer = Renderer::new(&gpu_context, RenderQuality::High);
let mesh = MeshHandle(0);
let material = MaterialHandle(0);
let cmd = DrawCommand { mesh, material, transform };
```

## License

MIT
