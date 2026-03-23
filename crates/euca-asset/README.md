# euca-asset

Asset loading pipeline: glTF import, LOD generation, mesh optimization, and skeletal animation data.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `load_gltf` for importing glTF scenes with meshes, textures, and skeletons
- `AssetStore` with `AssetHandle` and `LoadState` tracking
- `generate_lod_chain` and `simplify_mesh` for automatic LOD generation (QEM)
- Mesh optimization: vertex cache reordering, deduplication, tangent computation
- `Skeleton` for joint hierarchies and bind poses
- `AnimationClipData` / `AnimationProperty` for clip-level animation data
- `SkeletalAnimator` and `skeletal_animation_system` for runtime playback
- `FileWatcher` for hot-reload of modified assets

## Usage

```rust
use euca_asset::*;

let scene = load_gltf("assets/character.glb").unwrap();
for mesh in &scene.meshes {
    let lods = generate_lod_chain(&mesh.vertices, &mesh.indices, 4);
}
```

## License

MIT
