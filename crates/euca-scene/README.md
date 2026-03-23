# euca-scene

Transform hierarchy, spatial indexing, prefab system, and world streaming.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `LocalTransform` / `GlobalTransform` with dirty-flag BFS propagation
- `Parent` / `Children` hierarchy components
- `SpatialIndex` for fast spatial queries with `spatial_index_update_system`
- `Prefab` and `PrefabRegistry` for templated entity spawning
- World streaming with chunk-based loading/unloading (`StreamingConfig`, `WorldChunk`)
- Change-detection optimization -- only recomputes transforms modified since last tick

## Usage

```rust
use euca_scene::*;
use euca_math::{Transform, Vec3};
use euca_ecs::World;

let mut world = World::new();
let parent = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(10.0, 0.0, 0.0))));
world.insert(parent, GlobalTransform::default());

transform_propagation_system(&mut world);
```

## License

MIT
