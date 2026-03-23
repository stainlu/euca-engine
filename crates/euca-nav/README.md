# euca-nav

Grid-based navigation mesh with A* pathfinding, path smoothing, and RVO avoidance.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `NavMesh` with configurable `GridConfig` (cell size, dimensions, walkability)
- `build_navmesh_from_world` to generate navmesh from scene colliders
- 8-connected A* pathfinding via `find_path`
- `smooth_path` for corner-cutting path optimization
- `NavAgent` component with speed and steering radius
- `PathGoal` component for per-entity navigation targets
- `pathfinding_system` and `steering_system` for ECS integration
- RVO (Reciprocal Velocity Obstacle) avoidance module

## Usage

```rust
use euca_nav::*;

let navmesh = NavMesh::from_grid(GridConfig::default());
world.insert_resource(navmesh);

world.insert(entity, NavAgent::new(5.0));
world.insert(entity, PathGoal::new(target_position));

pathfinding_system(&mut world);
steering_system(&mut world, dt);
```

## License

MIT
