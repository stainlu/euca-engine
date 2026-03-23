# euca-ecs

Archetype-based Entity Component System with generational entities, parallel queries, and change detection.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- Generational `Entity` handles with O(1) alive checks
- Archetype storage for cache-friendly component iteration
- `Query<(&A, &mut B), With<C>>` with compile-time access tracking
- `Schedule` and `ParallelSchedule` for deterministic, batched system execution
- `Resources` for singleton world data (`Res<T>`, `ResMut<T>`)
- `Events<T>` channel for decoupled inter-system communication
- `Commands` for deferred entity spawning and component insertion
- `WorldSnapshot` for serializable world state capture
- Change detection via per-component tick tracking

## Usage

```rust
use euca_ecs::*;

let mut world = World::new();
let entity = world.spawn(Position { x: 0.0, y: 0.0 });
world.insert(entity, Velocity { x: 1.0, y: 0.0 });

let query = Query::<(&Position, &Velocity)>::new(&world);
for (pos, vel) in query.iter() {
    // process entities with both Position and Velocity
}
```

## License

MIT
