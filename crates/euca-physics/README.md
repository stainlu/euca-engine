# euca-physics

Collision detection, raycasting, rigid body dynamics, joints, and character/vehicle controllers.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `Collider` component with AABB, sphere, and capsule shapes
- Broadphase collision detection with collision layers
- `raycast_world`, `overlap_sphere`, and `sweep_sphere` spatial queries
- `PhysicsBody` with rigid body types (Dynamic, Static, Kinematic) and sleep detection
- `Joint` system with fixed, hinge, and spring joint kinds
- `CharacterController` for grounded movement with slopes and steps
- `Vehicle` physics with engine curves, wheel configs, and suspension
- Configurable gravity, substeps, and physics accumulator

## Usage

```rust
use euca_physics::*;

let ray = Ray { origin, direction };
if let Some(hit) = raycast_world(&world, &ray, 100.0) {
    println!("Hit entity {:?} at distance {}", hit.entity, hit.distance);
}
```

## License

MIT
