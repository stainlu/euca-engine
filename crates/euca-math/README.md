# euca-math

Core math primitives for game development: vectors, matrices, quaternions, transforms, and AABBs.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `Vec2`, `Vec3`, `Vec4` with standard arithmetic and geometric operations
- `Mat4` for 4x4 transformation matrices (perspective, look-at, TRS)
- `Quat` for rotation representation and interpolation
- `Transform` combining translation, rotation (quat), and scale
- `Aabb` for axis-aligned bounding boxes
- SIMD acceleration (SSE2 on x86_64, NEON on aarch64) behind a `simd` feature flag
- Optional `reflect` feature for runtime reflection integration

## Usage

```rust
use euca_math::*;

let t = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0));
let rotated = t.mul(Transform::from_rotation(Quat::from_axis_angle(Vec3::Y, 0.5)));
let matrix = Mat4::from_transform(&rotated);
```

## License

MIT
