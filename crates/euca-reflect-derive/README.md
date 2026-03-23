# euca-reflect-derive

Proc macro crate providing `#[derive(Reflect)]` for automatic runtime reflection.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `#[derive(Reflect)]` for named structs -- generates `field_ref`, `field_mut`, `set_field`, and `type_info`
- Tuple struct support with index-based field access
- Unit struct support as leaf types
- Enum support with variant name reflection
- Works with generics (respects `where` clauses)

## Usage

```rust
use euca_reflect::Reflect;

#[derive(Clone, Debug, Default, Reflect)]
struct Health {
    current: f32,
    max: f32,
}

let h = Health { current: 80.0, max: 100.0 };
assert_eq!(h.type_name(), "Health");
assert_eq!(h.fields().len(), 2);
```

## License

MIT
