# euca-reflect

Runtime reflection system: field access, type registry, and JSON serialization.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `Reflect` trait with `field_ref`, `field_mut`, `set_field`, and `clone_reflect`
- `TypeRegistry` for dynamic type creation by name
- `TypeInfo` and `FieldInfo` for runtime type introspection
- Built-in `Reflect` implementations for primitives (`f32`, `i32`, `bool`, `String`, etc.)
- Optional JSON serialization/deserialization via the `json` feature
- Re-exports `#[derive(Reflect)]` from `euca-reflect-derive`

## Usage

```rust
use euca_reflect::*;

let mut registry = TypeRegistry::new();
registry.register::<MyComponent>();

let value: &dyn Reflect = &my_component;
for (name, repr) in value.fields() {
    println!("{name}: {repr}");
}
```

## License

MIT
