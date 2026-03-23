# euca-script

Embedded Lua scripting with hot reload, sandboxing, and ECS bridge.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `ScriptEngine` resource managing the mlua Lua VM with instruction-count sandboxing
- `ScriptComponent` binding entities to Lua update functions (`on_update` by default)
- ECS bridge API: `euca.spawn()`, `euca.despawn()`, `euca.set_position()`, `euca.get_health()`, etc.
- Script event system: `euca.on("event", fn)` / `euca.emit("event", data)` for Lua-side handlers
- `ScriptWatcher` for hot-reload of modified `.lua` files
- `script_tick_system` entry point ticking all scripted entities per frame
- Configurable instruction limit (`DEFAULT_INSTRUCTION_LIMIT = 100,000`)

## Usage

```rust
use euca_script::*;

let mut engine = ScriptEngine::new(DEFAULT_INSTRUCTION_LIMIT).unwrap();
engine.load_string("patrol.lua", r#"
    function on_update(entity_id, dt)
        euca.set_position(entity_id, 0, 0, 0)
    end
"#).unwrap();

world.insert_resource(engine);
world.insert(entity, ScriptComponent::new("patrol.lua"));

script_tick_system(&mut world, dt);
```

## License

MIT
