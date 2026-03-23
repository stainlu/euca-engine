# euca-input

Input handling: keyboard, mouse, and gamepad state with action mapping and input contexts.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `InputState` resource tracking pressed, just-pressed, and just-released keys per frame
- `ActionMap` binding physical inputs to named game actions with JSON serialization
- `InputKey` variants: keyboard keys, mouse buttons, gamepad buttons and axes
- `GamepadState` for analog stick axes and button tracking
- `InputContextStack` for layered input contexts (Gameplay, Menu, Editor)
- `InputSnapshot` for network-serializable input capture and replay
- Mouse position, delta, and scroll wheel tracking

## Usage

```rust
use euca_input::*;

let mut input = InputState::new();
input.press(InputKey::Key("W".into()));

let mut actions = ActionMap::new();
actions.bind(InputKey::Key("W".into()), "move_forward");
actions.bind(InputKey::Key("Space".into()), "jump");

let active = actions.active_actions(&input);
```

## License

MIT
