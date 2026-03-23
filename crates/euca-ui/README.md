# euca-ui

Renderer-agnostic runtime UI framework: anchored layout, flex, widgets, and world-space projection.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `UiNode` component with anchor, margin, flex direction, and percentage/pixel sizing
- Flex layout solver: `FlexDirection`, `JustifyContent`, `AlignItems`
- Widget components: `UiText`, `UiImage`, `UiButton`, `UiProgressBar`, `UiPanel`
- `ui_layout_system` resolves node trees into `ResolvedRect` screen-space rectangles
- `ui_input_system` for hit-testing and button interaction state
- `UiDrawCommand` output for renderer consumption (textured/colored quads)
- World-space UI: entities with `UiNode` + `GlobalTransform` project 3D to screen
- `UiInputConsumed` flag for game-layer input filtering

## Usage

```rust
use euca_ui::*;

let node = UiNode::new()
    .with_anchor(Anchor::TopLeft)
    .with_size(UiSize::new(Val::Px(200.0), Val::Px(50.0)));

ui_layout_system(&mut world, &UiViewport { width: 1920.0, height: 1080.0 });
let commands = collect_ui_draw_data(&world);
```

## License

MIT
