# euca-editor

egui-based editor: gizmos, hierarchy panel, inspector, multi-select, undo/redo, terrain brushes, and auto-save.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `EditorState` tracking selection, play/pause, dirty flag, and auto-save debounce
- `GizmoState` for translate/rotate/scale gizmo interaction
- `UndoHistory` with undo/redo stack
- Multi-entity selection with toggle, add, and clear operations
- `hierarchy_panel`, `inspector_panel`, `toolbar_panel`, `content_browser_panel`, `terrain_panel`
- `SceneFile` serialization with versioned format and `PrefabRegistry`
- Terrain brush modes: Raise, Lower, Smooth, Flatten, Paint
- Snap-to-grid with configurable grid size
- Clipboard support for entity copy/paste
- Step-once simulation for frame-by-frame debugging

## Usage

```rust
use euca_editor::*;

let mut state = EditorState::new();
state.select(entity_index);

if state.should_tick() {
    // advance simulation one frame
}

hierarchy_panel(&egui_ctx, &world, &mut state);
inspector_panel(&egui_ctx, &world, &mut state);
```

## License

MIT
