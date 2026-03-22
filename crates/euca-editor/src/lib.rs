pub mod gizmo;
mod panels;
mod scene_file;
pub mod undo;

pub use gizmo::GizmoState;
pub use panels::{SpawnRequest, ToolbarAction, hierarchy_panel, inspector_panel, toolbar_panel};
pub use scene_file::{
    PrefabRegistry, SCENE_VERSION, SceneEntity, SceneFile, load_scene_into_world,
};
pub use undo::UndoHistory;

/// Editor state: tracks selection, play/pause, gizmo, undo history, dirty tracking.
pub struct EditorState {
    /// Currently selected entity index (if any).
    pub selected_entity: Option<u32>,
    /// Whether the simulation is running.
    pub playing: bool,
    /// Whether to advance a single tick (when paused).
    pub step_once: bool,
    /// Whether a reset was requested (stop + restore initial scene).
    pub reset_requested: bool,
    /// Transform gizmo state.
    pub gizmo: GizmoState,
    /// Undo/redo history.
    pub undo: UndoHistory,
    /// Whether the scene has unsaved changes.
    pub dirty: bool,
    /// Elapsed time (seconds) when dirty was last set — used for auto-save debounce.
    pub last_dirty_time: f64,
}

impl EditorState {
    pub fn new() -> Self {
        Self {
            selected_entity: None,
            playing: false,
            step_once: false,
            reset_requested: false,
            gizmo: GizmoState::new(),
            undo: UndoHistory::new(),
            dirty: false,
            last_dirty_time: 0.0,
        }
    }

    /// Should the simulation tick this frame?
    pub fn should_tick(&mut self) -> bool {
        if self.playing {
            return true;
        }
        if self.step_once {
            self.step_once = false;
            return true;
        }
        false
    }

    /// Mark the scene as dirty (has unsaved changes). Resets debounce timer.
    pub fn mark_dirty(&mut self, elapsed: f64) {
        self.dirty = true;
        self.last_dirty_time = elapsed;
    }
}

impl Default for EditorState {
    fn default() -> Self {
        Self::new()
    }
}

/// Try to find a living entity with the given index (checks generations 0..16).
pub fn find_alive_entity(world: &euca_ecs::World, index: u32) -> Option<euca_ecs::Entity> {
    for g in 0..16 {
        let e = euca_ecs::Entity::from_raw(index, g);
        if world.is_alive(e) {
            return Some(e);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paused_by_default() {
        let state = EditorState::new();
        assert!(!state.playing);
        assert!(state.selected_entity.is_none());
    }

    #[test]
    fn should_tick_when_playing() {
        let mut state = EditorState::new();
        state.playing = true;
        assert!(state.should_tick());
        assert!(state.should_tick());
    }

    #[test]
    fn should_tick_once_on_step() {
        let mut state = EditorState::new();
        state.step_once = true;
        assert!(state.should_tick());
        assert!(!state.should_tick());
    }

    #[test]
    fn paused_does_not_tick() {
        let mut state = EditorState::new();
        assert!(!state.should_tick());
    }

    #[test]
    fn dirty_tracking() {
        let mut state = EditorState::new();
        assert!(!state.dirty);
        state.mark_dirty(1.0);
        assert!(state.dirty);
        assert_eq!(state.last_dirty_time, 1.0);
        state.mark_dirty(3.5);
        assert_eq!(state.last_dirty_time, 3.5);
    }
}
