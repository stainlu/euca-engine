mod panels;
mod scene_file;

pub use panels::{SpawnRequest, ToolbarAction, hierarchy_panel, inspector_panel, toolbar_panel};
pub use scene_file::{SceneEntity, SceneFile, load_scene_into_world};

/// Editor state: tracks selection, play/pause, etc.
pub struct EditorState {
    /// Currently selected entity index (if any).
    pub selected_entity: Option<u32>,
    /// Whether the simulation is running.
    pub playing: bool,
    /// Whether to advance a single tick (when paused).
    pub step_once: bool,
    /// Whether a reset was requested (stop + restore initial scene).
    pub reset_requested: bool,
}

impl EditorState {
    pub fn new() -> Self {
        Self {
            selected_entity: None,
            playing: false,
            step_once: false,
            reset_requested: false,
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
}

impl Default for EditorState {
    fn default() -> Self {
        Self::new()
    }
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
        assert!(state.should_tick()); // continues ticking
    }

    #[test]
    fn should_tick_once_on_step() {
        let mut state = EditorState::new();
        state.step_once = true;
        assert!(state.should_tick()); // first call returns true
        assert!(!state.should_tick()); // second call returns false (step consumed)
    }

    #[test]
    fn paused_does_not_tick() {
        let mut state = EditorState::new();
        assert!(!state.should_tick());
    }
}
