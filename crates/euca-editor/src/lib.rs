mod panels;

pub use panels::{hierarchy_panel, inspector_panel, toolbar_panel};

/// Editor state: tracks selection, play/pause, etc.
pub struct EditorState {
    /// Currently selected entity index (if any).
    pub selected_entity: Option<u32>,
    /// Whether the simulation is running.
    pub playing: bool,
    /// Whether to advance a single tick (when paused).
    pub step_once: bool,
}

impl EditorState {
    pub fn new() -> Self {
        Self {
            selected_entity: None,
            playing: false,
            step_once: false,
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
