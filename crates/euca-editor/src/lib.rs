pub mod gizmo;
mod panels;
mod scene_file;
pub mod undo;

pub use gizmo::GizmoState;
pub use panels::{
    SpawnRequest, TerrainBrushAction, TerrainBrushMode, ToolbarAction, content_browser_panel,
    hierarchy_panel, inspector_panel, terrain_panel, toolbar_panel,
};
pub use scene_file::{
    PrefabRegistry, SCENE_VERSION, SceneEntity, SceneFile, load_scene_into_world,
};
pub use undo::UndoHistory;

/// Editor state: tracks selection, play/pause, gizmo, undo history, dirty tracking.
pub struct EditorState {
    /// Currently selected entity indices.
    pub selected_entities: Vec<u32>,
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
    /// Clipboard for copy/paste of entities.
    pub clipboard: Vec<SceneEntity>,
    /// Whether snap-to-grid is enabled.
    pub snap_to_grid: bool,
    /// Grid cell size used when snap-to-grid is enabled.
    pub grid_size: f32,
    /// Whether the terrain brush is currently active.
    pub terrain_brush_active: bool,
    /// Current terrain brush sculpting mode.
    pub terrain_brush_mode: TerrainBrushMode,
    /// Brush radius in world units.
    pub terrain_brush_radius: f32,
    /// Brush strength (how much each stroke affects the heightmap).
    pub terrain_brush_strength: f32,
    /// Target height used by the Flatten brush mode.
    pub terrain_brush_target_height: f32,
}

impl EditorState {
    pub fn new() -> Self {
        Self {
            selected_entities: Vec::new(),
            playing: false,
            step_once: false,
            reset_requested: false,
            gizmo: GizmoState::new(),
            undo: UndoHistory::new(),
            dirty: false,
            last_dirty_time: 0.0,
            clipboard: Vec::new(),
            snap_to_grid: false,
            grid_size: 1.0,
            terrain_brush_active: false,
            terrain_brush_mode: TerrainBrushMode::Raise,
            terrain_brush_radius: 5.0,
            terrain_brush_strength: 0.1,
            terrain_brush_target_height: 0.0,
        }
    }

    /// Clear selection and select a single entity.
    pub fn select(&mut self, idx: u32) {
        self.selected_entities.clear();
        self.selected_entities.push(idx);
    }

    /// Toggle an entity in the selection (add if absent, remove if present).
    pub fn toggle_select(&mut self, idx: u32) {
        if let Some(pos) = self.selected_entities.iter().position(|&i| i == idx) {
            self.selected_entities.remove(pos);
        } else {
            self.selected_entities.push(idx);
        }
    }

    /// Add an entity to the selection if not already present.
    pub fn add_select(&mut self, idx: u32) {
        if !self.selected_entities.contains(&idx) {
            self.selected_entities.push(idx);
        }
    }

    /// Clear all selected entities.
    pub fn clear_selection(&mut self) {
        self.selected_entities.clear();
    }

    /// Check whether an entity is currently selected.
    pub fn is_selected(&self, idx: u32) -> bool {
        self.selected_entities.contains(&idx)
    }

    /// Return the primary (first) selected entity, if any.
    pub fn primary_selected(&self) -> Option<u32> {
        self.selected_entities.first().copied()
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

/// Try to find a living entity with the given index by scanning generations 0..16.
///
/// The ECS recycles entity slots: when an entity is despawned its index may be
/// reused for a new entity with an incremented *generation*. The editor only
/// stores the raw index (e.g. from a selection list or an undo record), so it
/// does not know which generation is currently alive. Scanning a fixed window
/// of 16 generations is the simplest correct approach — it is O(1) in the
/// number of entities and covers all practical recycling depths. A direct
/// `index → generation` lookup would require exposing ECS internals that are
/// intentionally kept private.
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
        assert!(state.selected_entities.is_empty());
    }

    #[test]
    fn select_single() {
        let mut state = EditorState::new();
        state.select(5);
        assert_eq!(state.selected_entities, vec![5]);
        assert_eq!(state.primary_selected(), Some(5));
        state.select(10);
        assert_eq!(state.selected_entities, vec![10]);
    }

    #[test]
    fn add_and_toggle_select() {
        let mut state = EditorState::new();
        state.add_select(1);
        state.add_select(2);
        state.add_select(1); // duplicate — no-op
        assert_eq!(state.selected_entities, vec![1, 2]);
        assert!(state.is_selected(1));
        assert!(state.is_selected(2));
        state.toggle_select(1); // remove
        assert!(!state.is_selected(1));
        assert_eq!(state.selected_entities, vec![2]);
        state.toggle_select(3); // add
        assert_eq!(state.selected_entities, vec![2, 3]);
    }

    #[test]
    fn clear_selection() {
        let mut state = EditorState::new();
        state.select(1);
        state.add_select(2);
        state.clear_selection();
        assert!(state.selected_entities.is_empty());
        assert_eq!(state.primary_selected(), None);
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
