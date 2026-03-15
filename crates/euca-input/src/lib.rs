use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A named game action (e.g., "shoot", "move_forward", "jump").
/// Actions are strings so they can be defined by the game, not the engine.
pub type Action = String;

/// Raw input key/button identifier.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum InputKey {
    // Keyboard
    Key(String), // e.g., "W", "Space", "Escape"
    // Mouse
    MouseLeft,
    MouseRight,
    MouseMiddle,
    // Gamepad (future)
    GamepadButton(u32),
}

/// Current state of all inputs for one frame.
///
/// This is an ECS resource. Updated at the start of each frame from OS events
/// (for humans) or from the agent API (for AI agents).
#[derive(Clone, Debug, Default)]
pub struct InputState {
    /// Keys currently held down.
    pressed: HashSet<InputKey>,
    /// Keys pressed this frame (just went down).
    just_pressed: HashSet<InputKey>,
    /// Keys released this frame (just went up).
    just_released: HashSet<InputKey>,
    /// Mouse position (pixels from top-left).
    pub mouse_position: [f32; 2],
    /// Mouse movement delta this frame.
    pub mouse_delta: [f32; 2],
    /// Scroll wheel delta this frame.
    pub scroll_delta: f32,
    /// The world tick when this input was captured (for networking).
    pub tick: u64,
}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Call at the start of each frame to clear per-frame state.
    pub fn begin_frame(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
        self.mouse_delta = [0.0, 0.0];
        self.scroll_delta = 0.0;
    }

    /// Record a key press event.
    pub fn press(&mut self, key: InputKey) {
        if self.pressed.insert(key.clone()) {
            self.just_pressed.insert(key);
        }
    }

    /// Record a key release event.
    pub fn release(&mut self, key: InputKey) {
        if self.pressed.remove(&key) {
            self.just_released.insert(key);
        }
    }

    /// Is the key currently held down?
    pub fn is_pressed(&self, key: &InputKey) -> bool {
        self.pressed.contains(key)
    }

    /// Was the key pressed this frame?
    pub fn is_just_pressed(&self, key: &InputKey) -> bool {
        self.just_pressed.contains(key)
    }

    /// Was the key released this frame?
    pub fn is_just_released(&self, key: &InputKey) -> bool {
        self.just_released.contains(key)
    }

    /// Record mouse movement.
    pub fn move_mouse(&mut self, dx: f32, dy: f32) {
        self.mouse_delta[0] += dx;
        self.mouse_delta[1] += dy;
    }

    /// Set absolute mouse position.
    pub fn set_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_position = [x, y];
    }
}

/// Maps physical inputs to game actions.
///
/// Example: "W" → "move_forward", "Space" → "jump", MouseLeft → "shoot"
#[derive(Clone, Debug, Default)]
pub struct ActionMap {
    bindings: HashMap<InputKey, Action>,
}

impl ActionMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a key to an action.
    pub fn bind(&mut self, key: InputKey, action: impl Into<Action>) -> &mut Self {
        self.bindings.insert(key, action.into());
        self
    }

    /// Get the action bound to a key (if any).
    pub fn action_for(&self, key: &InputKey) -> Option<&str> {
        self.bindings.get(key).map(|s| s.as_str())
    }

    /// Get all actions that are currently active (their bound key is pressed).
    pub fn active_actions(&self, input: &InputState) -> Vec<&str> {
        self.bindings
            .iter()
            .filter(|(key, _)| input.is_pressed(key))
            .map(|(_, action)| action.as_str())
            .collect()
    }

    /// Get actions that just started this frame.
    pub fn just_started_actions(&self, input: &InputState) -> Vec<&str> {
        self.bindings
            .iter()
            .filter(|(key, _)| input.is_just_pressed(key))
            .map(|(_, action)| action.as_str())
            .collect()
    }
}

/// Timestamped input snapshot for networking.
///
/// The server receives these from clients and replays them at the correct tick.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputSnapshot {
    /// The tick this input was generated at.
    pub tick: u64,
    /// Keys pressed in this snapshot.
    pub pressed_keys: Vec<InputKey>,
    /// Mouse position.
    pub mouse_position: [f32; 2],
    /// Mouse movement delta.
    pub mouse_delta: [f32; 2],
}

impl InputSnapshot {
    /// Create a snapshot from the current input state.
    pub fn capture(input: &InputState) -> Self {
        Self {
            tick: input.tick,
            pressed_keys: input.pressed.iter().cloned().collect(),
            mouse_position: input.mouse_position,
            mouse_delta: input.mouse_delta,
        }
    }

    /// Apply this snapshot to an InputState (for server-side replay).
    pub fn apply_to(&self, input: &mut InputState) {
        input.begin_frame();
        input.tick = self.tick;
        input.mouse_position = self.mouse_position;
        input.mouse_delta = self.mouse_delta;
        for key in &self.pressed_keys {
            input.press(key.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn press_and_release() {
        let mut input = InputState::new();
        let w = InputKey::Key("W".into());

        input.press(w.clone());
        assert!(input.is_pressed(&w));
        assert!(input.is_just_pressed(&w));

        input.begin_frame();
        assert!(input.is_pressed(&w));
        assert!(!input.is_just_pressed(&w)); // not just pressed anymore

        input.release(w.clone());
        assert!(!input.is_pressed(&w));
        assert!(input.is_just_released(&w));
    }

    #[test]
    fn action_map() {
        let mut map = ActionMap::new();
        map.bind(InputKey::Key("W".into()), "move_forward");
        map.bind(InputKey::Key("Space".into()), "jump");
        map.bind(InputKey::MouseLeft, "shoot");

        let mut input = InputState::new();
        input.press(InputKey::Key("W".into()));
        input.press(InputKey::MouseLeft);

        let active = map.active_actions(&input);
        assert!(active.contains(&"move_forward"));
        assert!(active.contains(&"shoot"));
        assert!(!active.contains(&"jump"));
    }

    #[test]
    fn input_snapshot_roundtrip() {
        let mut input = InputState::new();
        input.tick = 42;
        input.press(InputKey::Key("W".into()));
        input.press(InputKey::Key("A".into()));
        input.set_mouse_position(100.0, 200.0);
        input.move_mouse(5.0, -3.0);

        let snapshot = InputSnapshot::capture(&input);
        assert_eq!(snapshot.tick, 42);
        assert_eq!(snapshot.pressed_keys.len(), 2);

        // Apply to fresh state
        let mut restored = InputState::new();
        snapshot.apply_to(&mut restored);
        assert_eq!(restored.tick, 42);
        assert!(restored.is_pressed(&InputKey::Key("W".into())));
        assert!(restored.is_pressed(&InputKey::Key("A".into())));
        assert_eq!(restored.mouse_position, [100.0, 200.0]);
    }

    #[test]
    fn mouse_delta_accumulates() {
        let mut input = InputState::new();
        input.move_mouse(1.0, 2.0);
        input.move_mouse(3.0, 4.0);
        assert_eq!(input.mouse_delta, [4.0, 6.0]);

        input.begin_frame();
        assert_eq!(input.mouse_delta, [0.0, 0.0]);
    }
}
