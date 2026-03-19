use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A named game action (e.g., "shoot", "move_forward", "jump").
/// Actions are strings so they can be defined by the game, not the engine.
pub type Action = String;

/// Active input context — determines which action map is active.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputContext {
    Gameplay,
    Menu,
    Editor,
}

impl Default for InputContext {
    fn default() -> Self {
        Self::Gameplay
    }
}

/// Stack of input contexts. Top of stack is the active context.
#[derive(Clone, Debug, Default)]
pub struct InputContextStack {
    stack: Vec<InputContext>,
}

impl InputContextStack {
    pub fn new() -> Self {
        Self {
            stack: vec![InputContext::Gameplay],
        }
    }

    /// Push a new context (becomes active).
    pub fn push(&mut self, ctx: InputContext) {
        self.stack.push(ctx);
    }

    /// Pop the top context. Returns the popped context.
    pub fn pop(&mut self) -> Option<InputContext> {
        if self.stack.len() > 1 {
            self.stack.pop()
        } else {
            None // Don't pop the last context
        }
    }

    /// Get the active (top) context.
    pub fn active(&self) -> &InputContext {
        self.stack.last().unwrap_or(&InputContext::Gameplay)
    }
}

/// Raw input key/button identifier.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum InputKey {
    // Keyboard
    Key(String), // e.g., "W", "Space", "Escape"
    // Mouse
    MouseLeft,
    MouseRight,
    MouseMiddle,
    // Gamepad
    GamepadButton(u32),
    GamepadAxis(u32, GamepadAxisType),
}

/// Which axis of a gamepad stick or trigger.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum GamepadAxisType {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
    LeftTrigger,
    RightTrigger,
}

/// Gamepad state tracking.
#[derive(Clone, Debug, Default)]
pub struct GamepadState {
    /// Axis values keyed by (gamepad_id, axis_type).
    pub axes: HashMap<(u32, GamepadAxisType), f32>,
    /// Buttons currently held.
    pub buttons: HashSet<(u32, u32)>,
}

impl GamepadState {
    pub fn set_axis(&mut self, gamepad: u32, axis: GamepadAxisType, value: f32) {
        self.axes.insert((gamepad, axis), value);
    }

    pub fn axis_value(&self, gamepad: u32, axis: &GamepadAxisType) -> f32 {
        self.axes
            .get(&(gamepad, axis.clone()))
            .copied()
            .unwrap_or(0.0)
    }

    pub fn press_button(&mut self, gamepad: u32, button: u32) {
        self.buttons.insert((gamepad, button));
    }

    pub fn release_button(&mut self, gamepad: u32, button: u32) {
        self.buttons.remove(&(gamepad, button));
    }

    pub fn is_button_pressed(&self, gamepad: u32, button: u32) -> bool {
        self.buttons.contains(&(gamepad, button))
    }
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

    /// Remove a binding for a key.
    pub fn unbind(&mut self, key: &InputKey) -> Option<Action> {
        self.bindings.remove(key)
    }

    /// Get all current bindings.
    pub fn bindings(&self) -> &HashMap<InputKey, Action> {
        &self.bindings
    }

    /// Serialize bindings to JSON string.
    pub fn save_to_json(&self) -> String {
        let pairs: Vec<(&InputKey, &Action)> = self.bindings.iter().collect();
        serde_json::to_string_pretty(&pairs).unwrap_or_default()
    }

    /// Load bindings from JSON string (replaces existing bindings).
    pub fn load_from_json(&mut self, json: &str) -> Result<(), String> {
        let pairs: Vec<(InputKey, Action)> =
            serde_json::from_str(json).map_err(|e| format!("Invalid bindings JSON: {e}"))?;
        self.bindings = pairs.into_iter().collect();
        Ok(())
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

    #[test]
    fn input_context_stack() {
        let mut stack = InputContextStack::new();
        assert_eq!(*stack.active(), InputContext::Gameplay);

        stack.push(InputContext::Menu);
        assert_eq!(*stack.active(), InputContext::Menu);

        stack.push(InputContext::Editor);
        assert_eq!(*stack.active(), InputContext::Editor);

        assert_eq!(stack.pop(), Some(InputContext::Editor));
        assert_eq!(*stack.active(), InputContext::Menu);

        stack.pop();
        assert_eq!(*stack.active(), InputContext::Gameplay);

        // Can't pop the last one
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn action_map_unbind() {
        let mut map = ActionMap::new();
        map.bind(InputKey::Key("W".into()), "move_forward");
        assert!(map.action_for(&InputKey::Key("W".into())).is_some());

        map.unbind(&InputKey::Key("W".into()));
        assert!(map.action_for(&InputKey::Key("W".into())).is_none());
    }

    #[test]
    fn action_map_json_roundtrip() {
        let mut map = ActionMap::new();
        map.bind(InputKey::Key("W".into()), "move_forward");
        map.bind(InputKey::Key("Space".into()), "jump");

        let json = map.save_to_json();
        assert!(!json.is_empty());

        let mut restored = ActionMap::new();
        restored.load_from_json(&json).unwrap();
        assert_eq!(
            restored.action_for(&InputKey::Key("W".into())),
            Some("move_forward")
        );
        assert_eq!(
            restored.action_for(&InputKey::Key("Space".into())),
            Some("jump")
        );
    }

    #[test]
    fn gamepad_state() {
        let mut gp = GamepadState::default();
        gp.set_axis(0, GamepadAxisType::LeftStickX, 0.75);
        assert!((gp.axis_value(0, &GamepadAxisType::LeftStickX) - 0.75).abs() < 0.01);

        gp.press_button(0, 1);
        assert!(gp.is_button_pressed(0, 1));
        gp.release_button(0, 1);
        assert!(!gp.is_button_pressed(0, 1));
    }
}
