//! Animation state machine — states connected by conditional transitions.

use std::collections::HashMap;

/// A condition that must be met for a transition to fire.
#[derive(Clone, Debug)]
pub enum TransitionCondition {
    /// Parameter must be greater than the threshold.
    GreaterThan { param: String, threshold: f32 },
    /// Parameter must be less than the threshold.
    LessThan { param: String, threshold: f32 },
    /// Boolean parameter must be true.
    IsTrue { param: String },
    /// Boolean parameter must be false.
    IsFalse { param: String },
}

impl TransitionCondition {
    /// Evaluate this condition against a parameter set.
    pub fn evaluate(&self, params: &AnimationParameters) -> bool {
        match self {
            TransitionCondition::GreaterThan { param, threshold } => {
                params.get_float(param).is_some_and(|v| v > *threshold)
            }
            TransitionCondition::LessThan { param, threshold } => {
                params.get_float(param).is_some_and(|v| v < *threshold)
            }
            TransitionCondition::IsTrue { param } => {
                params.get_bool(param).unwrap_or(false)
            }
            TransitionCondition::IsFalse { param } => {
                !params.get_bool(param).unwrap_or(true)
            }
        }
    }
}

/// A transition between two states.
#[derive(Clone, Debug)]
pub struct StateTransition {
    /// Source state index. `None` means "any state" (fires from any current state).
    pub from: Option<usize>,
    /// Target state index.
    pub to: usize,
    /// All conditions must be true for the transition to fire.
    pub conditions: Vec<TransitionCondition>,
    /// Duration of the crossfade blend (seconds).
    pub blend_duration: f32,
}

/// A single state in the animation state machine.
#[derive(Clone, Debug)]
pub struct AnimationState {
    /// Human-readable name.
    pub name: String,
    /// Index into the animation library's clip list.
    pub clip_index: usize,
    /// Playback speed multiplier.
    pub speed: f32,
    /// Whether the clip should loop.
    pub looping: bool,
}

/// Typed parameter storage for the state machine.
#[derive(Clone, Debug, Default)]
pub struct AnimationParameters {
    floats: HashMap<String, f32>,
    bools: HashMap<String, bool>,
}

impl AnimationParameters {
    pub fn set_float(&mut self, name: impl Into<String>, value: f32) {
        self.floats.insert(name.into(), value);
    }

    pub fn get_float(&self, name: &str) -> Option<f32> {
        self.floats.get(name).copied()
    }

    pub fn set_bool(&mut self, name: impl Into<String>, value: bool) {
        self.bools.insert(name.into(), value);
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.bools.get(name).copied()
    }
}

/// Tracks the runtime state of a crossfade transition.
#[derive(Clone, Debug)]
pub struct ActiveTransition {
    /// State we are blending away from.
    pub from_state: usize,
    /// Playback time in the source clip when the transition started.
    pub from_time: f32,
    /// Total blend duration (seconds).
    pub blend_duration: f32,
    /// Elapsed time since the transition began (seconds).
    pub elapsed: f32,
}

impl ActiveTransition {
    /// Returns the blend progress in [0.0, 1.0].
    pub fn progress(&self) -> f32 {
        if self.blend_duration <= 0.0 {
            1.0
        } else {
            (self.elapsed / self.blend_duration).clamp(0.0, 1.0)
        }
    }

    /// Returns true when the transition has completed.
    pub fn is_complete(&self) -> bool {
        self.elapsed >= self.blend_duration
    }
}

/// ECS component: an animation state machine instance attached to an entity.
#[derive(Clone, Debug)]
pub struct AnimationStateMachine {
    /// All defined states.
    pub states: Vec<AnimationState>,
    /// All defined transitions (including any-state transitions where `from` is `None`).
    pub transitions: Vec<StateTransition>,
    /// Index of the currently active state.
    pub current_state: usize,
    /// Current playback time in the active state's clip.
    pub current_time: f32,
    /// Parameters that drive transition conditions.
    pub parameters: AnimationParameters,
    /// Active crossfade transition (if any).
    pub transition: Option<ActiveTransition>,
}

impl AnimationStateMachine {
    /// Create a new state machine starting in the given state.
    pub fn new(states: Vec<AnimationState>, transitions: Vec<StateTransition>, initial_state: usize) -> Self {
        Self {
            states,
            transitions,
            current_state: initial_state,
            current_time: 0.0,
            parameters: AnimationParameters::default(),
            transition: None,
        }
    }

    /// Evaluate all transitions and begin a crossfade if one fires.
    ///
    /// Priority: any-state transitions are checked first (they can escape any state),
    /// then state-specific transitions in definition order. The first matching transition wins.
    pub fn evaluate_transitions(&mut self) {
        // Don't start a new transition while one is in progress.
        if self.transition.is_some() {
            return;
        }

        let current = self.current_state;

        // Check any-state transitions first, then state-specific.
        let fired = self
            .transitions
            .iter()
            .filter(|t| {
                // Any-state transition: from is None, but don't transition to self.
                // State-specific transition: from matches current state.
                match t.from {
                    None => t.to != current,
                    Some(from) => from == current,
                }
            })
            .find(|t| t.conditions.iter().all(|c| c.evaluate(&self.parameters)));

        if let Some(t) = fired {
            let to = t.to;
            let blend_duration = t.blend_duration;
            self.transition = Some(ActiveTransition {
                from_state: current,
                from_time: self.current_time,
                blend_duration,
                elapsed: 0.0,
            });
            self.current_state = to;
            self.current_time = 0.0;
        }
    }

    /// Advance playback time and transition progress.
    pub fn advance(&mut self, dt: f32, clip_durations: &[f32]) {
        // Advance current state time.
        let state = &self.states[self.current_state];
        let effective_dt = dt * state.speed;
        self.current_time += effective_dt;

        if let Some(duration) = clip_durations.get(state.clip_index)
            && *duration > 0.0
        {
            if state.looping {
                self.current_time %= *duration;
            } else {
                self.current_time = self.current_time.min(*duration);
            }
        }

        // Advance transition blend.
        if let Some(ref mut transition) = self.transition {
            // Also advance the "from" clip time during the blend.
            if let Some(from_state) = self.states.get(transition.from_state) {
                transition.from_time += dt * from_state.speed;
                if let Some(duration) = clip_durations.get(from_state.clip_index)
                    && *duration > 0.0
                    && from_state.looping
                {
                    transition.from_time %= *duration;
                }
            }

            transition.elapsed += dt;
            if transition.is_complete() {
                self.transition = None;
            }
        }
    }

    /// Returns the clip index and time for the current state.
    pub fn current_clip_and_time(&self) -> (usize, f32) {
        let state = &self.states[self.current_state];
        (state.clip_index, self.current_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idle_run_machine() -> AnimationStateMachine {
        let states = vec![
            AnimationState {
                name: "idle".into(),
                clip_index: 0,
                speed: 1.0,
                looping: true,
            },
            AnimationState {
                name: "run".into(),
                clip_index: 1,
                speed: 1.0,
                looping: true,
            },
        ];
        let transitions = vec![
            StateTransition {
                from: Some(0),
                to: 1,
                conditions: vec![TransitionCondition::GreaterThan {
                    param: "speed".into(),
                    threshold: 0.5,
                }],
                blend_duration: 0.2,
            },
            StateTransition {
                from: Some(1),
                to: 0,
                conditions: vec![TransitionCondition::LessThan {
                    param: "speed".into(),
                    threshold: 0.5,
                }],
                blend_duration: 0.2,
            },
        ];
        AnimationStateMachine::new(states, transitions, 0)
    }

    #[test]
    fn starts_in_initial_state() {
        let sm = idle_run_machine();
        assert_eq!(sm.current_state, 0);
        assert_eq!(sm.current_time, 0.0);
        assert!(sm.transition.is_none());
    }

    #[test]
    fn transition_fires_when_condition_met() {
        let mut sm = idle_run_machine();
        sm.parameters.set_float("speed", 1.0);
        sm.evaluate_transitions();
        assert_eq!(sm.current_state, 1); // moved to "run"
        assert!(sm.transition.is_some());
    }

    #[test]
    fn no_transition_when_condition_not_met() {
        let mut sm = idle_run_machine();
        sm.parameters.set_float("speed", 0.3);
        sm.evaluate_transitions();
        assert_eq!(sm.current_state, 0); // still "idle"
        assert!(sm.transition.is_none());
    }

    #[test]
    fn transition_completes_after_blend_duration() {
        let mut sm = idle_run_machine();
        sm.parameters.set_float("speed", 1.0);
        sm.evaluate_transitions();

        let durations = vec![2.0, 2.0];
        sm.advance(0.1, &durations);
        assert!(sm.transition.is_some());

        sm.advance(0.15, &durations);
        assert!(sm.transition.is_none()); // 0.25 > 0.2 blend duration
    }

    #[test]
    fn no_new_transition_during_active_blend() {
        let mut sm = idle_run_machine();
        sm.parameters.set_float("speed", 1.0);
        sm.evaluate_transitions();
        assert!(sm.transition.is_some());

        // Change params to trigger reverse transition — should NOT fire during blend.
        sm.parameters.set_float("speed", 0.0);
        sm.evaluate_transitions();
        // Still on state 1, transition still in progress.
        assert_eq!(sm.current_state, 1);
    }

    #[test]
    fn any_state_transition() {
        let states = vec![
            AnimationState { name: "idle".into(), clip_index: 0, speed: 1.0, looping: true },
            AnimationState { name: "run".into(), clip_index: 1, speed: 1.0, looping: true },
            AnimationState { name: "death".into(), clip_index: 2, speed: 1.0, looping: false },
        ];
        let transitions = vec![
            // Any-state -> death when "dead" is true.
            StateTransition {
                from: None,
                to: 2,
                conditions: vec![TransitionCondition::IsTrue { param: "dead".into() }],
                blend_duration: 0.1,
            },
        ];
        let mut sm = AnimationStateMachine::new(states, transitions, 0);
        sm.parameters.set_bool("dead", true);
        sm.evaluate_transitions();
        assert_eq!(sm.current_state, 2);
    }

    #[test]
    fn advance_wraps_looping_clip() {
        let mut sm = idle_run_machine();
        let durations = vec![1.0, 1.0];
        sm.advance(1.5, &durations);
        // Looping: 1.5 % 1.0 = 0.5
        assert!((sm.current_time - 0.5).abs() < 0.01);
    }

    #[test]
    fn advance_clamps_non_looping_clip() {
        let states = vec![
            AnimationState { name: "attack".into(), clip_index: 0, speed: 1.0, looping: false },
        ];
        let mut sm = AnimationStateMachine::new(states, vec![], 0);
        let durations = vec![1.0];
        sm.advance(2.0, &durations);
        assert!((sm.current_time - 1.0).abs() < 0.01);
    }

    #[test]
    fn bool_conditions() {
        let cond_true = TransitionCondition::IsTrue { param: "jumping".into() };
        let cond_false = TransitionCondition::IsFalse { param: "jumping".into() };

        let mut params = AnimationParameters::default();
        params.set_bool("jumping", true);

        assert!(cond_true.evaluate(&params));
        assert!(!cond_false.evaluate(&params));
    }
}
