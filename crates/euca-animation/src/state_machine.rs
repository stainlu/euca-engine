//! Animation state machine: states connected by conditional transitions.
//!
//! Each state references an animation clip (or blend tree). Transitions have
//! conditions evaluated against a parameter map, and crossfade durations.

use std::collections::HashMap;

use crate::blend::Crossfade;

/// A named parameter value used by transition conditions.
#[derive(Clone, Debug)]
pub enum ParamValue {
    Float(f32),
    Bool(bool),
}

/// Comparison operators for transition conditions.
#[derive(Clone, Debug)]
pub enum CompareOp {
    Greater,
    Less,
    GreaterOrEqual,
    LessOrEqual,
    Equal,
}

/// A single condition that must be satisfied for a transition to fire.
#[derive(Clone, Debug)]
pub enum TransitionCondition {
    /// Compare a float parameter against a threshold.
    FloatCompare {
        param: String,
        op: CompareOp,
        threshold: f32,
    },
    /// Require a bool parameter to have a specific value.
    BoolEquals { param: String, value: bool },
}

impl TransitionCondition {
    /// Evaluate this condition against the current parameter values.
    pub fn evaluate(&self, params: &HashMap<String, ParamValue>) -> bool {
        match self {
            Self::FloatCompare {
                param,
                op,
                threshold,
            } => {
                let val = match params.get(param) {
                    Some(ParamValue::Float(f)) => *f,
                    _ => return false,
                };
                match op {
                    CompareOp::Greater => val > *threshold,
                    CompareOp::Less => val < *threshold,
                    CompareOp::GreaterOrEqual => val >= *threshold,
                    CompareOp::LessOrEqual => val <= *threshold,
                    CompareOp::Equal => (val - threshold).abs() < f32::EPSILON,
                }
            }
            Self::BoolEquals { param, value } => {
                matches!(params.get(param), Some(ParamValue::Bool(b)) if *b == *value)
            }
        }
    }
}

/// A transition between two states.
#[derive(Clone, Debug)]
pub struct StateTransition {
    /// Target state index.
    pub target: usize,
    /// All conditions must be true for the transition to fire.
    pub conditions: Vec<TransitionCondition>,
    /// Crossfade duration in seconds.
    pub duration: f32,
}

/// An animation state: references a clip and has outgoing transitions.
#[derive(Clone, Debug)]
pub struct AnimState {
    /// Display name for debugging.
    pub name: String,
    /// Index into the clip library.
    pub clip_index: usize,
    /// Playback speed multiplier.
    pub speed: f32,
    /// Whether this state loops.
    pub looping: bool,
    /// Outgoing transitions (checked in order, first match wins).
    pub transitions: Vec<StateTransition>,
}

/// Internal record for an any-state transition.
#[derive(Clone, Debug)]
struct AnyStateTransition {
    transition: StateTransition,
}

/// A parametric animation state machine.
///
/// # Usage
/// 1. Add states with `add_state`.
/// 2. Add transitions with `add_transition` or `add_any_state_transition`.
/// 3. Each frame, set parameters with `set_float` / `set_bool`.
/// 4. Call `update(dt)` to evaluate transitions and advance playback.
#[derive(Clone, Debug)]
pub struct AnimStateMachine {
    states: Vec<AnimState>,
    any_state_transitions: Vec<AnyStateTransition>,
    params: HashMap<String, ParamValue>,
    /// Current active state index.
    current_state: usize,
    /// Current playback time within the active state's clip.
    pub current_time: f32,
    /// Active crossfade (if transitioning).
    active_crossfade: Option<CrossfadeState>,
}

/// Tracks an in-progress crossfade between states.
#[derive(Clone, Debug)]
struct CrossfadeState {
    /// State we're transitioning from.
    from_state: usize,
    from_time: f32,
    /// The crossfade timing.
    crossfade: Crossfade,
}

impl AnimStateMachine {
    /// Create a new state machine starting in the given state.
    pub fn new(initial_state: usize) -> Self {
        Self {
            states: Vec::new(),
            any_state_transitions: Vec::new(),
            params: HashMap::new(),
            current_state: initial_state,
            current_time: 0.0,
            active_crossfade: None,
        }
    }

    /// Add a state and return its index.
    pub fn add_state(&mut self, name: impl Into<String>, clip_index: usize) -> usize {
        let idx = self.states.len();
        self.states.push(AnimState {
            name: name.into(),
            clip_index,
            speed: 1.0,
            looping: true,
            transitions: Vec::new(),
        });
        idx
    }

    /// Get a mutable reference to a state for further configuration.
    pub fn state_mut(&mut self, index: usize) -> Option<&mut AnimState> {
        self.states.get_mut(index)
    }

    /// Add a transition from `from_state` to `target_state`.
    pub fn add_transition(
        &mut self,
        from_state: usize,
        target_state: usize,
        conditions: Vec<TransitionCondition>,
        duration: f32,
    ) {
        if let Some(state) = self.states.get_mut(from_state) {
            state.transitions.push(StateTransition {
                target: target_state,
                conditions,
                duration,
            });
        }
    }

    /// Add a transition that can fire from any state.
    pub fn add_any_state_transition(
        &mut self,
        target_state: usize,
        conditions: Vec<TransitionCondition>,
        duration: f32,
    ) {
        self.any_state_transitions.push(AnyStateTransition {
            transition: StateTransition {
                target: target_state,
                conditions,
                duration,
            },
        });
    }

    /// Set a float parameter.
    pub fn set_float(&mut self, name: impl Into<String>, value: f32) {
        self.params.insert(name.into(), ParamValue::Float(value));
    }

    /// Set a bool parameter.
    pub fn set_bool(&mut self, name: impl Into<String>, value: bool) {
        self.params.insert(name.into(), ParamValue::Bool(value));
    }

    /// Get a float parameter value.
    pub fn get_float(&self, name: &str) -> Option<f32> {
        match self.params.get(name) {
            Some(ParamValue::Float(f)) => Some(*f),
            _ => None,
        }
    }

    /// Get a bool parameter value.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.params.get(name) {
            Some(ParamValue::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// The currently active state index.
    pub fn current_state(&self) -> usize {
        self.current_state
    }

    /// The clip index for the current state.
    pub fn current_clip_index(&self) -> Option<usize> {
        self.states.get(self.current_state).map(|s| s.clip_index)
    }

    /// Whether a crossfade is currently in progress.
    pub fn is_transitioning(&self) -> bool {
        self.active_crossfade.is_some()
    }

    /// Returns the current crossfade info if transitioning:
    /// `(from_clip_index, from_time, outgoing_weight, incoming_weight)`.
    ///
    /// The `from_clip_index` is the clip index of the state we're transitioning from.
    pub fn crossfade_info(&self) -> Option<(usize, f32, f32, f32)> {
        self.active_crossfade.as_ref().and_then(|cf| {
            let from_clip = self.states.get(cf.from_state)?.clip_index;
            Some((
                from_clip,
                cf.from_time,
                cf.crossfade.outgoing_weight(),
                cf.crossfade.incoming_weight(),
            ))
        })
    }

    /// Advance time and evaluate transitions. Call once per frame.
    ///
    /// Returns `true` if a transition occurred this frame.
    pub fn update(&mut self, dt: f32, clip_durations: &[f32]) -> bool {
        let mut transitioned = false;

        // Advance crossfade if active
        let mut just_completed_transition = false;
        if let Some(ref mut cf_state) = self.active_crossfade
            && cf_state.crossfade.advance(dt)
        {
            // Crossfade complete -- fully in the new state.
            // Set flag to prevent a new transition from firing on the
            // same frame the previous one finishes (double-transition).
            self.active_crossfade = None;
            just_completed_transition = true;
        }

        // Check transitions only if not already mid-crossfade and no
        // transition just completed this frame.
        if self.active_crossfade.is_none() && !just_completed_transition {
            // Check any-state transitions first (higher priority)
            let any_target = self
                .any_state_transitions
                .iter()
                .find(|ast| {
                    ast.transition.target != self.current_state
                        && ast
                            .transition
                            .conditions
                            .iter()
                            .all(|c| c.evaluate(&self.params))
                })
                .map(|ast| (ast.transition.target, ast.transition.duration));

            if let Some((target, duration)) = any_target {
                self.begin_transition(target, duration);
                transitioned = true;
            } else if let Some(state) = self.states.get(self.current_state) {
                // Check state-specific transitions
                let state_target = state
                    .transitions
                    .iter()
                    .find(|t| t.conditions.iter().all(|c| c.evaluate(&self.params)))
                    .map(|t| (t.target, t.duration));

                if let Some((target, duration)) = state_target {
                    self.begin_transition(target, duration);
                    transitioned = true;
                }
            }
        }

        // Advance playback time
        if let Some(state) = self.states.get(self.current_state) {
            let speed = state.speed;
            let looping = state.looping;
            self.current_time += dt * speed;

            if let Some(&clip_dur) = clip_durations.get(state.clip_index)
                && clip_dur > 0.0
            {
                if looping {
                    self.current_time %= clip_dur;
                } else {
                    self.current_time = self.current_time.min(clip_dur);
                }
            }
        }

        transitioned
    }

    /// Begin a transition to a new state.
    fn begin_transition(&mut self, target: usize, duration: f32) {
        self.active_crossfade = Some(CrossfadeState {
            from_state: self.current_state,
            from_time: self.current_time,
            crossfade: Crossfade::new(duration),
        });
        self.current_state = target;
        self.current_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_walk_run_sm() -> AnimStateMachine {
        let mut sm = AnimStateMachine::new(0);
        sm.add_state("idle", 0);
        sm.add_state("walk", 1);
        sm.add_state("run", 2);

        sm.add_transition(
            0,
            1,
            vec![TransitionCondition::FloatCompare {
                param: "speed".into(),
                op: CompareOp::Greater,
                threshold: 0.1,
            }],
            0.2,
        );
        sm.add_transition(
            1,
            2,
            vec![TransitionCondition::FloatCompare {
                param: "speed".into(),
                op: CompareOp::Greater,
                threshold: 0.6,
            }],
            0.3,
        );
        sm.add_transition(
            1,
            0,
            vec![TransitionCondition::FloatCompare {
                param: "speed".into(),
                op: CompareOp::LessOrEqual,
                threshold: 0.1,
            }],
            0.2,
        );

        sm.set_float("speed", 0.0);
        sm
    }

    #[test]
    fn starts_in_initial_state() {
        let sm = setup_walk_run_sm();
        assert_eq!(sm.current_state(), 0);
    }

    #[test]
    fn transitions_on_condition() {
        let mut sm = setup_walk_run_sm();
        let durations = [1.0, 1.0, 1.0];

        sm.set_float("speed", 0.5);
        let transitioned = sm.update(0.016, &durations);
        assert!(transitioned);
        assert_eq!(sm.current_state(), 1);
        assert!(sm.is_transitioning());
    }

    #[test]
    fn no_transition_when_condition_not_met() {
        let mut sm = setup_walk_run_sm();
        let durations = [1.0, 1.0, 1.0];

        sm.set_float("speed", 0.0);
        let transitioned = sm.update(0.016, &durations);
        assert!(!transitioned);
        assert_eq!(sm.current_state(), 0);
    }

    #[test]
    fn crossfade_completes() {
        let mut sm = setup_walk_run_sm();
        let durations = [1.0, 1.0, 1.0];

        sm.set_float("speed", 0.5);
        sm.update(0.016, &durations);
        assert!(sm.is_transitioning());

        for _ in 0..20 {
            sm.update(0.016, &durations);
        }
        assert!(!sm.is_transitioning());
    }

    #[test]
    fn any_state_transition() {
        let mut sm = AnimStateMachine::new(0);
        sm.add_state("normal", 0);
        sm.add_state("flinch", 1);

        sm.add_any_state_transition(
            1,
            vec![TransitionCondition::BoolEquals {
                param: "hit".into(),
                value: true,
            }],
            0.1,
        );

        let durations = [1.0, 0.5];
        sm.set_bool("hit", true);
        assert!(sm.update(0.016, &durations));
        assert_eq!(sm.current_state(), 1);
    }

    #[test]
    fn bool_condition() {
        let mut params = HashMap::new();
        params.insert("grounded".into(), ParamValue::Bool(true));

        let cond = TransitionCondition::BoolEquals {
            param: "grounded".into(),
            value: true,
        };
        assert!(cond.evaluate(&params));

        let cond_false = TransitionCondition::BoolEquals {
            param: "grounded".into(),
            value: false,
        };
        assert!(!cond_false.evaluate(&params));
    }

    #[test]
    fn no_double_transition_on_crossfade_complete() {
        // States: A(0) -> B(1), with an any-state transition to C(2).
        // When the A->B crossfade completes, the machine should land in B
        // and NOT immediately fire the any-state transition to C on the
        // same frame. This is the "double-transition" bug.
        let mut sm = AnimStateMachine::new(0);
        sm.add_state("A", 0);
        sm.add_state("B", 1);
        sm.add_state("C", 2);

        // A -> B transition triggered by "go" bool
        sm.add_transition(
            0,
            1,
            vec![TransitionCondition::BoolEquals {
                param: "go".into(),
                value: true,
            }],
            0.1, // 100ms crossfade
        );

        // Any-state -> C triggered by "alert" bool.
        // This will be set true DURING the crossfade, so it's ready to
        // fire the moment the crossfade completes.
        sm.add_any_state_transition(
            2,
            vec![TransitionCondition::BoolEquals {
                param: "alert".into(),
                value: true,
            }],
            0.1,
        );

        let durations = [1.0, 1.0, 1.0];

        // Trigger A -> B
        sm.set_bool("go", true);
        sm.update(0.016, &durations);
        assert_eq!(sm.current_state(), 1, "Should begin transitioning to B");
        assert!(sm.is_transitioning());

        // While mid-crossfade, enable the any-state condition.
        sm.set_bool("alert", true);

        // Advance frame-by-frame until the crossfade completes.
        // On the exact frame it completes, the any-state transition to C
        // should NOT fire (that would be a double-transition).
        for _ in 0..20 {
            sm.update(0.016, &durations);
            if !sm.is_transitioning() {
                // Crossfade just completed this frame.
                // The machine should be in B, NOT C.
                assert_eq!(
                    sm.current_state(),
                    1,
                    "Should be in state B, NOT C (no double-transition on completion frame)"
                );
                return;
            }
        }
        panic!("Crossfade did not complete within expected time");
    }

    #[test]
    fn time_advances_with_speed() {
        let mut sm = AnimStateMachine::new(0);
        let idx = sm.add_state("fast", 0);
        sm.state_mut(idx).unwrap().speed = 2.0;

        let durations = [10.0];
        sm.update(1.0, &durations);
        assert!((sm.current_time - 2.0).abs() < 1e-5);
    }

    #[test]
    fn time_loops() {
        let mut sm = AnimStateMachine::new(0);
        sm.add_state("loop", 0);

        let durations = [1.0];
        sm.update(1.5, &durations);
        assert!((sm.current_time - 0.5).abs() < 1e-5);
    }
}
