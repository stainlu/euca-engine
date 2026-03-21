//! Animation events: time-stamped callbacks within clips.
//!
//! Events fire at specific times during animation playback (e.g. footstep
//! sounds, VFX spawns, damage window start/end).

use std::collections::HashMap;

/// An event embedded in an animation clip at a specific time.
#[derive(Clone, Debug)]
pub struct AnimationEvent {
    /// Time within the clip (seconds) when this event fires.
    pub time: f32,
    /// Event name (e.g. "footstep_left", "vfx_slash", "damage_start").
    pub name: String,
    /// Optional payload -- arbitrary key-value data for the event handler.
    pub payload: HashMap<String, EventValue>,
}

/// A value that can be attached to an animation event.
#[derive(Clone, Debug)]
pub enum EventValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    String(String),
}

/// Collection of events for a single animation clip, sorted by time.
#[derive(Clone, Debug, Default)]
pub struct ClipEvents {
    events: Vec<AnimationEvent>,
}

impl ClipEvents {
    /// Create a new event collection. Events are sorted by time on construction.
    pub fn new(mut events: Vec<AnimationEvent>) -> Self {
        events.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { events }
    }

    /// Add a simple named event at the given time.
    pub fn add(&mut self, time: f32, name: impl Into<String>) {
        self.insert_sorted(AnimationEvent {
            time,
            name: name.into(),
            payload: HashMap::new(),
        });
    }

    /// Add an event with payload.
    pub fn add_with_payload(
        &mut self,
        time: f32,
        name: impl Into<String>,
        payload: HashMap<String, EventValue>,
    ) {
        self.insert_sorted(AnimationEvent {
            time,
            name: name.into(),
            payload,
        });
    }

    /// Insert an event in sorted order using binary search.
    fn insert_sorted(&mut self, event: AnimationEvent) {
        let pos = self.events.partition_point(|e| e.time < event.time);
        self.events.insert(pos, event);
    }

    /// Query which events fired between `prev_time` and `curr_time`.
    ///
    /// Handles looping: if `curr_time < prev_time`, events from `prev_time`
    /// to the end AND from the start to `curr_time` are returned.
    pub fn query(&self, prev_time: f32, curr_time: f32) -> Vec<&AnimationEvent> {
        if self.events.is_empty() {
            return Vec::new();
        }

        if curr_time >= prev_time {
            // Normal playback: events in (prev_time, curr_time]
            self.events
                .iter()
                .filter(|e| e.time > prev_time && e.time <= curr_time)
                .collect()
        } else {
            // Looped: events in (prev_time, end] + [0, curr_time]
            self.events
                .iter()
                .filter(|e| e.time > prev_time || e.time <= curr_time)
                .collect()
        }
    }
}

/// Resource: maps clip indices to their event data.
#[derive(Clone, Debug, Default)]
pub struct AnimationEventLibrary {
    clip_events: HashMap<usize, ClipEvents>,
}

impl AnimationEventLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the events for a clip.
    pub fn set_clip_events(&mut self, clip_index: usize, events: ClipEvents) {
        self.clip_events.insert(clip_index, events);
    }

    /// Get the events for a clip.
    pub fn get_clip_events(&self, clip_index: usize) -> Option<&ClipEvents> {
        self.clip_events.get(&clip_index)
    }
}

/// Collects fired events for a single entity during one frame.
#[derive(Clone, Debug, Default)]
pub struct FiredAnimationEvents {
    pub events: Vec<FiredEvent>,
}

/// A single fired event with context.
#[derive(Clone, Debug)]
pub struct FiredEvent {
    pub name: String,
    pub payload: HashMap<String, EventValue>,
    /// The clip index that fired this event.
    pub clip_index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_sorted_on_construction() {
        let events = ClipEvents::new(vec![
            AnimationEvent {
                time: 0.5,
                name: "b".into(),
                payload: HashMap::new(),
            },
            AnimationEvent {
                time: 0.1,
                name: "a".into(),
                payload: HashMap::new(),
            },
        ]);
        assert_eq!(events.events[0].name, "a");
        assert_eq!(events.events[1].name, "b");
    }

    #[test]
    fn query_normal_playback() {
        let mut events = ClipEvents::default();
        events.add(0.1, "step_left");
        events.add(0.3, "step_right");
        events.add(0.6, "step_left");

        let fired = events.query(0.0, 0.2);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].name, "step_left");

        let fired = events.query(0.0, 0.4);
        assert_eq!(fired.len(), 2);
    }

    #[test]
    fn query_no_events_in_range() {
        let mut events = ClipEvents::default();
        events.add(0.5, "hit");

        let fired = events.query(0.0, 0.3);
        assert!(fired.is_empty());
    }

    #[test]
    fn query_loop_wrap() {
        let mut events = ClipEvents::default();
        events.add(0.1, "start");
        events.add(0.9, "end");

        // prev_time=0.8, curr_time=0.2 means we looped
        let fired = events.query(0.8, 0.2);
        assert_eq!(fired.len(), 2);
    }

    #[test]
    fn query_exact_boundary() {
        let mut events = ClipEvents::default();
        events.add(0.5, "hit");

        // Event at exactly curr_time should fire
        let fired = events.query(0.0, 0.5);
        assert_eq!(fired.len(), 1);

        // Event at exactly prev_time should NOT fire (already fired last frame)
        let fired = events.query(0.5, 0.8);
        assert!(fired.is_empty());
    }

    #[test]
    fn event_library() {
        let mut lib = AnimationEventLibrary::new();
        let mut events = ClipEvents::default();
        events.add(0.2, "footstep");
        lib.set_clip_events(0, events);

        assert!(lib.get_clip_events(0).is_some());
        assert!(lib.get_clip_events(1).is_none());
    }
}
