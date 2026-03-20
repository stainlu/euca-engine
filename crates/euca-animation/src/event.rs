//! Animation events — time-stamped callbacks within clips, emitted as ECS events.

use euca_ecs::Entity;

/// A named event marker placed at a specific time within an animation clip.
#[derive(Clone, Debug)]
pub struct AnimationEventMarker {
    /// Time offset within the clip (seconds).
    pub time: f32,
    /// Event name (e.g., "footstep_left", "vfx_muzzle_flash", "damage_start").
    pub name: String,
}

/// Configuration: event markers attached to a specific clip.
#[derive(Clone, Debug, Default)]
pub struct ClipEventMarkers {
    /// Sorted list of event markers by time.
    markers: Vec<AnimationEventMarker>,
}

impl ClipEventMarkers {
    /// Create a new set of markers. They will be sorted by time automatically.
    pub fn new(mut markers: Vec<AnimationEventMarker>) -> Self {
        markers.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        Self { markers }
    }

    /// Returns the sorted markers.
    pub fn markers(&self) -> &[AnimationEventMarker] {
        &self.markers
    }

    /// Collect events that should fire when playback advances from `prev_time` to `curr_time`.
    ///
    /// For looping clips that wrap around, `wrapped` should be true and the clip `duration`
    /// must be provided.
    pub fn collect_fired(
        &self,
        prev_time: f32,
        curr_time: f32,
        wrapped: bool,
        duration: f32,
    ) -> Vec<&AnimationEventMarker> {
        if wrapped && duration > 0.0 {
            // Wrapped: events in [prev_time..duration) + [0..curr_time]
            self.markers
                .iter()
                .filter(|m| (m.time >= prev_time && m.time < duration) || m.time < curr_time)
                .collect()
        } else {
            // Normal: events in [prev_time..curr_time)
            self.markers
                .iter()
                .filter(|m| m.time >= prev_time && m.time < curr_time)
                .collect()
        }
    }
}

/// ECS event emitted when an animation event marker is reached during playback.
#[derive(Clone, Debug)]
pub struct AnimationEvent {
    /// The entity whose animation triggered this event.
    pub entity: Entity,
    /// The clip index that contains the event marker.
    pub clip_index: usize,
    /// The name of the event marker.
    pub name: String,
    /// The time within the clip at which the event was placed.
    pub time: f32,
}

/// World resource: stores event markers for all clips.
///
/// Keyed by clip index. Clips without markers simply have no entry.
#[derive(Clone, Debug, Default)]
pub struct AnimationEventLibrary {
    markers: Vec<Option<ClipEventMarkers>>,
}

impl AnimationEventLibrary {
    /// Register event markers for a clip index. Overwrites any existing markers.
    pub fn set_markers(&mut self, clip_index: usize, markers: ClipEventMarkers) {
        if clip_index >= self.markers.len() {
            self.markers.resize(clip_index + 1, None);
        }
        self.markers[clip_index] = Some(markers);
    }

    /// Get event markers for a clip, if any.
    pub fn get_markers(&self, clip_index: usize) -> Option<&ClipEventMarkers> {
        self.markers.get(clip_index).and_then(|m| m.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn footstep_markers() -> ClipEventMarkers {
        ClipEventMarkers::new(vec![
            AnimationEventMarker { time: 0.25, name: "footstep_left".into() },
            AnimationEventMarker { time: 0.75, name: "footstep_right".into() },
        ])
    }

    #[test]
    fn fires_events_in_time_range() {
        let markers = footstep_markers();
        let fired = markers.collect_fired(0.0, 0.5, false, 1.0);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].name, "footstep_left");
    }

    #[test]
    fn no_events_outside_range() {
        let markers = footstep_markers();
        let fired = markers.collect_fired(0.3, 0.7, false, 1.0);
        assert!(fired.is_empty());
    }

    #[test]
    fn fires_multiple_events_in_large_dt() {
        let markers = footstep_markers();
        let fired = markers.collect_fired(0.0, 1.0, false, 1.0);
        assert_eq!(fired.len(), 2);
    }

    #[test]
    fn wrapped_playback_fires_events_across_boundary() {
        let markers = footstep_markers();
        // Wrapped from 0.8 to 0.3 — should fire footstep_left at 0.25.
        let fired = markers.collect_fired(0.8, 0.3, true, 1.0);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].name, "footstep_left");
    }

    #[test]
    fn empty_markers_fires_nothing() {
        let markers = ClipEventMarkers::default();
        let fired = markers.collect_fired(0.0, 1.0, false, 1.0);
        assert!(fired.is_empty());
    }

    #[test]
    fn event_library_set_and_get() {
        let mut lib = AnimationEventLibrary::default();
        lib.set_markers(3, footstep_markers());
        assert!(lib.get_markers(0).is_none());
        assert!(lib.get_markers(3).is_some());
        assert_eq!(lib.get_markers(3).unwrap().markers().len(), 2);
    }
}
