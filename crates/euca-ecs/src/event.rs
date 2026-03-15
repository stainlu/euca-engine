use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Double-buffered event storage for a single event type.
///
/// Events live for 2 frames: current + previous. This allows systems
/// running at different points in the frame to read all events.
struct EventBuffer {
    /// Events added this frame.
    current: Vec<Box<dyn Any + Send + Sync>>,
    /// Events from last frame (cleared on next swap).
    previous: Vec<Box<dyn Any + Send + Sync>>,
}

impl EventBuffer {
    fn new() -> Self {
        Self {
            current: Vec::new(),
            previous: Vec::new(),
        }
    }

    /// Swap buffers: current becomes previous, previous is cleared.
    fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.previous);
        self.current.clear();
    }
}

/// Manages all event types in the world.
pub struct Events {
    buffers: HashMap<TypeId, EventBuffer>,
}

impl Events {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Send an event.
    pub fn send<T: Send + Sync + 'static>(&mut self, event: T) {
        self.buffers
            .entry(TypeId::of::<T>())
            .or_insert_with(EventBuffer::new)
            .current
            .push(Box::new(event));
    }

    /// Read all events of type T from current and previous buffers.
    pub fn read<T: Send + Sync + 'static>(&self) -> impl Iterator<Item = &T> {
        let empty_prev: &[Box<dyn Any + Send + Sync>] = &[];
        let empty_curr: &[Box<dyn Any + Send + Sync>] = &[];

        let (prev, curr) = self
            .buffers
            .get(&TypeId::of::<T>())
            .map(|buf| (buf.previous.as_slice(), buf.current.as_slice()))
            .unwrap_or((empty_prev, empty_curr));

        prev.iter()
            .chain(curr.iter())
            .filter_map(|e| e.downcast_ref::<T>())
    }

    /// Swap all event buffers. Call once per tick.
    pub fn update(&mut self) {
        for buffer in self.buffers.values_mut() {
            buffer.swap();
        }
    }

    /// Clear all events immediately.
    pub fn clear<T: Send + Sync + 'static>(&mut self) {
        if let Some(buffer) = self.buffers.get_mut(&TypeId::of::<T>()) {
            buffer.current.clear();
            buffer.previous.clear();
        }
    }
}

impl Default for Events {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Collision {
        a: u32,
        b: u32,
    }

    #[derive(Debug, PartialEq)]
    struct Damage(f32);

    #[test]
    fn send_and_read() {
        let mut events = Events::new();
        events.send(Collision { a: 1, b: 2 });
        events.send(Collision { a: 3, b: 4 });

        let collisions: Vec<_> = events.read::<Collision>().collect();
        assert_eq!(collisions.len(), 2);
        assert_eq!(collisions[0], &Collision { a: 1, b: 2 });
    }

    #[test]
    fn events_persist_one_frame() {
        let mut events = Events::new();
        events.send(Damage(10.0));

        // After one swap, events move to previous — still readable
        events.update();
        let damages: Vec<_> = events.read::<Damage>().collect();
        assert_eq!(damages.len(), 1);

        // After second swap, previous is cleared
        events.update();
        let damages: Vec<_> = events.read::<Damage>().collect();
        assert_eq!(damages.len(), 0);
    }

    #[test]
    fn current_and_previous_combined() {
        let mut events = Events::new();
        events.send(Damage(10.0)); // frame 0
        events.update();
        events.send(Damage(20.0)); // frame 1

        // Should see both: previous (10) + current (20)
        let damages: Vec<_> = events.read::<Damage>().collect();
        assert_eq!(damages.len(), 2);
    }

    #[test]
    fn different_event_types_independent() {
        let mut events = Events::new();
        events.send(Collision { a: 1, b: 2 });
        events.send(Damage(5.0));

        assert_eq!(events.read::<Collision>().count(), 1);
        assert_eq!(events.read::<Damage>().count(), 1);
    }

    #[test]
    fn read_empty() {
        let events = Events::new();
        assert_eq!(events.read::<Damage>().count(), 0);
    }
}
