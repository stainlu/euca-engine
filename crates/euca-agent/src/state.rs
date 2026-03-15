use euca_ecs::{Schedule, World};
use std::sync::{Arc, Mutex};

/// Thread-safe wrapper around the ECS world and schedule.
///
/// Used to share the simulation state between the HTTP server and the main loop.
/// For headless mode, the HTTP handlers own all mutation via this shared state.
#[derive(Clone)]
pub struct SharedWorld {
    inner: Arc<Mutex<WorldState>>,
}

pub struct WorldState {
    pub world: World,
    pub schedule: Schedule,
}

impl SharedWorld {
    pub fn new(world: World, schedule: Schedule) -> Self {
        Self {
            inner: Arc::new(Mutex::new(WorldState { world, schedule })),
        }
    }

    /// Run a closure with exclusive access to the world and schedule.
    pub fn with<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut World, &mut Schedule) -> R,
    {
        let mut state = self.inner.lock().unwrap();
        let WorldState {
            ref mut world,
            ref mut schedule,
        } = *state;
        f(world, schedule)
    }

    /// Run a closure with read-only access to the world.
    pub fn with_world<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&World) -> R,
    {
        let state = self.inner.lock().unwrap();
        f(&state.world)
    }
}
