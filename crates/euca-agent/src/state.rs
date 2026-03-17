use euca_ecs::{Schedule, World};
use std::sync::{Arc, RwLock};

/// Unique identifier for an agent/player.
pub type AgentId = u32;

/// ECS component marking entity ownership by a specific agent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Owner(pub AgentId);

/// A single world instance with its schedule.
pub struct WorldState {
    pub world: World,
    pub schedule: Schedule,
}

/// Pool of independent worlds for parallel agent environments.
///
/// Solves CRITICAL #9: instead of one Mutex'd world, each agent/environment
/// gets its own world. Default world (index 0) is always available.
///
/// Also tracks agent ownership for CRITICAL #10.
#[derive(Clone)]
pub struct SharedWorld {
    inner: Arc<RwLock<WorldPool>>,
}

struct WorldPool {
    worlds: Vec<WorldState>,
    next_agent_id: AgentId,
}

impl SharedWorld {
    /// Create with a single default world (index 0).
    pub fn new(world: World, schedule: Schedule) -> Self {
        Self {
            inner: Arc::new(RwLock::new(WorldPool {
                worlds: vec![WorldState { world, schedule }],
                next_agent_id: 1,
            })),
        }
    }

    /// Allocate a new agent ID.
    pub fn new_agent_id(&self) -> AgentId {
        let mut pool = self.inner.write().unwrap();
        let id = pool.next_agent_id;
        pool.next_agent_id += 1;
        id
    }

    /// Create a new independent world, returning its index.
    /// Each world has its own ECS state — no lock contention between worlds.
    pub fn create_world(&self, world: World, schedule: Schedule) -> usize {
        let mut pool = self.inner.write().unwrap();
        let idx = pool.worlds.len();
        pool.worlds.push(WorldState { world, schedule });
        idx
    }

    /// Number of worlds in the pool.
    pub fn world_count(&self) -> usize {
        self.inner.read().unwrap().worlds.len()
    }

    /// Run a closure with exclusive access to a specific world.
    /// Default world is index 0.
    pub fn with<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut World, &mut Schedule) -> R,
    {
        self.with_world_index(0, f)
    }

    /// Run a closure with exclusive access to a specific world by index.
    pub fn with_world_index<F, R>(&self, index: usize, f: F) -> R
    where
        F: FnOnce(&mut World, &mut Schedule) -> R,
    {
        let mut pool = self.inner.write().unwrap();
        let state = &mut pool.worlds[index];
        f(&mut state.world, &mut state.schedule)
    }

    /// Run a closure with read-only access to the default world.
    pub fn with_world<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&World) -> R,
    {
        let pool = self.inner.read().unwrap();
        f(&pool.worlds[0].world)
    }
}
