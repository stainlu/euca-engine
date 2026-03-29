//! Thread-safe shared world access for concurrent editor + server use.
//!
//! Provides `SharedWorld` — an `Arc<RwLock<...>>` wrapper around a pool of
//! independent ECS worlds. Both the editor (main thread) and the agent HTTP
//! server (tokio threads) access the same world through this type.

use crate::{Schedule, World};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A single world instance with its schedule.
pub struct WorldState {
    pub world: World,
    pub schedule: Schedule,
}

/// Pool of independent worlds for parallel agent environments.
///
/// Default world (index 0) is always available. Additional worlds can be
/// created for isolated agent environments (e.g. RL training).
pub struct WorldPool {
    pub worlds: Vec<WorldState>,
    next_id: u32,
}

impl WorldPool {
    /// Get mutable reference to the default world (index 0).
    pub fn world(&mut self) -> &mut World {
        &mut self.worlds[0].world
    }

    /// Get shared reference to the default world (index 0).
    pub fn world_ref(&self) -> &World {
        &self.worlds[0].world
    }

    /// Get mutable reference to the default schedule (index 0).
    pub fn schedule(&mut self) -> &mut Schedule {
        &mut self.worlds[0].schedule
    }

    /// Allocate a new sequential ID.
    pub fn next_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// Thread-safe handle to a pool of ECS worlds.
///
/// Clone is cheap (Arc clone). Multiple systems can hold handles to the same
/// pool and access it through read/write locks.
#[derive(Clone)]
pub struct SharedWorld {
    inner: Arc<RwLock<WorldPool>>,
}

impl SharedWorld {
    /// Create with a single default world (index 0).
    pub fn new(world: World, schedule: Schedule) -> Self {
        Self {
            inner: Arc::new(RwLock::new(WorldPool {
                worlds: vec![WorldState { world, schedule }],
                next_id: 1,
            })),
        }
    }

    /// Allocate a new sequential ID (for agents, sessions, etc.).
    pub fn next_id(&self) -> u32 {
        crate::lock_util::write_or_recover(&self.inner, "SharedWorld::next_id").next_id()
    }

    /// Create a new independent world, returning its index.
    pub fn create_world(&self, world: World, schedule: Schedule) -> usize {
        let mut pool = crate::lock_util::write_or_recover(&self.inner, "SharedWorld::create_world");
        let idx = pool.worlds.len();
        pool.worlds.push(WorldState { world, schedule });
        idx
    }

    /// Number of worlds in the pool.
    pub fn world_count(&self) -> usize {
        crate::lock_util::read_or_recover(&self.inner, "SharedWorld::world_count")
            .worlds
            .len()
    }

    /// Run a closure with exclusive access to the default world + schedule.
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
        let mut pool =
            crate::lock_util::write_or_recover(&self.inner, "SharedWorld::with_world_index");
        let state = &mut pool.worlds[index];
        f(&mut state.world, &mut state.schedule)
    }

    /// Run a closure with read-only access to the default world.
    pub fn with_world<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&World) -> R,
    {
        let pool = crate::lock_util::read_or_recover(&self.inner, "SharedWorld::with_world");
        f(&pool.worlds[0].world)
    }

    /// Acquire a write lock on the world pool.
    ///
    /// Returns a guard that provides `&mut World` access via `world()`.
    /// The lock is released when the guard is dropped.
    ///
    /// Use this when you need to hold the world lock across multiple
    /// operations that also access other state (e.g. editor fields).
    pub fn lock(&self) -> WorldWriteGuard<'_> {
        WorldWriteGuard {
            guard: crate::lock_util::write_or_recover(&self.inner, "SharedWorld::lock"),
        }
    }

    /// Acquire a read lock on the world pool.
    pub fn lock_read(&self) -> WorldReadGuard<'_> {
        WorldReadGuard {
            guard: crate::lock_util::read_or_recover(&self.inner, "SharedWorld::lock_read"),
        }
    }
}

/// RAII write guard for the world pool.
///
/// Provides `&mut World` access while holding the write lock.
/// Other fields of the owning struct remain accessible (disjoint borrows).
pub struct WorldWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, WorldPool>,
}

impl WorldWriteGuard<'_> {
    /// Get mutable reference to the default world.
    pub fn world(&mut self) -> &mut World {
        &mut self.guard.worlds[0].world
    }

    /// Get mutable reference to the default schedule.
    pub fn schedule(&mut self) -> &mut Schedule {
        &mut self.guard.worlds[0].schedule
    }

    /// Get mutable reference to a world by index.
    pub fn world_at(&mut self, index: usize) -> &mut World {
        &mut self.guard.worlds[index].world
    }

    /// Allocate a new sequential ID.
    pub fn next_id(&mut self) -> u32 {
        self.guard.next_id()
    }
}

/// RAII read guard for the world pool.
pub struct WorldReadGuard<'a> {
    guard: RwLockReadGuard<'a, WorldPool>,
}

impl WorldReadGuard<'_> {
    /// Get shared reference to the default world.
    pub fn world(&self) -> &World {
        &self.guard.worlds[0].world
    }

    /// Get shared reference to a world by index.
    pub fn world_at(&self, index: usize) -> &World {
        &self.guard.worlds[index].world
    }
}
