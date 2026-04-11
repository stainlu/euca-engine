//! Thread-safe shared world access for concurrent editor + server use.
//!
//! Provides `SharedWorld` — an `Arc<RwLock<...>>` wrapper around an ECS
//! world pool. Both the editor (main thread) and the agent HTTP server
//! (tokio threads) access the same world through this type.
//!
//! # Main world + forks
//!
//! A [`WorldPool`] always owns one **main** world. Agents can create
//! named **forks** — independent deep copies of the main world — for
//! counterfactual reasoning ("what if I did X?"). Forks share the
//! single [`Schedule`] owned by the pool, so every fork runs the same
//! systems as the main world when stepped.
//!
//! Sharing the schedule (rather than cloning it) keeps fork creation
//! cheap and avoids requiring every `System` trait object to be
//! cloneable. Because the schedule is locked per-call alongside the
//! world, only one world (main or any single fork) can tick at a time.
//! Agents call `/fork/step` synchronously, so this is a non-issue.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{Schedule, World};

/// A pool of ECS worlds backing a [`SharedWorld`].
///
/// Owns one main world plus an arbitrary number of agent-created forks,
/// keyed by agent-chosen ids (strings). All worlds share a single
/// [`Schedule`] so that stepping a fork runs the same systems as the
/// main world.
pub struct WorldPool {
    main: World,
    schedule: Schedule,
    forks: HashMap<String, World>,
    next_id: u32,
}

impl WorldPool {
    /// Get mutable reference to the main world.
    pub fn world(&mut self) -> &mut World {
        &mut self.main
    }

    /// Get shared reference to the main world.
    pub fn world_ref(&self) -> &World {
        &self.main
    }

    /// Get mutable reference to the shared schedule. Every world in the
    /// pool (main and forks) runs against this schedule when stepped.
    pub fn schedule(&mut self) -> &mut Schedule {
        &mut self.schedule
    }

    /// Allocate a new sequential ID.
    pub fn next_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Iterate the ids of all currently active forks.
    pub fn fork_ids(&self) -> impl Iterator<Item = &str> {
        self.forks.keys().map(|s| s.as_str())
    }

    /// Number of active forks.
    pub fn fork_count(&self) -> usize {
        self.forks.len()
    }

    /// Whether a fork with the given id exists.
    pub fn fork_exists(&self, fork_id: &str) -> bool {
        self.forks.contains_key(fork_id)
    }

    /// Mutable access to a specific fork's world.
    pub fn fork_mut(&mut self, fork_id: &str) -> Option<&mut World> {
        self.forks.get_mut(fork_id)
    }

    /// Shared access to a specific fork's world.
    pub fn fork_ref(&self, fork_id: &str) -> Option<&World> {
        self.forks.get(fork_id)
    }
}

/// Thread-safe handle to a pool of ECS worlds.
///
/// Clone is cheap (Arc clone). Multiple systems can hold handles to the
/// same pool and access it through read/write locks.
#[derive(Clone)]
pub struct SharedWorld {
    inner: Arc<RwLock<WorldPool>>,
}

impl SharedWorld {
    /// Create with a single main world and its shared schedule.
    pub fn new(world: World, schedule: Schedule) -> Self {
        Self {
            inner: Arc::new(RwLock::new(WorldPool {
                main: world,
                schedule,
                forks: HashMap::new(),
                next_id: 1,
            })),
        }
    }

    /// Allocate a new sequential ID (for agents, sessions, etc.).
    pub fn next_id(&self) -> u32 {
        crate::lock_util::write_or_recover(&self.inner, "SharedWorld::next_id").next_id()
    }

    /// Run a closure with exclusive access to the main world + schedule.
    pub fn with<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut World, &mut Schedule) -> R,
    {
        let mut pool = crate::lock_util::write_or_recover(&self.inner, "SharedWorld::with");
        let WorldPool {
            main, schedule, ..
        } = &mut *pool;
        f(main, schedule)
    }

    /// Run a closure with read-only access to the main world.
    pub fn with_world<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&World) -> R,
    {
        let pool = crate::lock_util::read_or_recover(&self.inner, "SharedWorld::with_world");
        f(&pool.main)
    }

    /// Acquire a write lock on the world pool.
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

    // ── Fork API ──

    /// Create a new fork by deep-cloning the main world. The fork starts
    /// with the same entities, components, resources, events, and tick
    /// as the main world at the moment of forking. Running the fork's
    /// schedule afterwards only affects the fork — the main world is
    /// untouched.
    ///
    /// Returns an error if a fork with the same id already exists.
    pub fn fork(&self, fork_id: impl Into<String>) -> Result<(), ForkError> {
        let id = fork_id.into();
        let mut pool = crate::lock_util::write_or_recover(&self.inner, "SharedWorld::fork");
        if pool.forks.contains_key(&id) {
            return Err(ForkError::AlreadyExists(id));
        }
        let cloned = pool.main.clone();
        pool.forks.insert(id, cloned);
        Ok(())
    }

    /// Delete a fork. Returns `true` if the fork existed, `false` otherwise.
    pub fn delete_fork(&self, fork_id: &str) -> bool {
        let mut pool = crate::lock_util::write_or_recover(&self.inner, "SharedWorld::delete_fork");
        pool.forks.remove(fork_id).is_some()
    }

    /// List all active fork ids.
    pub fn list_forks(&self) -> Vec<String> {
        let pool = crate::lock_util::read_or_recover(&self.inner, "SharedWorld::list_forks");
        pool.forks.keys().cloned().collect()
    }

    /// Check whether a fork with the given id exists.
    pub fn fork_exists(&self, fork_id: &str) -> bool {
        let pool = crate::lock_util::read_or_recover(&self.inner, "SharedWorld::fork_exists");
        pool.forks.contains_key(fork_id)
    }

    /// Run a closure with exclusive access to a fork's world and the
    /// shared schedule. Returns `None` if the fork does not exist.
    ///
    /// The fork and the main world share the same `Schedule`; while the
    /// closure is executing no other tick (main or fork) can run,
    /// because the whole pool is locked for the duration.
    pub fn with_fork<F, R>(&self, fork_id: &str, f: F) -> Option<R>
    where
        F: FnOnce(&mut World, &mut Schedule) -> R,
    {
        let mut pool = crate::lock_util::write_or_recover(&self.inner, "SharedWorld::with_fork");
        let WorldPool {
            schedule, forks, ..
        } = &mut *pool;
        let fork_world = forks.get_mut(fork_id)?;
        Some(f(fork_world, schedule))
    }

    /// Run a closure with read-only access to a fork's world. Returns
    /// `None` if the fork does not exist.
    pub fn with_fork_ref<F, R>(&self, fork_id: &str, f: F) -> Option<R>
    where
        F: FnOnce(&World) -> R,
    {
        let pool = crate::lock_util::read_or_recover(&self.inner, "SharedWorld::with_fork_ref");
        pool.forks.get(fork_id).map(f)
    }
}

/// Errors that can occur during fork lifecycle operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForkError {
    /// A fork with this id already exists.
    AlreadyExists(String),
    /// No fork with the requested id.
    NotFound(String),
}

impl std::fmt::Display for ForkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForkError::AlreadyExists(id) => write!(f, "fork '{id}' already exists"),
            ForkError::NotFound(id) => write!(f, "fork '{id}' not found"),
        }
    }
}

impl std::error::Error for ForkError {}

/// RAII write guard for the world pool.
///
/// Provides `&mut World` access while holding the write lock.
/// Other fields of the owning struct remain accessible (disjoint borrows).
pub struct WorldWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, WorldPool>,
}

impl WorldWriteGuard<'_> {
    /// Get mutable reference to the main world.
    pub fn world(&mut self) -> &mut World {
        &mut self.guard.main
    }

    /// Get mutable reference to the shared schedule.
    pub fn schedule(&mut self) -> &mut Schedule {
        &mut self.guard.schedule
    }

    /// Get mutable reference to a specific fork's world.
    pub fn fork(&mut self, fork_id: &str) -> Option<&mut World> {
        self.guard.fork_mut(fork_id)
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
    /// Get shared reference to the main world.
    pub fn world(&self) -> &World {
        &self.guard.main
    }

    /// Get shared reference to a specific fork's world.
    pub fn fork(&self, fork_id: &str) -> Option<&World> {
        self.guard.fork_ref(fork_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Schedule;

    #[derive(Clone, Debug, PartialEq)]
    struct Hp(i32);

    #[derive(Clone, Debug, PartialEq)]
    struct Tick(u32);

    fn setup() -> SharedWorld {
        let mut world = World::new();
        world.insert_resource(Tick(0));
        world.spawn(Hp(100));
        SharedWorld::new(world, Schedule::new())
    }

    #[test]
    fn fork_creates_independent_copy() {
        let shared = setup();
        shared.fork("scenario-a").unwrap();

        // Mutate the fork via with_fork.
        shared
            .with_fork("scenario-a", |w, _| {
                let mut hp_values: Vec<_> = crate::Query::<&Hp>::new(w)
                    .iter()
                    .map(|h| h.0)
                    .collect();
                hp_values.sort();
                assert_eq!(hp_values, vec![100]);
                // Mutate: set hp to 1 on all entities.
                let entities: Vec<_> = w.all_entities();
                for e in entities {
                    if let Some(hp) = w.get_mut::<Hp>(e) {
                        hp.0 = 1;
                    }
                }
            })
            .unwrap();

        // Main world is unchanged.
        shared.with(|w, _| {
            let hp_values: Vec<_> = crate::Query::<&Hp>::new(w).iter().map(|h| h.0).collect();
            assert_eq!(hp_values, vec![100]);
        });

        // Fork sees its own mutation.
        shared
            .with_fork("scenario-a", |w, _| {
                let hp_values: Vec<_> = crate::Query::<&Hp>::new(w).iter().map(|h| h.0).collect();
                assert_eq!(hp_values, vec![1]);
            })
            .unwrap();
    }

    #[test]
    fn fork_duplicate_id_is_error() {
        let shared = setup();
        shared.fork("a").unwrap();
        assert_eq!(
            shared.fork("a"),
            Err(ForkError::AlreadyExists("a".to_string()))
        );
    }

    #[test]
    fn delete_fork_removes_it() {
        let shared = setup();
        shared.fork("a").unwrap();
        assert!(shared.fork_exists("a"));
        assert!(shared.delete_fork("a"));
        assert!(!shared.fork_exists("a"));
        assert!(!shared.delete_fork("a"));
    }

    #[test]
    fn list_forks_returns_all_active() {
        let shared = setup();
        shared.fork("a").unwrap();
        shared.fork("b").unwrap();
        let mut ids = shared.list_forks();
        ids.sort();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn with_fork_missing_returns_none() {
        let shared = setup();
        assert!(shared.with_fork("nope", |_, _| ()).is_none());
    }

    #[test]
    fn fork_resources_are_independent() {
        let shared = setup();
        shared.fork("a").unwrap();

        shared
            .with_fork("a", |w, _| {
                *w.resource_mut::<Tick>().unwrap() = Tick(999);
            })
            .unwrap();

        shared.with(|w, _| {
            assert_eq!(w.resource::<Tick>().unwrap().0, 0);
        });

        shared
            .with_fork("a", |w, _| {
                assert_eq!(w.resource::<Tick>().unwrap().0, 999);
            })
            .unwrap();
    }
}
