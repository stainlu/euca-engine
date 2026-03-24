use std::any::TypeId;
use std::collections::HashSet;

use crate::event::Events;
use crate::system::{IntoSystem, System};
use crate::system_param::validate_no_conflicts;
use crate::world::World;

/// A batch of systems that can run in parallel (no access conflicts).
struct Batch {
    /// Indices into Stage::systems.
    system_indices: Vec<usize>,
}

/// A stage is a group of systems. Batches within a stage run sequentially,
/// but systems within a batch run in parallel if they have declared
/// non-conflicting accesses.
struct Stage {
    systems: Vec<Box<dyn System>>,
    batches: Vec<Batch>,
    dirty: bool,
}

impl Stage {
    fn new() -> Self {
        Self {
            systems: Vec::new(),
            batches: Vec::new(),
            dirty: true,
        }
    }

    fn add_system(&mut self, system: Box<dyn System>) {
        self.systems.push(system);
        self.dirty = true;
    }

    fn rebuild_batches(&mut self) {
        self.batches.clear();
        let order = self.topological_order();

        for &sys_idx in &order {
            let sys_accesses = self.systems[sys_idx].accesses();

            if sys_accesses.is_empty() {
                self.batches.push(Batch {
                    system_indices: vec![sys_idx],
                });
                continue;
            }

            let mut placed = false;
            for batch in &mut self.batches {
                let mut conflicts = false;
                for &other_idx in &batch.system_indices {
                    let other_accesses = self.systems[other_idx].accesses();
                    if other_accesses.is_empty()
                        || !validate_no_conflicts(sys_accesses, other_accesses)
                    {
                        conflicts = true;
                        break;
                    }
                }
                if !conflicts {
                    batch.system_indices.push(sys_idx);
                    placed = true;
                    break;
                }
            }

            if !placed {
                self.batches.push(Batch {
                    system_indices: vec![sys_idx],
                });
            }
        }

        self.dirty = false;
    }

    fn topological_order(&self) -> Vec<usize> {
        use std::collections::HashMap;

        let mut label_to_idx: HashMap<&str, usize> = HashMap::new();
        for (i, sys) in self.systems.iter().enumerate() {
            if let Some(label) = sys.label() {
                label_to_idx.insert(label, i);
            }
        }

        let n = self.systems.len();
        let mut in_degree = vec![0u32; n];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (i, sys) in self.systems.iter().enumerate() {
            for &dep_label in sys.after() {
                if let Some(&dep_idx) = label_to_idx.get(dep_label) {
                    dependents[dep_idx].push(i);
                    in_degree[i] += 1;
                }
            }
        }

        let mut queue: std::collections::VecDeque<usize> =
            (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(idx) = queue.pop_front() {
            order.push(idx);
            for &dep in &dependents[idx] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    queue.push_back(dep);
                }
            }
        }

        if order.len() < n {
            let unresolved: Vec<_> = (0..n)
                .filter(|i| !order.contains(i))
                .map(|i| self.systems[i].name().to_string())
                .collect();
            log::warn!(
                "System dependency cycle detected involving: {:?}. \
                 These systems will run in declaration order.",
                unresolved
            );
            for i in 0..n {
                if !order.contains(&i) {
                    order.push(i);
                }
            }
        }

        order
    }

    fn run(&mut self, world: &mut World) {
        if self.dirty {
            self.rebuild_batches();
        }

        for batch in &self.batches {
            if batch.system_indices.len() == 1 {
                self.systems[batch.system_indices[0]].run(world);
            } else {
                unsafe {
                    run_batch_parallel(&mut self.systems, &batch.system_indices, world);
                }
            }
        }
    }
}

struct SystemJob(*mut dyn System, *mut World);
unsafe impl Send for SystemJob {}

impl SystemJob {
    unsafe fn run(self) {
        unsafe { (*self.0).run(&mut *self.1) }
    }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn run_batch_parallel(
    systems: &mut [Box<dyn System>],
    indices: &[usize],
    world: &mut World,
) {
    let world_ptr = world as *mut World;

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for &idx in indices {
            let job = SystemJob(&mut *systems[idx] as *mut dyn System, world_ptr);
            handles.push(s.spawn(move || unsafe { job.run() }));
        }
        for h in handles {
            match h.join() {
                Ok(()) => {}
                Err(e) => log::error!("System panicked during parallel execution: {:?}", e),
            }
        }
    });
}

/// WASM fallback: no thread support, run systems sequentially.
#[cfg(target_arch = "wasm32")]
unsafe fn run_batch_parallel(
    systems: &mut [Box<dyn System>],
    indices: &[usize],
    world: &mut World,
) {
    for &idx in indices {
        systems[idx].run(world);
    }
}

/// An ordered collection of stages, each containing systems.
///
/// Systems within a stage may run in parallel if they declare non-conflicting
/// accesses via `System::accesses()`. Stages run sequentially with barriers.
///
/// Systems without declared accesses run sequentially (conservative default).
pub struct Schedule {
    stages: Vec<Stage>,
    startup_systems: Vec<Box<dyn System>>,
    shutdown_systems: Vec<Box<dyn System>>,
    started: bool,
}

impl Schedule {
    /// Creates a schedule with a single default stage.
    pub fn new() -> Self {
        Self {
            stages: vec![Stage::new()],
            startup_systems: Vec::new(),
            shutdown_systems: Vec::new(),
            started: false,
        }
    }

    /// Add a system to the default (first) stage.
    pub fn add_system<M: 'static, S: IntoSystem<M> + 'static>(&mut self, system: S) -> &mut Self
    where
        S::System: 'static,
    {
        self.stages[0].add_system(Box::new(system.into_system()));
        self
    }

    /// Add a new empty stage and return its index.
    pub fn add_stage(&mut self) -> usize {
        let idx = self.stages.len();
        self.stages.push(Stage::new());
        idx
    }

    /// Add a system to a specific stage.
    pub fn add_system_to_stage<M: 'static, S: IntoSystem<M> + 'static>(
        &mut self,
        stage: usize,
        system: S,
    ) -> &mut Self
    where
        S::System: 'static,
    {
        self.stages[stage].add_system(Box::new(system.into_system()));
        self
    }

    /// Add a system that runs once on the first call to `run()`.
    pub fn add_startup_system<M: 'static, S: IntoSystem<M> + 'static>(
        &mut self,
        system: S,
    ) -> &mut Self
    where
        S::System: 'static,
    {
        self.startup_systems.push(Box::new(system.into_system()));
        self
    }

    /// Add a system that runs when `shutdown()` is called.
    pub fn add_shutdown_system<M: 'static, S: IntoSystem<M> + 'static>(
        &mut self,
        system: S,
    ) -> &mut Self
    where
        S::System: 'static,
    {
        self.shutdown_systems.push(Box::new(system.into_system()));
        self
    }

    /// Run all stages in order, then advance the world tick.
    pub fn run(&mut self, world: &mut World) {
        if !self.started {
            self.started = true;
            for sys in &mut self.startup_systems {
                sys.run(world);
            }
        }

        for stage in &mut self.stages {
            stage.run(world);
        }
        world.update_events();
        // Also update the Events resource if present (gameplay systems use this)
        if let Some(events) = world.resource_mut::<Events>() {
            events.update();
        }
        world.tick();
    }

    /// Run shutdown systems. Call once when the application is exiting.
    pub fn shutdown(&mut self, world: &mut World) {
        for sys in &mut self.shutdown_systems {
            sys.run(world);
        }
    }

    /// Total number of systems across all stages.
    pub fn len(&self) -> usize {
        self.stages.iter().map(|s| s.systems.len()).sum()
    }

    /// Whether the schedule has no systems.
    pub fn is_empty(&self) -> bool {
        self.stages.iter().all(|s| s.systems.is_empty())
    }
}

impl Default for Schedule {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ParallelSchedule - opt-in parallel system scheduler
// ────────────────────────────────────────────────────────────────────────────

/// Describes the component and resource accesses of a system for parallel scheduling.
///
/// Two systems conflict if one writes a component/resource that the other reads or writes.
/// Systems with no conflicts can execute simultaneously.
#[derive(Clone, Debug, Default)]
pub struct ParallelSystemAccess {
    /// Component `TypeId`s this system reads.
    pub reads: HashSet<TypeId>,
    /// Component `TypeId`s this system writes.
    pub writes: HashSet<TypeId>,
    /// Resource `TypeId`s this system reads.
    pub resources_read: HashSet<TypeId>,
    /// Resource `TypeId`s this system writes.
    pub resources_write: HashSet<TypeId>,
}

impl ParallelSystemAccess {
    /// Create an empty access descriptor (no declared accesses).
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a component type as read.
    pub fn read<T: 'static>(mut self) -> Self {
        self.reads.insert(TypeId::of::<T>());
        self
    }

    /// Declare a component type as written.
    pub fn write<T: 'static>(mut self) -> Self {
        self.writes.insert(TypeId::of::<T>());
        self
    }

    /// Declare a resource type as read.
    pub fn resource_read<T: 'static>(mut self) -> Self {
        self.resources_read.insert(TypeId::of::<T>());
        self
    }

    /// Declare a resource type as written.
    pub fn resource_write<T: 'static>(mut self) -> Self {
        self.resources_write.insert(TypeId::of::<T>());
        self
    }

    /// Returns true if this access set has no declared accesses at all.
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty()
            && self.writes.is_empty()
            && self.resources_read.is_empty()
            && self.resources_write.is_empty()
    }

    /// Returns true if `self` and `other` conflict.
    ///
    /// A conflict exists when one system writes a component/resource that the
    /// other reads or writes.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        if !self.writes.is_disjoint(&other.reads)
            || !self.writes.is_disjoint(&other.writes)
            || !self.reads.is_disjoint(&other.writes)
        {
            return true;
        }

        if !self.resources_write.is_disjoint(&other.resources_read)
            || !self.resources_write.is_disjoint(&other.resources_write)
            || !self.resources_read.is_disjoint(&other.resources_write)
        {
            return true;
        }

        false
    }
}

/// A group of non-conflicting systems that can execute simultaneously.
pub struct SystemBatch {
    indices: Vec<usize>,
}

impl SystemBatch {
    /// Returns the system indices in this batch.
    pub fn system_indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns how many systems are in this batch.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true if this batch contains no systems.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Internal entry for a system registered with `ParallelSchedule`.
struct ParallelSystemEntry {
    name: String,
    system: Box<dyn System>,
    access: ParallelSystemAccess,
    after: Vec<String>,
}

/// An alternative to [`Schedule`] that analyzes access patterns and groups
/// non-conflicting systems into parallel batches.
///
/// Users opt in to `ParallelSchedule` when they want fine-grained control
/// over declared accesses and explicit parallelism.
///
/// # Usage
///
/// ```ignore
/// let mut sched = ParallelSchedule::new();
/// sched.add_system("physics", physics_fn, ParallelSystemAccess::new().write::<Vel>());
/// sched.add_system("render", render_fn, ParallelSystemAccess::new().read::<Pos>());
/// sched.build();
/// sched.run(&mut world);
/// ```
pub struct ParallelSchedule {
    systems: Vec<ParallelSystemEntry>,
    batches: Vec<SystemBatch>,
    built: bool,
}

impl ParallelSchedule {
    /// Create an empty parallel schedule.
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            batches: Vec::new(),
            built: false,
        }
    }

    /// Register a system with a name and declared access pattern.
    ///
    /// Returns a [`SystemHandle`] that can be used to add ordering constraints.
    pub fn add_system<M: 'static, S: IntoSystem<M> + 'static>(
        &mut self,
        name: &str,
        system: S,
        access: ParallelSystemAccess,
    ) -> SystemHandle<'_>
    where
        S::System: 'static,
    {
        let idx = self.systems.len();
        self.systems.push(ParallelSystemEntry {
            name: name.to_string(),
            system: Box::new(system.into_system()),
            access,
            after: Vec::new(),
        });
        self.built = false;
        SystemHandle {
            schedule: self,
            index: idx,
        }
    }

    /// Analyze access patterns, resolve ordering constraints, and group
    /// non-conflicting systems into parallel batches.
    ///
    /// Must be called before `run()`. Calling `add_system()` after `build()`
    /// invalidates the schedule and requires another `build()`.
    pub fn build(&mut self) {
        let order = self.topological_order();
        self.batches.clear();

        for &sys_idx in &order {
            let sys_access = &self.systems[sys_idx].access;
            let min_batch = self.min_batch_for(sys_idx);

            let mut placed = false;
            for batch_idx in min_batch..self.batches.len() {
                let batch = &self.batches[batch_idx];
                let conflicts = batch
                    .indices
                    .iter()
                    .any(|&other_idx| sys_access.conflicts_with(&self.systems[other_idx].access));
                if !conflicts {
                    self.batches[batch_idx].indices.push(sys_idx);
                    placed = true;
                    break;
                }
            }

            if !placed {
                self.batches.push(SystemBatch {
                    indices: vec![sys_idx],
                });
            }
        }

        self.built = true;
    }

    /// Execute each batch. Systems within a batch run in parallel using
    /// `std::thread::scope`; batches run sequentially.
    ///
    /// # Panics
    ///
    /// Panics if `build()` has not been called since the last `add_system()`.
    pub fn run(&mut self, world: &mut World) {
        assert!(
            self.built,
            "ParallelSchedule::run() called before build(). Call build() first."
        );

        for batch_idx in 0..self.batches.len() {
            let batch_len = self.batches[batch_idx].indices.len();
            if batch_len == 1 {
                let sys_idx = self.batches[batch_idx].indices[0];
                self.systems[sys_idx].system.run(world);
            } else {
                let indices: Vec<usize> = self.batches[batch_idx].indices.clone();
                // SAFETY: Systems in this batch have been validated to have
                // non-conflicting accesses. Each system accesses disjoint data.
                unsafe {
                    self.run_batch_parallel(&indices, world);
                }
            }
        }
    }

    /// Returns the batches produced by `build()`.
    pub fn batches(&self) -> &[SystemBatch] {
        &self.batches
    }

    /// Total number of registered systems.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// Whether the schedule has no systems.
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// Topological sort respecting `after()` dependencies using Kahn's algorithm.
    fn topological_order(&self) -> Vec<usize> {
        use std::collections::HashMap;
        use std::collections::VecDeque;

        let n = self.systems.len();

        let name_to_idx: HashMap<&str, usize> = self
            .systems
            .iter()
            .enumerate()
            .map(|(i, entry)| (entry.name.as_str(), i))
            .collect();

        let mut in_degree = vec![0u32; n];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (i, entry) in self.systems.iter().enumerate() {
            for dep_name in &entry.after {
                if let Some(&dep_idx) = name_to_idx.get(dep_name.as_str()) {
                    dependents[dep_idx].push(i);
                    in_degree[i] += 1;
                }
            }
        }

        let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(idx) = queue.pop_front() {
            order.push(idx);
            for &dep in &dependents[idx] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    queue.push_back(dep);
                }
            }
        }

        if order.len() < n {
            let unresolved: Vec<_> = (0..n)
                .filter(|i| !order.contains(i))
                .map(|i| self.systems[i].name.clone())
                .collect();
            panic!("dependency cycle detected: {:?}", unresolved);
        }

        order
    }

    /// Find the earliest batch index where `sys_idx` can be placed,
    /// respecting `after()` ordering constraints.
    fn min_batch_for(&self, sys_idx: usize) -> usize {
        let entry = &self.systems[sys_idx];
        if entry.after.is_empty() {
            return 0;
        }

        let name_to_idx: std::collections::HashMap<&str, usize> = self
            .systems
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name.as_str(), i))
            .collect();

        let mut min_batch = 0;
        for dep_name in &entry.after {
            if let Some(&dep_idx) = name_to_idx.get(dep_name.as_str()) {
                for (batch_idx, batch) in self.batches.iter().enumerate() {
                    if batch.indices.contains(&dep_idx) {
                        min_batch = min_batch.max(batch_idx + 1);
                    }
                }
            }
        }
        min_batch
    }

    /// Run a batch of systems in parallel using `std::thread::scope`.
    ///
    /// # Safety
    ///
    /// Caller must ensure that the systems at the given indices have
    /// non-conflicting accesses.
    #[cfg(not(target_arch = "wasm32"))]
    unsafe fn run_batch_parallel(&mut self, indices: &[usize], world: &mut World) {
        let world_ptr = world as *mut World;

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for &idx in indices {
                let job = SystemJob(&mut *self.systems[idx].system as *mut dyn System, world_ptr);
                handles.push(s.spawn(move || unsafe { job.run() }));
            }
            for h in handles {
                match h.join() {
                    Ok(()) => {}
                    Err(e) => log::error!("System panicked during parallel execution: {:?}", e),
                }
            }
        });
    }

    /// WASM fallback: no thread support, run systems sequentially.
    #[cfg(target_arch = "wasm32")]
    unsafe fn run_batch_parallel(&mut self, indices: &[usize], world: &mut World) {
        for &idx in indices {
            self.systems[idx].system.run(world);
        }
    }
}

impl Default for ParallelSchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle returned by [`ParallelSchedule::add_system`] for adding ordering
/// constraints via method chaining.
pub struct SystemHandle<'a> {
    schedule: &'a mut ParallelSchedule,
    index: usize,
}

impl SystemHandle<'_> {
    /// Declare that this system must run after the named system.
    pub fn after(self, dependency: &str) -> Self {
        self.schedule.systems[self.index]
            .after
            .push(dependency.to_string());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::Query;
    use crate::system::AccessSystem;
    use crate::system_param::SystemAccess;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Debug, Clone, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }

    #[test]
    fn schedule_runs_systems_in_order() {
        let mut world = World::new();
        world.insert_resource(Vec::<String>::new());

        let mut schedule = Schedule::new();
        schedule.add_system(|w: &mut World| {
            w.resource_mut::<Vec<String>>().unwrap().push("A".into());
        });
        schedule.add_system(|w: &mut World| {
            w.resource_mut::<Vec<String>>().unwrap().push("B".into());
        });
        schedule.add_system(|w: &mut World| {
            w.resource_mut::<Vec<String>>().unwrap().push("C".into());
        });

        schedule.run(&mut world);

        let log = world.resource::<Vec<String>>().unwrap();
        assert_eq!(
            log,
            &vec!["A".to_string(), "B".to_string(), "C".to_string()]
        );
    }

    #[test]
    fn schedule_advances_tick() {
        let mut world = World::new();
        let mut schedule = Schedule::new();

        assert_eq!(world.current_tick(), 0);
        schedule.run(&mut world);
        assert_eq!(world.current_tick(), 1);
        schedule.run(&mut world);
        assert_eq!(world.current_tick(), 2);
    }

    #[test]
    fn movement_system_integration() {
        let mut world = World::new();

        let e1 = world.spawn(Position { x: 0.0, y: 0.0 });
        world.insert(e1, Velocity { dx: 1.0, dy: 2.0 });

        let e2 = world.spawn(Position { x: 10.0, y: 10.0 });
        world.insert(e2, Velocity { dx: -1.0, dy: 0.0 });

        let mut schedule = Schedule::new();
        schedule.add_system(|world: &mut World| {
            let updates: Vec<_> = {
                let query = Query::<(crate::Entity, &Velocity)>::new(world);
                query.iter().map(|(e, v)| (e, v.dx, v.dy)).collect()
            };
            for (entity, dx, dy) in updates {
                if let Some(pos) = world.get_mut::<Position>(entity) {
                    pos.x += dx;
                    pos.y += dy;
                }
            }
        });

        for _ in 0..3 {
            schedule.run(&mut world);
        }

        assert_eq!(
            world.get::<Position>(e1).unwrap(),
            &Position { x: 3.0, y: 6.0 }
        );
        assert_eq!(
            world.get::<Position>(e2).unwrap(),
            &Position { x: 7.0, y: 10.0 }
        );
        assert_eq!(world.current_tick(), 3);
    }

    #[test]
    fn stages_run_in_order() {
        let mut world = World::new();
        world.insert_resource(Vec::<String>::new());

        let mut schedule = Schedule::new();

        schedule.add_system(|w: &mut World| {
            w.resource_mut::<Vec<String>>().unwrap().push("S0".into());
        });

        let s1 = schedule.add_stage();
        schedule.add_system_to_stage(s1, |w: &mut World| {
            w.resource_mut::<Vec<String>>().unwrap().push("S1".into());
        });

        let s2 = schedule.add_stage();
        schedule.add_system_to_stage(s2, |w: &mut World| {
            w.resource_mut::<Vec<String>>().unwrap().push("S2".into());
        });

        schedule.run(&mut world);

        let log = world.resource::<Vec<String>>().unwrap();
        assert_eq!(
            log,
            &vec!["S0".to_string(), "S1".to_string(), "S2".to_string()]
        );
    }

    #[test]
    fn par_for_each_processes_all_entities() {
        let mut world = World::new();
        for i in 0..100 {
            world.spawn(Position {
                x: i as f32,
                y: 0.0,
            });
        }

        let count = AtomicU32::new(0);
        world.par_for_each::<Position>(|_entity, _pos| {
            count.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(count.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn undeclared_systems_run_sequentially() {
        let mut world = World::new();
        world.insert_resource(Vec::<String>::new());

        let mut schedule = Schedule::new();
        schedule.add_system(|w: &mut World| {
            w.resource_mut::<Vec<String>>()
                .unwrap()
                .push("first".into());
        });
        schedule.add_system(|w: &mut World| {
            w.resource_mut::<Vec<String>>()
                .unwrap()
                .push("second".into());
        });

        schedule.run(&mut world);

        let log = world.resource::<Vec<String>>().unwrap();
        assert_eq!(log, &vec!["first".to_string(), "second".to_string()]);
    }

    #[test]
    fn parallel_batch_no_conflict() {
        use std::any::TypeId;

        let counter = std::sync::Arc::new(AtomicU32::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();

        let sys_a = AccessSystem::new(
            (move |_w: &mut World| {
                c1.fetch_add(1, Ordering::Relaxed);
            })
            .into_system(),
            vec![SystemAccess::ResourceRead(TypeId::of::<Position>())],
        );

        let sys_b = AccessSystem::new(
            (move |_w: &mut World| {
                c2.fetch_add(1, Ordering::Relaxed);
            })
            .into_system(),
            vec![SystemAccess::ResourceRead(TypeId::of::<Velocity>())],
        );

        let mut schedule = Schedule::new();
        schedule.stages[0].add_system(Box::new(sys_a));
        schedule.stages[0].add_system(Box::new(sys_b));

        let mut world = World::new();
        schedule.run(&mut world);

        assert_eq!(counter.load(Ordering::Relaxed), 2);
        assert_eq!(schedule.stages[0].batches.len(), 1);
        assert_eq!(schedule.stages[0].batches[0].system_indices.len(), 2);
    }

    #[test]
    fn parallel_batch_with_conflict() {
        use std::any::TypeId;

        let sys_a = AccessSystem::new(
            (|_w: &mut World| {}).into_system(),
            vec![SystemAccess::ResourceWrite(TypeId::of::<Position>())],
        );

        let sys_b = AccessSystem::new(
            (|_w: &mut World| {}).into_system(),
            vec![SystemAccess::ResourceWrite(TypeId::of::<Position>())],
        );

        let mut schedule = Schedule::new();
        schedule.stages[0].add_system(Box::new(sys_a));
        schedule.stages[0].add_system(Box::new(sys_b));

        let mut world = World::new();
        schedule.run(&mut world);

        assert_eq!(schedule.stages[0].batches.len(), 2);
    }

    #[test]
    fn mixed_declared_and_undeclared() {
        use std::any::TypeId;

        let mut schedule = Schedule::new();
        schedule.add_system(|_w: &mut World| {});

        let sys_a = AccessSystem::new(
            (|_w: &mut World| {}).into_system(),
            vec![SystemAccess::ResourceRead(TypeId::of::<Position>())],
        );
        let sys_b = AccessSystem::new(
            (|_w: &mut World| {}).into_system(),
            vec![SystemAccess::ResourceRead(TypeId::of::<Velocity>())],
        );
        schedule.stages[0].add_system(Box::new(sys_a));
        schedule.stages[0].add_system(Box::new(sys_b));

        let mut world = World::new();
        schedule.run(&mut world);

        assert_eq!(schedule.stages[0].batches.len(), 2);
        assert_eq!(schedule.stages[0].batches[0].system_indices.len(), 1);
        assert_eq!(schedule.stages[0].batches[1].system_indices.len(), 2);
    }

    // -- ParallelSchedule tests --

    #[derive(Debug)]
    #[allow(dead_code)]
    struct Health(f32);

    #[derive(Debug)]
    #[allow(dead_code)]
    struct Mana(f32);

    #[test]
    fn parallel_schedule_no_conflict_same_batch() {
        let counter = std::sync::Arc::new(AtomicU32::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();

        let mut schedule = ParallelSchedule::new();
        schedule.add_system(
            "read_pos",
            move |_w: &mut World| {
                c1.fetch_add(1, Ordering::Relaxed);
            },
            ParallelSystemAccess::new().read::<Position>(),
        );
        schedule.add_system(
            "read_vel",
            move |_w: &mut World| {
                c2.fetch_add(1, Ordering::Relaxed);
            },
            ParallelSystemAccess::new().read::<Velocity>(),
        );
        schedule.build();

        let mut world = World::new();
        schedule.run(&mut world);

        assert_eq!(counter.load(Ordering::Relaxed), 2);
        assert_eq!(schedule.batches().len(), 1);
        assert_eq!(schedule.batches()[0].len(), 2);
    }

    #[test]
    fn parallel_schedule_write_write_conflict_sequential() {
        let mut schedule = ParallelSchedule::new();
        schedule.add_system(
            "write_pos_a",
            |_w: &mut World| {},
            ParallelSystemAccess::new().write::<Position>(),
        );
        schedule.add_system(
            "write_pos_b",
            |_w: &mut World| {},
            ParallelSystemAccess::new().write::<Position>(),
        );
        schedule.build();

        assert_eq!(schedule.batches().len(), 2);
        assert_eq!(schedule.batches()[0].len(), 1);
        assert_eq!(schedule.batches()[1].len(), 1);
    }

    #[test]
    fn parallel_schedule_read_read_same_batch() {
        let mut schedule = ParallelSchedule::new();
        schedule.add_system(
            "read_pos_a",
            |_w: &mut World| {},
            ParallelSystemAccess::new().read::<Position>(),
        );
        schedule.add_system(
            "read_pos_b",
            |_w: &mut World| {},
            ParallelSystemAccess::new().read::<Position>(),
        );
        schedule.build();

        assert_eq!(schedule.batches().len(), 1);
        assert_eq!(schedule.batches()[0].len(), 2);
    }

    #[test]
    fn parallel_schedule_ordering_constraint_respected() {
        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let o1 = order.clone();
        let o2 = order.clone();

        let mut schedule = ParallelSchedule::new();
        schedule
            .add_system(
                "render",
                move |_w: &mut World| {
                    o2.lock().unwrap().push("render".into());
                },
                ParallelSystemAccess::new().read::<Position>(),
            )
            .after("physics");
        schedule.add_system(
            "physics",
            move |_w: &mut World| {
                o1.lock().unwrap().push("physics".into());
            },
            ParallelSystemAccess::new().read::<Velocity>(),
        );
        schedule.build();

        let mut world = World::new();
        schedule.run(&mut world);

        let log = order.lock().unwrap();
        assert_eq!(*log, vec!["physics".to_string(), "render".to_string()]);
        assert_eq!(schedule.batches().len(), 2);
    }

    #[test]
    fn parallel_schedule_empty() {
        let mut schedule = ParallelSchedule::new();
        schedule.build();

        assert!(schedule.batches().is_empty());
        assert!(schedule.is_empty());
        assert_eq!(schedule.len(), 0);

        let mut world = World::new();
        schedule.run(&mut world);
    }

    #[test]
    fn parallel_schedule_read_write_conflict() {
        let mut schedule = ParallelSchedule::new();
        schedule.add_system(
            "reader",
            |_w: &mut World| {},
            ParallelSystemAccess::new().read::<Position>(),
        );
        schedule.add_system(
            "writer",
            |_w: &mut World| {},
            ParallelSystemAccess::new().write::<Position>(),
        );
        schedule.build();

        assert_eq!(schedule.batches().len(), 2);
    }

    #[test]
    fn parallel_schedule_mixed_accesses_batching() {
        let mut schedule = ParallelSchedule::new();
        schedule.add_system(
            "A",
            |_w: &mut World| {},
            ParallelSystemAccess::new().read::<Position>(),
        );
        schedule.add_system(
            "B",
            |_w: &mut World| {},
            ParallelSystemAccess::new().read::<Velocity>(),
        );
        schedule.add_system(
            "C",
            |_w: &mut World| {},
            ParallelSystemAccess::new().write::<Position>(),
        );
        schedule.add_system(
            "D",
            |_w: &mut World| {},
            ParallelSystemAccess::new().read::<Health>(),
        );
        schedule.build();

        assert_eq!(schedule.batches().len(), 2);
        assert_eq!(schedule.batches()[0].len(), 3);
        assert_eq!(schedule.batches()[1].len(), 1);
    }

    #[test]
    #[should_panic(expected = "build()")]
    fn parallel_schedule_run_without_build_panics() {
        let mut schedule = ParallelSchedule::new();
        schedule.add_system("sys", |_w: &mut World| {}, ParallelSystemAccess::new());
        let mut world = World::new();
        schedule.run(&mut world);
    }

    #[test]
    #[should_panic(expected = "dependency cycle detected")]
    fn parallel_schedule_cycle_panics() {
        let mut schedule = ParallelSchedule::new();
        schedule
            .add_system(
                "A",
                |_w: &mut World| {},
                ParallelSystemAccess::new().read::<Position>(),
            )
            .after("B");
        schedule
            .add_system(
                "B",
                |_w: &mut World| {},
                ParallelSystemAccess::new().read::<Velocity>(),
            )
            .after("A");
        schedule.build(); // should panic due to cycle
    }

    #[test]
    fn parallel_system_access_conflicts_with() {
        let a = ParallelSystemAccess::new()
            .read::<Position>()
            .write::<Velocity>();
        let b = ParallelSystemAccess::new().read::<Velocity>();
        assert!(a.conflicts_with(&b));

        let c = ParallelSystemAccess::new().read::<Position>();
        assert!(!a.conflicts_with(&c));

        let d = ParallelSystemAccess::new().resource_write::<Health>();
        let e = ParallelSystemAccess::new().resource_read::<Health>();
        assert!(d.conflicts_with(&e));

        let f = ParallelSystemAccess::new().resource_read::<Health>();
        let g = ParallelSystemAccess::new().resource_read::<Health>();
        assert!(!f.conflicts_with(&g));
    }
}
