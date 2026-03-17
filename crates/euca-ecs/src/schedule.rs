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

    /// Group systems into batches for parallel execution.
    ///
    /// Algorithm: greedy batching. For each system, try to place it in an
    /// existing batch. If it conflicts with any system in that batch, or if
    /// it has no declared accesses (conservative), start a new batch.
    fn rebuild_batches(&mut self) {
        self.batches.clear();

        for sys_idx in 0..self.systems.len() {
            let sys_accesses = self.systems[sys_idx].accesses();

            // Systems with no declared accesses are conservative — own batch
            if sys_accesses.is_empty() {
                self.batches.push(Batch {
                    system_indices: vec![sys_idx],
                });
                continue;
            }

            // Try to fit into an existing batch
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

    fn run(&mut self, world: &mut World) {
        if self.dirty {
            self.rebuild_batches();
        }

        for batch in &self.batches {
            if batch.system_indices.len() == 1 {
                // Single system — run directly, no overhead
                self.systems[batch.system_indices[0]].run(world);
            } else {
                // Multiple systems — run in parallel via rayon
                // SAFETY: Systems in this batch have been validated to have
                // non-conflicting accesses. Each system gets a raw pointer to
                // World, which is sound because they access disjoint data.
                unsafe {
                    run_batch_parallel(&mut self.systems, &batch.system_indices, world);
                }
            }
        }
    }
}

/// A system + world pointer pair that can be sent across threads.
///
/// Fields are private to force edition 2024 closures to capture the whole
/// struct (not individual fields), respecting our unsafe Send impl.
///
/// SAFETY: Caller must ensure no aliasing between different SystemJob instances.
struct SystemJob(*mut dyn System, *mut World);
unsafe impl Send for SystemJob {}

impl SystemJob {
    /// # Safety
    /// Caller must ensure exclusive access to the system and that the system's
    /// world access doesn't alias with other concurrently running systems.
    unsafe fn run(self) {
        unsafe { (*self.0).run(&mut *self.1) }
    }
}

/// Run multiple systems in parallel using std::thread::scope.
///
/// # Safety
/// Caller must ensure that the systems at the given indices have non-conflicting
/// accesses (validated by `validate_no_conflicts`).
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
            h.join().expect("System panicked during parallel execution");
        }
    });
}

/// An ordered collection of stages, each containing systems.
///
/// Systems within a stage may run in parallel if they declare non-conflicting
/// accesses via `System::accesses()`. Stages run sequentially with barriers.
///
/// Systems without declared accesses run sequentially (conservative default).
pub struct Schedule {
    stages: Vec<Stage>,
}

impl Schedule {
    /// Creates a schedule with a single default stage.
    pub fn new() -> Self {
        Self {
            stages: vec![Stage::new()],
        }
    }

    /// Add a system to the default (first) stage.
    ///
    /// Accepts both old-style `fn(&mut World)` and new-style typed-param systems.
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

    /// Run all stages in order, then advance the world tick.
    ///
    /// Within each stage, systems are grouped into batches. Systems in the same
    /// batch run in parallel via rayon. Batches run sequentially.
    pub fn run(&mut self, world: &mut World) {
        for stage in &mut self.stages {
            stage.run(world);
        }
        world.update_events();
        world.tick();
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

        // Stage 0 (default)
        schedule.add_system(|w: &mut World| {
            w.resource_mut::<Vec<String>>().unwrap().push("S0".into());
        });

        // Stage 1
        let s1 = schedule.add_stage();
        schedule.add_system_to_stage(s1, |w: &mut World| {
            w.resource_mut::<Vec<String>>().unwrap().push("S1".into());
        });

        // Stage 2
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

    // ── New tests: parallel scheduling ──

    #[test]
    fn undeclared_systems_run_sequentially() {
        // Systems with no accesses (default) should each get their own batch
        // and run in order — same behavior as before.
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
        // Two systems reading different resources → same batch → parallel
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
        // Must use add_system_to_stage with IntoSystem — but AccessSystem is already a System.
        // We need to box them directly.
        schedule.stages[0].add_system(Box::new(sys_a));
        schedule.stages[0].add_system(Box::new(sys_b));

        let mut world = World::new();
        schedule.run(&mut world);

        // Both systems ran
        assert_eq!(counter.load(Ordering::Relaxed), 2);

        // Verify they were batched together (1 batch with 2 systems)
        assert_eq!(schedule.stages[0].batches.len(), 1);
        assert_eq!(schedule.stages[0].batches[0].system_indices.len(), 2);
    }

    #[test]
    fn parallel_batch_with_conflict() {
        // Two systems writing the same resource → separate batches
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

        // Should be 2 separate batches (conflict)
        assert_eq!(schedule.stages[0].batches.len(), 2);
    }

    #[test]
    fn mixed_declared_and_undeclared() {
        // Undeclared system gets its own batch, declared systems may batch together
        use std::any::TypeId;

        let mut schedule = Schedule::new();

        // Undeclared (default) — own batch
        schedule.add_system(|_w: &mut World| {});

        // Two declared, non-conflicting — can batch
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

        // 2 batches: [undeclared] + [sys_a, sys_b]
        assert_eq!(schedule.stages[0].batches.len(), 2);
        assert_eq!(schedule.stages[0].batches[0].system_indices.len(), 1); // undeclared alone
        assert_eq!(schedule.stages[0].batches[1].system_indices.len(), 2); // declared together
    }
}
