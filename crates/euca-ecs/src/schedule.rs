use crate::system::{IntoSystem, System};
use crate::world::World;

/// A stage is a group of systems that run in sequence.
/// Multiple stages run in order. Future: systems within a stage may run in parallel.
struct Stage {
    systems: Vec<Box<dyn System>>,
}

/// An ordered collection of stages, each containing systems.
///
/// Deterministic: given the same system order and world state,
/// execution always produces the same result.
///
/// Systems added via `add_system()` go into the default stage.
/// Use `add_stage()` and `add_system_to_stage()` for explicit ordering.
pub struct Schedule {
    stages: Vec<Stage>,
}

impl Schedule {
    /// Creates a schedule with a single default stage.
    pub fn new() -> Self {
        Self {
            stages: vec![Stage {
                systems: Vec::new(),
            }],
        }
    }

    /// Add a system to the default (first) stage.
    ///
    /// Accepts both old-style `fn(&mut World)` and new-style typed-param systems.
    pub fn add_system<M: 'static, S: IntoSystem<M> + 'static>(&mut self, system: S) -> &mut Self
    where
        S::System: 'static,
    {
        self.stages[0].systems.push(Box::new(system.into_system()));
        self
    }

    /// Add a new empty stage and return its index.
    pub fn add_stage(&mut self) -> usize {
        let idx = self.stages.len();
        self.stages.push(Stage {
            systems: Vec::new(),
        });
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
        self.stages[stage]
            .systems
            .push(Box::new(system.into_system()));
        self
    }

    /// Run all stages in order, then advance the world tick.
    pub fn run(&mut self, world: &mut World) {
        for stage in &mut self.stages {
            for system in &mut stage.systems {
                system.run(world);
            }
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
        use std::sync::atomic::{AtomicU32, Ordering};

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
}
