use crate::system::{IntoSystem, System};
use crate::world::World;

/// An ordered collection of systems that execute sequentially.
///
/// Deterministic: given the same system order and world state,
/// execution always produces the same result.
pub struct Schedule {
    systems: Vec<Box<dyn System>>,
}

impl Schedule {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
        }
    }

    /// Add a system to the end of the schedule.
    pub fn add_system<S: IntoSystem + 'static>(&mut self, system: S) -> &mut Self
    where
        S::System: 'static,
    {
        self.systems.push(Box::new(system.into_system()));
        self
    }

    /// Run all systems in order, then advance the world tick.
    pub fn run(&mut self, world: &mut World) {
        for system in &mut self.systems {
            system.run(world);
        }
        world.update_events();
        world.tick();
    }

    /// Number of systems in this schedule.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// Whether the schedule is empty.
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
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

        // Can't easily share state between closures without unsafe,
        // so use resources instead
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

        // Spawn entities with Position + Velocity
        let e1 = world.spawn(Position { x: 0.0, y: 0.0 });
        world.insert(e1, Velocity { dx: 1.0, dy: 2.0 });

        let e2 = world.spawn(Position { x: 10.0, y: 10.0 });
        world.insert(e2, Velocity { dx: -1.0, dy: 0.0 });

        // Movement system
        let mut schedule = Schedule::new();
        schedule.add_system(|world: &mut World| {
            // Collect entity + velocity data first (avoid borrow conflict)
            let updates: Vec<_> = {
                let query = Query::<(crate::Entity, &Velocity)>::new(world);
                query.iter().map(|(e, v)| (e, v.dx, v.dy)).collect()
            };
            // Apply position changes
            for (entity, dx, dy) in updates {
                if let Some(pos) = world.get_mut::<Position>(entity) {
                    pos.x += dx;
                    pos.y += dy;
                }
            }
        });

        // Run 3 ticks
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
}
