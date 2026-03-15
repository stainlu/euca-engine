use crate::world::World;

/// A system that operates on the world.
///
/// Systems are the "S" in ECS — they contain the logic that operates on
/// entities and components. For now, systems are simple closures that
/// take `&mut World`. Parallel execution and parameter extraction will
/// be added in a later iteration.
pub trait System: Send + Sync {
    /// Execute this system.
    fn run(&mut self, world: &mut World);

    /// Optional name for debugging/profiling.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// Wrapper that converts a closure into a System.
pub struct FunctionSystem<F: FnMut(&mut World) + Send + Sync> {
    func: F,
    name: &'static str,
}

impl<F: FnMut(&mut World) + Send + Sync> System for FunctionSystem<F> {
    fn run(&mut self, world: &mut World) {
        (self.func)(world);
    }

    fn name(&self) -> &str {
        self.name
    }
}

/// Trait for converting things into systems.
pub trait IntoSystem {
    type System: System;
    fn into_system(self) -> Self::System;
}

/// Any `FnMut(&mut World) + Send + Sync` can become a system.
impl<F: FnMut(&mut World) + Send + Sync + 'static> IntoSystem for F {
    type System = FunctionSystem<F>;

    fn into_system(self) -> Self::System {
        FunctionSystem {
            func: self,
            name: std::any::type_name::<F>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Counter(u32);

    #[test]
    fn function_system() {
        let mut world = World::new();
        world.insert_resource(Counter(0));

        let mut sys = (|w: &mut World| {
            w.resource_mut::<Counter>().unwrap().0 += 1;
        })
        .into_system();

        sys.run(&mut world);
        sys.run(&mut world);

        assert_eq!(world.resource::<Counter>().unwrap().0, 2);
    }
}
