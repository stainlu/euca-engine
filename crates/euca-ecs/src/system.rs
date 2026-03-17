use crate::system_param::SystemAccess;
use crate::world::World;

/// A system that operates on the world.
///
/// Systems are the "S" in ECS — they contain the logic that operates on
/// entities and components.
pub trait System: Send + Sync {
    /// Execute this system.
    fn run(&mut self, world: &mut World);

    /// Optional name for debugging/profiling.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Declare what this system reads/writes, for parallel scheduling.
    /// Returns empty by default (conservative: assumed to access everything).
    fn accesses(&self) -> &[SystemAccess] {
        &[]
    }

    /// Optional label for ordering dependencies (e.g., "physics", "movement").
    fn label(&self) -> Option<&str> {
        None
    }

    /// Systems that must run before this one (by label).
    fn after(&self) -> &[&str] {
        &[]
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
///
/// The `Marker` type parameter disambiguates between different function
/// signatures. Currently only `()` is used (for `fn(&mut World)`).
pub trait IntoSystem<Marker = ()> {
    type System: System;
    fn into_system(self) -> Self::System;
}

/// Any `FnMut(&mut World) + Send + Sync` can become a system.
impl<F: FnMut(&mut World) + Send + Sync + 'static> IntoSystem<()> for F {
    type System = FunctionSystem<F>;

    fn into_system(self) -> Self::System {
        FunctionSystem {
            func: self,
            name: std::any::type_name::<F>(),
        }
    }
}

/// A system with explicitly declared access metadata.
///
/// Wraps any system and attaches access information for the parallel scheduler.
pub struct AccessSystem<S: System> {
    system: S,
    accesses: Vec<SystemAccess>,
}

impl<S: System> System for AccessSystem<S> {
    fn run(&mut self, world: &mut World) {
        self.system.run(world);
    }

    fn name(&self) -> &str {
        self.system.name()
    }

    fn accesses(&self) -> &[SystemAccess] {
        &self.accesses
    }
}

impl<S: System> AccessSystem<S> {
    /// Wrap a system with explicit access declarations.
    pub fn new(system: S, accesses: Vec<SystemAccess>) -> Self {
        Self { system, accesses }
    }
}

/// A system with a label and ordering dependencies.
pub struct LabeledSystem<S: System> {
    system: S,
    label_str: &'static str,
    after_labels: Vec<&'static str>,
}

impl<S: System> System for LabeledSystem<S> {
    fn run(&mut self, world: &mut World) {
        self.system.run(world);
    }
    fn name(&self) -> &str {
        self.system.name()
    }
    fn accesses(&self) -> &[SystemAccess] {
        self.system.accesses()
    }
    fn label(&self) -> Option<&str> {
        Some(self.label_str)
    }
    fn after(&self) -> &[&str] {
        &self.after_labels
    }
}

impl<S: System> LabeledSystem<S> {
    /// Create a labeled system.
    pub fn new(system: S, label: &'static str) -> Self {
        Self {
            system,
            label_str: label,
            after_labels: Vec::new(),
        }
    }

    /// Declare that this system must run after another labeled system.
    pub fn after(mut self, label: &'static str) -> Self {
        self.after_labels.push(label);
        self
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

    #[test]
    fn access_system_metadata() {
        use std::any::TypeId;

        let sys = (|_w: &mut World| {}).into_system();
        let access_sys = AccessSystem::new(
            sys,
            vec![
                SystemAccess::ResourceRead(TypeId::of::<Counter>()),
                SystemAccess::ComponentWrite(crate::ComponentId::from_raw(0)),
            ],
        );

        assert_eq!(access_sys.accesses().len(), 2);
    }
}
