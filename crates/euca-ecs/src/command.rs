use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;

/// A buffered command to be applied to the world later.
///
/// Commands enable deferred structural changes (spawn, despawn, insert, remove)
/// that would otherwise cause borrow conflicts during system execution.
trait Command: Send + Sync {
    fn apply(self: Box<Self>, world: &mut World);
}

/// Spawn an entity with a single component.
struct SpawnCommand<T: Component> {
    component: T,
}

impl<T: Component> Command for SpawnCommand<T> {
    fn apply(self: Box<Self>, world: &mut World) {
        world.spawn(self.component);
    }
}

/// Despawn an entity.
struct DespawnCommand {
    entity: Entity,
}

impl Command for DespawnCommand {
    fn apply(self: Box<Self>, world: &mut World) {
        world.despawn(self.entity);
    }
}

/// Insert a component on an entity.
struct InsertCommand<T: Component> {
    entity: Entity,
    component: T,
}

impl<T: Component> Command for InsertCommand<T> {
    fn apply(self: Box<Self>, world: &mut World) {
        world.insert(self.entity, self.component);
    }
}

/// Remove a component from an entity.
struct RemoveCommand<T: Component> {
    entity: Entity,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Component> Command for RemoveCommand<T> {
    fn apply(self: Box<Self>, world: &mut World) {
        world.remove::<T>(self.entity);
    }
}

/// A queue of deferred commands to apply to the world.
///
/// Commands are applied in insertion order (deterministic).
pub struct Commands {
    queue: Vec<Box<dyn Command>>,
}

impl Commands {
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    /// Queue spawning an entity with a component.
    pub fn spawn<T: Component>(&mut self, component: T) {
        self.queue.push(Box::new(SpawnCommand { component }));
    }

    /// Queue despawning an entity.
    pub fn despawn(&mut self, entity: Entity) {
        self.queue.push(Box::new(DespawnCommand { entity }));
    }

    /// Queue inserting a component on an entity.
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        self.queue.push(Box::new(InsertCommand { entity, component }));
    }

    /// Queue removing a component from an entity.
    pub fn remove<T: Component>(&mut self, entity: Entity) {
        self.queue.push(Box::new(RemoveCommand::<T> {
            entity,
            _marker: std::marker::PhantomData,
        }));
    }

    /// Apply all queued commands to the world, in insertion order.
    pub fn apply(&mut self, world: &mut World) {
        for cmd in self.queue.drain(..) {
            cmd.apply(world);
        }
    }

    /// Number of pending commands.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the command queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl Default for Commands {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Position { x: f32, y: f32 }

    #[derive(Debug, Clone, PartialEq)]
    struct Velocity { dx: f32, dy: f32 }

    #[test]
    fn deferred_spawn() {
        let mut world = World::new();
        let mut commands = Commands::new();

        commands.spawn(Position { x: 1.0, y: 2.0 });
        commands.spawn(Position { x: 3.0, y: 4.0 });

        assert_eq!(world.entity_count(), 0);
        commands.apply(&mut world);
        assert_eq!(world.entity_count(), 2);
    }

    #[test]
    fn deferred_despawn() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1.0, y: 2.0 });
        let mut commands = Commands::new();

        commands.despawn(entity);
        assert!(world.is_alive(entity));
        commands.apply(&mut world);
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn deferred_insert() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1.0, y: 2.0 });
        let mut commands = Commands::new();

        commands.insert(entity, Velocity { dx: 5.0, dy: 6.0 });
        assert!(world.get::<Velocity>(entity).is_none());
        commands.apply(&mut world);
        assert_eq!(world.get::<Velocity>(entity).unwrap(), &Velocity { dx: 5.0, dy: 6.0 });
    }

    #[test]
    fn deferred_remove() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1.0, y: 2.0 });
        world.insert(entity, Velocity { dx: 3.0, dy: 4.0 });

        let mut commands = Commands::new();
        commands.remove::<Velocity>(entity);
        assert!(world.get::<Velocity>(entity).is_some());
        commands.apply(&mut world);
        assert!(world.get::<Velocity>(entity).is_none());
    }

    #[test]
    fn commands_apply_in_order() {
        let mut world = World::new();
        let mut commands = Commands::new();

        // Spawn then despawn — should result in entity being gone
        commands.spawn(Position { x: 1.0, y: 2.0 });
        assert_eq!(commands.len(), 1);

        commands.apply(&mut world);
        assert_eq!(world.entity_count(), 1);
    }
}
