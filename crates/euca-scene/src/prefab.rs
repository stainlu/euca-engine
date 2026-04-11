//! Prefab system: serializable entity templates with a typed component data model.
//!
//! [`ComponentData`] enumerates the common component types that can appear in a
//! prefab definition. This gives us serialisation, cloning, and exhaustive
//! matching for free, while [`PrefabRegistry`] manages named templates and
//! spawns entities from them.

use std::collections::HashMap;

use euca_ecs::{Entity, World};
use euca_math::{Transform, Vec3};
use serde::{Deserialize, Serialize};

use crate::{GlobalTransform, LocalTransform};

/// Serializable, type-safe representation of a component value that can be
/// stored inside a [`Prefab`].
///
/// Each variant maps 1:1 to a concrete component type. Adding new component
/// kinds is a matter of extending this enum and the `insert_into` helper.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ComponentData {
    /// World-space position.
    Position(Vec3),
    /// Hit points / health value.
    Health(f32),
    /// Team / faction identifier.
    Team(u8),
    /// Human-readable entity name or label.
    Name(String),
    /// Movement speed (units per second).
    Speed(f32),
    /// Damage output value.
    Damage(f32),
}

/// Lightweight wrapper components for the data-driven prefab variants.
///
/// These are the concrete ECS components that [`ComponentData`] maps to.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Health(pub f32);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Team(pub u8);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Name(pub String);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Speed(pub f32);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Damage(pub f32);

impl ComponentData {
    /// Insert this component data as a concrete ECS component on `entity`.
    fn insert_into(&self, world: &mut World, entity: Entity) {
        match self {
            ComponentData::Position(pos) => {
                let transform = Transform::from_translation(*pos);
                world.insert(entity, LocalTransform(transform));
                world.insert(entity, GlobalTransform(transform));
            }
            ComponentData::Health(hp) => {
                world.insert(entity, Health(*hp));
            }
            ComponentData::Team(team) => {
                world.insert(entity, Team(*team));
            }
            ComponentData::Name(name) => {
                world.insert(entity, Name(name.clone()));
            }
            ComponentData::Speed(speed) => {
                world.insert(entity, Speed(*speed));
            }
            ComponentData::Damage(damage) => {
                world.insert(entity, Damage(*damage));
            }
        }
    }
}

/// A named template for spawning entities with a predefined set of components.
///
/// Prefabs are serializable and cloneable: every call to
/// [`PrefabRegistry::spawn`] creates a fresh entity with its own component
/// copies.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prefab {
    /// Unique name used to look up this prefab in the registry.
    pub name: String,
    /// The components that will be inserted on each spawned entity.
    pub components: Vec<ComponentData>,
}

impl Prefab {
    /// Create a new prefab with the given name and component list.
    pub fn new(name: impl Into<String>, components: Vec<ComponentData>) -> Self {
        Self {
            name: name.into(),
            components,
        }
    }

    /// Spawn a new entity from this prefab, inserting all registered components.
    fn spawn(&self, world: &mut World) -> Entity {
        let entity = world.spawn_empty();
        for component in &self.components {
            component.insert_into(world, entity);
        }
        entity
    }
}

/// A registry of named [`Prefab`] templates.
///
/// Store this as a resource in the ECS world. Look up prefabs by name
/// and spawn entities from them.
///
/// # Example
/// ```ignore
/// let mut registry = PrefabRegistry::new();
/// registry.register(Prefab::new("soldier", vec![
///     ComponentData::Position(Vec3::ZERO),
///     ComponentData::Health(100.0),
///     ComponentData::Team(1),
/// ]));
/// world.insert_resource(registry);
///
/// let entity = world.resource_mut::<PrefabRegistry>()
///     .unwrap()
///     .spawn("soldier", &mut world)
///     .unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct PrefabRegistry {
    prefabs: HashMap<String, Prefab>,
}

impl PrefabRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            prefabs: HashMap::new(),
        }
    }

    /// Register a prefab. Overwrites any existing prefab with the same name.
    pub fn register(&mut self, prefab: Prefab) {
        self.prefabs.insert(prefab.name.clone(), prefab);
    }

    /// Remove a prefab by name, returning it if found.
    pub fn unregister(&mut self, name: &str) -> Option<Prefab> {
        self.prefabs.remove(name)
    }

    /// Check whether a prefab with the given name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.prefabs.contains_key(name)
    }

    /// Get a reference to a prefab by name.
    pub fn get(&self, name: &str) -> Option<&Prefab> {
        self.prefabs.get(name)
    }

    /// Returns an iterator over all registered prefab names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.prefabs.keys().map(String::as_str)
    }

    /// Number of registered prefabs.
    pub fn len(&self) -> usize {
        self.prefabs.len()
    }

    /// Returns `true` if no prefabs are registered.
    pub fn is_empty(&self) -> bool {
        self.prefabs.is_empty()
    }

    /// Spawn an entity from the named prefab. Returns `None` if the name
    /// is not registered.
    pub fn spawn(&self, name: &str, world: &mut World) -> Option<Entity> {
        let prefab = self.prefabs.get(name)?;
        Some(prefab.spawn(world))
    }
}

impl Default for PrefabRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for [`World`] that adds prefab spawning.
pub trait WorldPrefabExt {
    /// Spawn an entity from the named prefab registered in the world's
    /// [`PrefabRegistry`] resource. Returns `None` if the prefab name
    /// is not found or if no registry is present.
    fn spawn_prefab(&mut self, name: &str) -> Option<Entity>;
}

impl WorldPrefabExt for World {
    fn spawn_prefab(&mut self, name: &str) -> Option<Entity> {
        // Temporarily remove the registry to avoid a double borrow:
        // spawning needs &mut World, and the registry lives as a resource in World.
        let registry = self.remove_resource::<PrefabRegistry>()?;
        let entity = registry.spawn(name, self);
        self.insert_resource(registry);
        entity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefab_spawn_creates_entity_with_components() {
        let mut world = World::new();
        let prefab = Prefab::new(
            "soldier",
            vec![
                ComponentData::Position(Vec3::new(1.0, 2.0, 3.0)),
                ComponentData::Health(100.0),
                ComponentData::Team(1),
            ],
        );

        let entity = prefab.spawn(&mut world);

        let lt = world.get::<LocalTransform>(entity).unwrap();
        assert_eq!(lt.0.translation, Vec3::new(1.0, 2.0, 3.0));

        let hp = world.get::<Health>(entity).unwrap();
        assert_eq!(hp.0, 100.0);

        let team = world.get::<Team>(entity).unwrap();
        assert_eq!(team.0, 1);
    }

    #[test]
    fn registry_register_and_spawn() {
        let mut world = World::new();
        let mut registry = PrefabRegistry::new();

        registry.register(Prefab::new(
            "tree",
            vec![ComponentData::Position(Vec3::new(10.0, 0.0, 5.0))],
        ));
        assert!(registry.contains("tree"));
        assert_eq!(registry.len(), 1);

        let entity = registry.spawn("tree", &mut world).unwrap();
        let lt = world.get::<LocalTransform>(entity).unwrap();
        assert_eq!(lt.0.translation, Vec3::new(10.0, 0.0, 5.0));
    }

    #[test]
    fn registry_spawn_unknown_returns_none() {
        let mut world = World::new();
        let registry = PrefabRegistry::new();
        assert!(registry.spawn("nonexistent", &mut world).is_none());
    }

    #[test]
    fn world_prefab_ext_spawn() {
        let mut world = World::new();
        let mut registry = PrefabRegistry::new();
        registry.register(Prefab::new(
            "rock",
            vec![
                ComponentData::Position(Vec3::ZERO),
                ComponentData::Name("Boulder".into()),
            ],
        ));
        world.insert_resource(registry);

        let entity = world.spawn_prefab("rock").unwrap();
        let name = world.get::<Name>(entity).unwrap();
        assert_eq!(name.0, "Boulder");
    }

    #[test]
    fn prefab_with_all_component_variants() {
        let mut world = World::new();
        let prefab = Prefab::new(
            "hero",
            vec![
                ComponentData::Position(Vec3::new(5.0, 0.0, 5.0)),
                ComponentData::Health(200.0),
                ComponentData::Team(2),
                ComponentData::Name("Hero".into()),
                ComponentData::Speed(10.0),
                ComponentData::Damage(25.0),
            ],
        );

        let entity = prefab.spawn(&mut world);

        assert_eq!(
            world.get::<LocalTransform>(entity).unwrap().0.translation,
            Vec3::new(5.0, 0.0, 5.0)
        );
        assert_eq!(world.get::<Health>(entity).unwrap().0, 200.0);
        assert_eq!(world.get::<Team>(entity).unwrap().0, 2);
        assert_eq!(world.get::<Name>(entity).unwrap().0, "Hero");
        assert_eq!(world.get::<Speed>(entity).unwrap().0, 10.0);
        assert_eq!(world.get::<Damage>(entity).unwrap().0, 25.0);
    }

    #[test]
    fn registry_unregister() {
        let mut registry = PrefabRegistry::new();
        registry.register(Prefab::new("temp", vec![]));
        assert!(registry.contains("temp"));

        let removed = registry.unregister("temp");
        assert!(removed.is_some());
        assert!(!registry.contains("temp"));
        assert!(registry.is_empty());
    }
}
