use std::alloc::Layout;
use std::any::TypeId;
use std::collections::HashMap;

/// Marker trait for types that can be used as ECS components.
///
/// Components must be `'static + Send + Sync` for safe parallel system execution.
pub trait Component: 'static + Send + Sync {}

// Blanket impl: any 'static + Send + Sync type is a Component.
impl<T: 'static + Send + Sync> Component for T {}

/// Unique identifier for a component type.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ComponentId(pub(crate) u32);

impl ComponentId {
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }
}

/// Runtime metadata about a component type.
#[derive(Clone, Debug)]
pub struct ComponentInfo {
    pub id: ComponentId,
    pub name: &'static str,
    pub layout: Layout,
    pub type_id: TypeId,
    /// Function pointer to drop a component value in place.
    pub(crate) drop_fn: Option<unsafe fn(*mut u8)>,
}

/// Type-erased function to drop a component.
unsafe fn drop_in_place<T>(ptr: *mut u8) {
    unsafe { std::ptr::drop_in_place(ptr as *mut T) };
}

/// Registry mapping Rust types to ComponentIds and metadata.
#[derive(Default)]
pub struct ComponentStorage {
    /// Map from TypeId to ComponentId for fast lookup.
    type_to_id: HashMap<TypeId, ComponentId>,
    /// Metadata indexed by ComponentId.
    infos: Vec<ComponentInfo>,
}

impl ComponentStorage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or register a component type, returning its ID.
    pub fn register<T: Component>(&mut self) -> ComponentId {
        let type_id = TypeId::of::<T>();
        if let Some(&id) = self.type_to_id.get(&type_id) {
            return id;
        }

        let id = ComponentId(self.infos.len() as u32);
        let layout = Layout::new::<T>();
        let needs_drop = std::mem::needs_drop::<T>();

        self.infos.push(ComponentInfo {
            id,
            name: std::any::type_name::<T>(),
            layout,
            type_id,
            drop_fn: if needs_drop {
                Some(drop_in_place::<T> as unsafe fn(*mut u8))
            } else {
                None
            },
        });
        self.type_to_id.insert(type_id, id);
        id
    }

    /// Look up a component ID by its Rust type.
    #[inline]
    pub fn id_of<T: Component>(&self) -> Option<ComponentId> {
        self.type_to_id.get(&TypeId::of::<T>()).copied()
    }

    /// Get component metadata by ID.
    #[inline]
    pub fn info(&self, id: ComponentId) -> &ComponentInfo {
        &self.infos[id.0 as usize]
    }

    /// Total number of registered component types.
    #[inline]
    pub fn count(&self) -> usize {
        self.infos.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Position { x: f32, y: f32 }
    struct Velocity { dx: f32, dy: f32 }

    #[test]
    fn register_and_lookup() {
        let mut storage = ComponentStorage::new();
        let pos_id = storage.register::<Position>();
        let vel_id = storage.register::<Velocity>();

        assert_ne!(pos_id, vel_id);
        assert_eq!(storage.id_of::<Position>(), Some(pos_id));
        assert_eq!(storage.id_of::<Velocity>(), Some(vel_id));
        assert_eq!(storage.count(), 2);
    }

    #[test]
    fn idempotent_register() {
        let mut storage = ComponentStorage::new();
        let id1 = storage.register::<Position>();
        let id2 = storage.register::<Position>();
        assert_eq!(id1, id2);
        assert_eq!(storage.count(), 1);
    }

    #[test]
    fn component_info() {
        let mut storage = ComponentStorage::new();
        let id = storage.register::<Position>();
        let info = storage.info(id);
        assert_eq!(info.layout, Layout::new::<Position>());
        assert!(info.name.contains("Position"));
    }
}
