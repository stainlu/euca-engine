use std::collections::HashMap;
use std::sync::RwLock;

use crate::archetype::{Archetype, ArchetypeId};
use crate::component::{Component, ComponentId, ComponentStorage};
use crate::entity::{Entity, EntityAllocator};
use crate::event::Events;
use crate::query::QueryCache;
use crate::resource::Resources;

#[derive(Clone, Copy, Debug)]
pub(crate) struct EntityLocation {
    pub archetype_id: ArchetypeId,
    pub row: usize,
}

/// The ECS world: owns all entities, components, archetypes, resources, and events.
///
/// The `World` is the central data structure of the ECS. All entity and
/// component operations go through it. Supports change detection via
/// per-component tick tracking and both dense (archetype) and sparse
/// (`SparseSet`) component storage.
///
/// # Examples
///
/// ```
/// # use euca_ecs::World;
/// # #[derive(Debug, PartialEq)] struct Position { x: f32, y: f32 }
/// # #[derive(Debug, PartialEq)] struct Velocity { dx: f32, dy: f32 }
/// let mut world = World::new();
///
/// // Spawn entities with components
/// let entity = world.spawn(Position { x: 0.0, y: 0.0 });
/// world.insert(entity, Velocity { dx: 1.0, dy: 2.0 });
///
/// // Read and modify components
/// assert_eq!(world.get::<Position>(entity).unwrap().x, 0.0);
/// world.get_mut::<Position>(entity).unwrap().x = 10.0;
///
/// // Use resources for global state
/// world.insert_resource(42u32);
/// assert_eq!(*world.resource::<u32>().unwrap(), 42);
/// ```
pub struct World {
    pub(crate) entities: EntityAllocator,
    pub(crate) components: ComponentStorage,
    pub(crate) archetypes: Vec<Archetype>,
    archetype_index: HashMap<Vec<ComponentId>, ArchetypeId>,
    pub(crate) entity_locations: Vec<Option<EntityLocation>>,
    pub(crate) tick: u64,
    /// Increments whenever a new archetype is created. Used by Query for cache invalidation.
    pub(crate) archetype_generation: u64,
    /// Sparse component storage — HashMap<Entity, T> for rarely-used components.
    pub(crate) sparse_storage: HashMap<ComponentId, crate::sparse::SparseSet>,
    resources: Resources,
    events: Events,
    /// Shared query cache. Uses `RwLock` for interior mutability so that
    /// `Query::new_cached(&World)` can update the cache through a shared reference.
    /// `RwLock` (unlike `RefCell`) is `Sync`, required for `par_for_each`.
    pub(crate) query_cache: RwLock<QueryCache>,
}

impl World {
    /// Creates an empty world with no entities, components, or resources.
    pub fn new() -> Self {
        Self {
            entities: EntityAllocator::new(),
            components: ComponentStorage::new(),
            archetypes: Vec::new(),
            archetype_index: HashMap::new(),
            entity_locations: Vec::new(),
            tick: 0,
            archetype_generation: 0,
            sparse_storage: HashMap::new(),
            resources: Resources::new(),
            events: Events::new(),
            query_cache: crate::query::new_query_cache_lock(),
        }
    }

    /// Register a component type, returning its ID. Idempotent.
    pub fn register<T: Component>(&mut self) -> ComponentId {
        self.components.register::<T>()
    }

    /// Look up the ID of a previously registered component type.
    pub fn component_id<T: Component>(&self) -> Option<ComponentId> {
        self.components.id_of::<T>()
    }

    /// Spawn a new entity with no components.
    pub fn spawn_empty(&mut self) -> Entity {
        let entity = self.entities.allocate();
        let arch_id = self.get_or_create_archetype(&[]);
        let row =
            unsafe { self.archetypes[arch_id.0 as usize].push(entity, &[], self.tick as u32) };
        self.set_location(entity, arch_id, row);
        entity
    }

    /// Spawn a new entity with a single component.
    pub fn spawn<T: Component>(&mut self, component: T) -> Entity {
        let entity = self.entities.allocate();
        let comp_id = self.components.register::<T>();
        let arch_id = self.get_or_create_archetype(&[comp_id]);

        let row = unsafe {
            self.archetypes[arch_id.0 as usize].push(
                entity,
                &[(comp_id, &component as *const T as *const u8)],
                self.tick as u32,
            )
        };
        std::mem::forget(component);
        self.set_location(entity, arch_id, row);
        entity
    }

    /// Destroy an entity and drop all its components. Returns `false` if already dead.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }

        // Clean up any sparse components attached to this entity
        for sparse_set in self.sparse_storage.values_mut() {
            sparse_set.remove(entity);
        }

        let loc = match self.entity_locations[entity.index as usize] {
            Some(loc) => loc,
            None => return false,
        };

        let arch_idx = loc.archetype_id.0 as usize;
        let swapped = self.archetypes[arch_idx].swap_remove(loc.row);
        if let Some(swapped_entity) = swapped {
            self.entity_locations[swapped_entity.index as usize] = Some(EntityLocation {
                archetype_id: loc.archetype_id,
                row: loc.row,
            });
        }

        // If the archetype is now empty, increment archetype_generation to
        // force query cache invalidation. Without this, cached queries may
        // continue to iterate an empty archetype and miss the fact that
        // results have changed.
        if self.archetypes[arch_idx].is_empty() {
            self.archetype_generation += 1;
            crate::lock_util::write_or_recover(&self.query_cache, "World::despawn")
                .increment_generation();
        }

        self.entity_locations[entity.index as usize] = None;
        self.entities.deallocate(entity);
        true
    }

    /// Returns `true` if the entity handle is still valid.
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// Get an immutable reference to a component on an entity.
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        let comp_id = self.components.id_of::<T>()?;

        // Sparse path: component stored in SparseSet
        if self.components.info(comp_id).sparse {
            return self
                .sparse_storage
                .get(&comp_id)
                .and_then(|ss| unsafe { ss.get::<T>(entity) });
        }

        // Dense path: component stored in archetype column
        let loc = self.locate(entity)?;
        let arch = &self.archetypes[loc.archetype_id.0 as usize];
        if !arch.has_component(comp_id) {
            return None;
        }
        Some(unsafe { arch.get::<T>(comp_id, loc.row) })
    }

    /// Get a mutable reference to a component, marking it as changed at the current tick.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let comp_id = self.components.id_of::<T>()?;

        // Sparse path
        if self.components.info(comp_id).sparse {
            let tick = self.tick as u32;
            return self
                .sparse_storage
                .get_mut(&comp_id)
                .and_then(|ss| unsafe { ss.get_mut::<T>(entity, tick) });
        }

        // Dense path
        let loc = self.locate(entity)?;
        let arch = &mut self.archetypes[loc.archetype_id.0 as usize];
        if !arch.has_component(comp_id) {
            return None;
        }
        arch.set_change_tick(comp_id, loc.row, self.tick as u32);
        Some(unsafe { arch.get_mut::<T>(comp_id, loc.row) })
    }

    /// Add or overwrite a component on an entity. Returns `false` if the entity is dead.
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }

        let comp_id = self.components.register::<T>();

        // Sparse path: store in SparseSet, no archetype movement
        if self.components.info(comp_id).sparse {
            let info = self.components.info(comp_id);
            let sparse_set = self
                .sparse_storage
                .entry(comp_id)
                .or_insert_with(|| crate::sparse::SparseSet::new(info.layout, info.drop_fn));

            unsafe {
                sparse_set.insert(
                    entity,
                    &component as *const T as *const u8,
                    self.tick as u32,
                );
            }
            std::mem::forget(component);
            return true;
        }

        // Dense path: archetype column storage
        let loc = match self.locate(entity) {
            Some(loc) => loc,
            None => return false,
        };

        let old_arch = &self.archetypes[loc.archetype_id.0 as usize];

        if old_arch.has_component(comp_id) {
            unsafe {
                *self.archetypes[loc.archetype_id.0 as usize].get_mut::<T>(comp_id, loc.row) =
                    component;
            }
            return true;
        }

        let mut new_comp_ids = old_arch.component_ids.clone();
        new_comp_ids.push(comp_id);
        new_comp_ids.sort();
        let new_arch_id = self.get_or_create_archetype(&new_comp_ids);

        self.move_entity(
            entity,
            loc,
            new_arch_id,
            Some((comp_id, &component as *const T as *const u8)),
        );
        std::mem::forget(component);
        true
    }

    /// Remove a component from an entity, returning it if present.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        let comp_id = self.components.id_of::<T>()?;

        // Sparse path
        if self.components.info(comp_id).sparse
            && let Some(sparse_set) = self.sparse_storage.get_mut(&comp_id)
            && sparse_set.contains(entity)
        {
            let value = unsafe { std::ptr::read(sparse_set.get::<T>(entity)? as *const T) };
            sparse_set.remove(entity);
            return Some(value);
        } else if self.components.info(comp_id).sparse {
            return None;
        }

        // Dense path
        let loc = self.locate(entity)?;
        let old_arch = &self.archetypes[loc.archetype_id.0 as usize];
        if !old_arch.has_component(comp_id) {
            return None;
        }

        let value = unsafe { std::ptr::read(old_arch.get::<T>(comp_id, loc.row)) };

        let new_comp_ids: Vec<ComponentId> = old_arch
            .component_ids
            .iter()
            .copied()
            .filter(|&id| id != comp_id)
            .collect();
        let new_arch_id = self.get_or_create_archetype(&new_comp_ids);

        self.move_entity_excluding(entity, loc, new_arch_id, comp_id);
        Some(value)
    }

    /// Returns the number of currently alive entities.
    #[inline]
    pub fn entity_count(&self) -> u32 {
        self.entities.alive_count()
    }

    /// Returns the number of distinct archetypes in the world.
    #[inline]
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    /// Advance the world tick counter by one.
    pub fn tick(&mut self) {
        self.tick += 1;
    }

    /// Get entities whose component T was modified since `since_tick`.
    /// Essential for networking delta sync.
    pub fn changed_entities<T: Component>(&self, since_tick: u32) -> Vec<Entity> {
        let comp_id = match self.components.id_of::<T>() {
            Some(id) => id,
            None => return Vec::new(),
        };

        let mut result = Vec::new();
        for arch in &self.archetypes {
            if !arch.has_component(comp_id) {
                continue;
            }
            for row in 0..arch.len() {
                if arch.get_change_tick(comp_id, row) > since_tick {
                    result.push(arch.entities[row]);
                }
            }
        }
        result
    }

    /// Process all entities with component T in parallel using rayon.
    ///
    /// The closure receives `(Entity, &T)` for each entity. Read-only access.
    /// Uses all available CPU cores for iteration.
    pub fn par_for_each<T: Component>(&self, f: impl Fn(Entity, &T) + Send + Sync) {
        use rayon::prelude::*;

        let comp_id = match self.components.id_of::<T>() {
            Some(id) => id,
            None => return,
        };

        // Collect entity + archetype/row indices, then process in parallel
        let items: Vec<(Entity, usize, usize)> = self
            .archetypes
            .iter()
            .enumerate()
            .filter(|(_, arch)| arch.has_component(comp_id))
            .flat_map(|(arch_idx, arch)| {
                (0..arch.len()).map(move |row| (arch.entities[row], arch_idx, row))
            })
            .collect();

        items.par_iter().for_each(|(entity, arch_idx, row)| {
            // SAFETY: Each (arch_idx, row) is unique. Read-only access to distinct slots.
            let component = unsafe { self.archetypes[*arch_idx].get::<T>(comp_id, *row) };
            f(*entity, component);
        });
    }

    /// Get the change tick for a specific component on an entity.
    pub fn get_change_tick<T: Component>(&self, entity: Entity) -> Option<u32> {
        let comp_id = self.components.id_of::<T>()?;

        // Sparse path
        if self.components.info(comp_id).sparse {
            return self
                .sparse_storage
                .get(&comp_id)
                .and_then(|ss| ss.get_change_tick(entity));
        }

        // Dense path
        let loc = self.locate(entity)?;
        let arch = &self.archetypes[loc.archetype_id.0 as usize];
        if !arch.has_component(comp_id) {
            return None;
        }
        Some(arch.get_change_tick(comp_id, loc.row))
    }

    /// Register a component type as sparse. Sparse components are stored in
    /// a HashMap instead of archetype columns — use for rarely-attached
    /// components to avoid archetype explosion.
    pub fn register_sparse<T: Component>(&mut self) {
        self.components.register_sparse::<T>();
    }

    /// Returns the current world tick.
    #[inline]
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Returns the archetype generation counter (increments on new archetype creation).
    /// Used by Query for cache invalidation.
    #[inline]
    pub fn archetype_generation(&self) -> u64 {
        self.archetype_generation
    }

    // ── Resources ──

    /// Insert a singleton resource into the world. Overwrites if already present.
    pub fn insert_resource<T: Send + Sync + 'static>(&mut self, value: T) {
        self.resources.insert(value);
    }

    /// Get an immutable reference to a resource.
    pub fn resource<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.resources.get::<T>()
    }

    /// Get a mutable reference to a resource.
    pub fn resource_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.resources.get_mut::<T>()
    }

    /// Remove a resource from the world, returning it if present.
    pub fn remove_resource<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.resources.remove::<T>()
    }

    // ── Events ──

    /// Send an event into the world's double-buffered event system.
    pub fn send_event<T: Send + Sync + 'static>(&mut self, event: T) {
        self.events.send(event);
    }

    /// Read all events of type `T` from the current and previous frame.
    pub fn read_events<T: Send + Sync + 'static>(&self) -> impl Iterator<Item = &T> {
        self.events.read::<T>()
    }

    /// Swap event buffers. Call once per tick to age out old events.
    pub fn update_events(&mut self) {
        self.events.update();
    }

    // ── Internals ──

    fn locate(&self, entity: Entity) -> Option<EntityLocation> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        self.entity_locations
            .get(entity.index as usize)
            .copied()
            .flatten()
    }

    fn set_location(&mut self, entity: Entity, arch_id: ArchetypeId, row: usize) {
        let needed = entity.index as usize + 1;
        if self.entity_locations.len() < needed {
            self.entity_locations.resize(needed, None);
        }
        self.entity_locations[entity.index as usize] = Some(EntityLocation {
            archetype_id: arch_id,
            row,
        });
    }

    fn get_or_create_archetype(&mut self, comp_ids: &[ComponentId]) -> ArchetypeId {
        let mut sorted = comp_ids.to_vec();
        sorted.sort();

        if let Some(&id) = self.archetype_index.get(&sorted) {
            return id;
        }

        let id = ArchetypeId(self.archetypes.len() as u32);
        let infos: Vec<_> = sorted
            .iter()
            .map(|&cid| (cid, self.components.info(cid)))
            .collect();
        let archetype = Archetype::new(id, &infos);
        self.archetypes.push(archetype);
        self.archetype_index.insert(sorted, id);
        self.archetype_generation += 1;
        // Keep the shared QueryCache generation in sync
        crate::lock_util::write_or_recover(&self.query_cache, "World::get_or_create_archetype")
            .increment_generation();
        id
    }

    fn move_entity(
        &mut self,
        entity: Entity,
        old_loc: EntityLocation,
        new_arch_id: ArchetypeId,
        extra_component: Option<(ComponentId, *const u8)>,
    ) {
        let old_arch_idx = old_loc.archetype_id.0 as usize;
        let old_comp_ids = self.archetypes[old_arch_idx].component_ids.clone();

        // Copy component data out of old archetype
        let mut buffers: Vec<(ComponentId, Vec<u8>)> = Vec::new();
        for &cid in &old_comp_ids {
            let size = self.components.info(cid).layout.size();
            let mut buf = vec![0u8; size];
            unsafe {
                self.archetypes[old_arch_idx].read_component_raw(
                    cid,
                    old_loc.row,
                    buf.as_mut_ptr(),
                );
            }
            buffers.push((cid, buf));
        }

        // Remove from old archetype (no drop — we copied the data)
        let swapped = self.archetypes[old_arch_idx].swap_remove_no_drop(old_loc.row);
        if let Some(swapped_entity) = swapped {
            self.entity_locations[swapped_entity.index as usize] = Some(EntityLocation {
                archetype_id: old_loc.archetype_id,
                row: old_loc.row,
            });
        }

        // Build data for new archetype
        let mut new_data: Vec<(ComponentId, *const u8)> = buffers
            .iter()
            .map(|(id, buf)| (*id, buf.as_ptr()))
            .collect();
        if let Some((extra_id, extra_ptr)) = extra_component {
            new_data.push((extra_id, extra_ptr));
        }

        // Push to new archetype
        let new_arch_idx = new_arch_id.0 as usize;
        let new_row =
            unsafe { self.archetypes[new_arch_idx].push(entity, &new_data, self.tick as u32) };
        self.entity_locations[entity.index as usize] = Some(EntityLocation {
            archetype_id: new_arch_id,
            row: new_row,
        });
    }

    fn move_entity_excluding(
        &mut self,
        entity: Entity,
        old_loc: EntityLocation,
        new_arch_id: ArchetypeId,
        exclude_id: ComponentId,
    ) {
        let old_arch_idx = old_loc.archetype_id.0 as usize;
        let old_comp_ids = self.archetypes[old_arch_idx].component_ids.clone();

        // Copy component data (excluding the removed component)
        let mut buffers: Vec<(ComponentId, Vec<u8>)> = Vec::new();
        for &cid in &old_comp_ids {
            if cid == exclude_id {
                continue;
            }
            let size = self.components.info(cid).layout.size();
            let mut buf = vec![0u8; size];
            unsafe {
                self.archetypes[old_arch_idx].read_component_raw(
                    cid,
                    old_loc.row,
                    buf.as_mut_ptr(),
                );
            }
            buffers.push((cid, buf));
        }

        // Remove from old archetype (no drop — we either copied or already read the excluded one)
        let swapped = self.archetypes[old_arch_idx].swap_remove_no_drop(old_loc.row);
        if let Some(swapped_entity) = swapped {
            self.entity_locations[swapped_entity.index as usize] = Some(EntityLocation {
                archetype_id: old_loc.archetype_id,
                row: old_loc.row,
            });
        }

        // Push to new archetype
        let new_data: Vec<(ComponentId, *const u8)> = buffers
            .iter()
            .map(|(id, buf)| (*id, buf.as_ptr()))
            .collect();
        let new_arch_idx = new_arch_id.0 as usize;
        let new_row =
            unsafe { self.archetypes[new_arch_idx].push(entity, &new_data, self.tick as u32) };
        self.entity_locations[entity.index as usize] = Some(EntityLocation {
            archetype_id: new_arch_id,
            row: new_row,
        });
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

// ── UnsafeWorldCell: split-borrow of World for system parameter extraction ──

use std::marker::PhantomData;

#[allow(dead_code)] // Infrastructure for future system parameter extraction (#2) and parallel scheduling (#3)
/// Unsafe view into a [`World`] that bypasses borrow checking.
///
/// Created from `&mut World` during system execution. Allows multiple
/// system parameters to borrow disjoint parts of the world simultaneously.
///
/// # Safety
/// The caller must ensure that no two users access the same resource or
/// component column mutably at the same time.
#[derive(Copy, Clone)]
pub struct UnsafeWorldCell<'w> {
    world: *mut World,
    _marker: PhantomData<(&'w World, &'w mut World)>,
}

// SAFETY: Created from &mut World (which is Send+Sync).
// Access safety is the caller's responsibility via SystemAccess validation.
unsafe impl Send for UnsafeWorldCell<'_> {}
unsafe impl Sync for UnsafeWorldCell<'_> {}

#[allow(dead_code)]
impl<'w> UnsafeWorldCell<'w> {
    /// Create from `&mut World`.
    ///
    /// # Safety
    /// Caller must not use the original `&mut World` while this cell exists.
    #[inline]
    pub(crate) unsafe fn new(world: &'w mut World) -> Self {
        Self {
            world: world as *mut World,
            _marker: PhantomData,
        }
    }

    /// Get an immutable reference to the world.
    ///
    /// # Safety
    /// Caller must ensure no mutable references to world data are live.
    #[inline]
    pub unsafe fn world(self) -> &'w World {
        unsafe { &*self.world }
    }

    /// Get an immutable reference to a resource.
    ///
    /// # Safety
    /// Caller must ensure no mutable reference to this resource type exists.
    #[inline]
    pub unsafe fn resource<T: Send + Sync + 'static>(self) -> Option<&'w T> {
        unsafe { (*self.world).resources.get::<T>() }
    }

    /// Get a mutable reference to a resource.
    ///
    /// # Safety
    /// Caller must ensure exclusive access to this resource type.
    #[inline]
    pub unsafe fn resource_mut<T: Send + Sync + 'static>(self) -> Option<&'w mut T> {
        unsafe { (*self.world).resources.get_mut::<T>() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[derive(Debug, Clone, PartialEq)]
    struct Name(String);

    #[test]
    fn spawn_and_get() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1.0, y: 2.0 });
        assert!(world.is_alive(entity));
        assert_eq!(world.entity_count(), 1);
        assert_eq!(
            world.get::<Position>(entity).unwrap(),
            &Position { x: 1.0, y: 2.0 }
        );
    }

    #[test]
    fn spawn_and_despawn() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1.0, y: 2.0 });
        assert!(world.despawn(entity));
        assert!(!world.is_alive(entity));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn get_mut() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1.0, y: 2.0 });
        world.get_mut::<Position>(entity).unwrap().x = 10.0;
        assert_eq!(world.get::<Position>(entity).unwrap().x, 10.0);
    }

    #[test]
    fn insert_component() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1.0, y: 2.0 });
        world.insert(entity, Velocity { dx: 3.0, dy: 4.0 });
        assert_eq!(
            world.get::<Position>(entity).unwrap(),
            &Position { x: 1.0, y: 2.0 }
        );
        assert_eq!(
            world.get::<Velocity>(entity).unwrap(),
            &Velocity { dx: 3.0, dy: 4.0 }
        );
    }

    #[test]
    fn remove_component() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1.0, y: 2.0 });
        world.insert(entity, Velocity { dx: 3.0, dy: 4.0 });
        let removed = world.remove::<Velocity>(entity);
        assert_eq!(removed, Some(Velocity { dx: 3.0, dy: 4.0 }));
        assert!(world.get::<Velocity>(entity).is_none());
        assert_eq!(
            world.get::<Position>(entity).unwrap(),
            &Position { x: 1.0, y: 2.0 }
        );
    }

    #[test]
    fn overwrite_component() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1.0, y: 2.0 });
        world.insert(entity, Position { x: 99.0, y: 99.0 });
        assert_eq!(
            world.get::<Position>(entity).unwrap(),
            &Position { x: 99.0, y: 99.0 }
        );
    }

    #[test]
    fn drop_called_on_despawn() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static DROP_COUNT: AtomicU32 = AtomicU32::new(0);

        #[derive(Debug)]
        struct DropTracker;
        impl Drop for DropTracker {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        let mut world = World::new();
        let entity = world.spawn(DropTracker);
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 0);
        world.despawn(entity);
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn multiple_entities_same_archetype() {
        let mut world = World::new();
        let e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        let e2 = world.spawn(Position { x: 2.0, y: 2.0 });
        let e3 = world.spawn(Position { x: 3.0, y: 3.0 });

        assert_eq!(world.entity_count(), 3);
        assert_eq!(world.archetype_count(), 1);

        world.despawn(e2);
        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.get::<Position>(e1).unwrap().x, 1.0);
        assert_eq!(world.get::<Position>(e3).unwrap().x, 3.0);
    }

    #[test]
    fn string_component_no_leak() {
        let mut world = World::new();
        let e = world.spawn(Name("hello".into()));
        assert_eq!(world.get::<Name>(e).unwrap().0, "hello");
        world.despawn(e);
    }

    #[test]
    fn change_detection_tracks_mutations() {
        let mut world = World::new();
        let e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        let e2 = world.spawn(Position { x: 2.0, y: 2.0 });

        // Both spawned at tick 0
        assert_eq!(world.get_change_tick::<Position>(e1), Some(0));
        assert_eq!(world.get_change_tick::<Position>(e2), Some(0));

        // Advance tick and modify only e1
        world.tick();
        world.get_mut::<Position>(e1).unwrap().x = 99.0;

        // e1 should be at tick 1, e2 still at tick 0
        assert_eq!(world.get_change_tick::<Position>(e1), Some(1));
        assert_eq!(world.get_change_tick::<Position>(e2), Some(0));

        // changed_entities should return only e1
        let changed = world.changed_entities::<Position>(0);
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0], e1);
    }

    #[test]
    fn changed_entities_returns_all_when_since_zero() {
        let mut world = World::new();
        world.tick(); // tick = 1
        let _e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        let _e2 = world.spawn(Position { x: 2.0, y: 2.0 });

        // Both spawned at tick 1, query since tick 0 → both returned
        let changed = world.changed_entities::<Position>(0);
        assert_eq!(changed.len(), 2);
    }

    #[test]
    fn changed_entities_empty_when_nothing_changed() {
        let mut world = World::new();
        let _e1 = world.spawn(Position { x: 1.0, y: 1.0 });

        // Spawned at tick 0, query since tick 0 → nothing changed (tick must be > since)
        let changed = world.changed_entities::<Position>(0);
        assert_eq!(changed.len(), 0);
    }

    #[test]
    fn despawn_last_entity_invalidates_cache() {
        use crate::Query;

        let mut world = World::new();
        let entity = world.spawn(Position { x: 1.0, y: 2.0 });

        // Query should find the entity.
        let results: Vec<_> = {
            let q = Query::<&Position>::new(&world);
            q.iter().collect()
        };
        assert_eq!(results.len(), 1);

        // Despawn the only entity in its archetype, making it empty.
        world.despawn(entity);

        // A fresh query must reflect the empty world.
        let results: Vec<_> = {
            let q = Query::<&Position>::new(&world);
            q.iter().collect()
        };
        assert!(
            results.is_empty(),
            "Query should return empty results after despawning the last entity"
        );
    }
}
