use std::collections::HashMap;

use crate::archetype::{Archetype, ArchetypeId};
use crate::component::{Component, ComponentId, ComponentStorage};
use crate::entity::{Entity, EntityAllocator};
use crate::event::Events;
use crate::resource::Resources;

#[derive(Clone, Copy, Debug)]
pub(crate) struct EntityLocation {
    pub archetype_id: ArchetypeId,
    pub row: usize,
}

/// The ECS world: owns all entities, components, and archetypes.
pub struct World {
    pub(crate) entities: EntityAllocator,
    pub(crate) components: ComponentStorage,
    pub(crate) archetypes: Vec<Archetype>,
    archetype_index: HashMap<Vec<ComponentId>, ArchetypeId>,
    pub(crate) entity_locations: Vec<Option<EntityLocation>>,
    pub(crate) tick: u64,
    resources: Resources,
    events: Events,
}

impl World {
    pub fn new() -> Self {
        Self {
            entities: EntityAllocator::new(),
            components: ComponentStorage::new(),
            archetypes: Vec::new(),
            archetype_index: HashMap::new(),
            entity_locations: Vec::new(),
            tick: 0,
            resources: Resources::new(),
            events: Events::new(),
        }
    }

    pub fn register<T: Component>(&mut self) -> ComponentId {
        self.components.register::<T>()
    }

    pub fn component_id<T: Component>(&self) -> Option<ComponentId> {
        self.components.id_of::<T>()
    }

    pub fn spawn_empty(&mut self) -> Entity {
        let entity = self.entities.allocate();
        let arch_id = self.get_or_create_archetype(&[]);
        let row =
            unsafe { self.archetypes[arch_id.0 as usize].push(entity, &[], self.tick as u32) };
        self.set_location(entity, arch_id, row);
        entity
    }

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

    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        let loc = match self.entity_locations[entity.index as usize] {
            Some(loc) => loc,
            None => return false,
        };

        let swapped = self.archetypes[loc.archetype_id.0 as usize].swap_remove(loc.row);
        if let Some(swapped_entity) = swapped {
            self.entity_locations[swapped_entity.index as usize] = Some(EntityLocation {
                archetype_id: loc.archetype_id,
                row: loc.row,
            });
        }

        self.entity_locations[entity.index as usize] = None;
        self.entities.deallocate(entity);
        true
    }

    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        let loc = self.locate(entity)?;
        let comp_id = self.components.id_of::<T>()?;
        let arch = &self.archetypes[loc.archetype_id.0 as usize];
        if !arch.has_component(comp_id) {
            return None;
        }
        Some(unsafe { arch.get::<T>(comp_id, loc.row) })
    }

    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let loc = self.locate(entity)?;
        let comp_id = self.components.id_of::<T>()?;
        let arch = &mut self.archetypes[loc.archetype_id.0 as usize];
        if !arch.has_component(comp_id) {
            return None;
        }
        // Mark as changed at current tick
        arch.set_change_tick(comp_id, loc.row, self.tick as u32);
        Some(unsafe { arch.get_mut::<T>(comp_id, loc.row) })
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) -> bool {
        let loc = match self.locate(entity) {
            Some(loc) => loc,
            None => return false,
        };

        let comp_id = self.components.register::<T>();
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

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        let loc = self.locate(entity)?;
        let comp_id = self.components.id_of::<T>()?;
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

    #[inline]
    pub fn entity_count(&self) -> u32 {
        self.entities.alive_count()
    }

    #[inline]
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

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

    /// Get the change tick for a specific component on an entity.
    pub fn get_change_tick<T: Component>(&self, entity: Entity) -> Option<u32> {
        let loc = self.locate(entity)?;
        let comp_id = self.components.id_of::<T>()?;
        let arch = &self.archetypes[loc.archetype_id.0 as usize];
        if !arch.has_component(comp_id) {
            return None;
        }
        Some(arch.get_change_tick(comp_id, loc.row))
    }

    #[inline]
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    // ── Resources ──

    pub fn insert_resource<T: Send + Sync + 'static>(&mut self, value: T) {
        self.resources.insert(value);
    }

    pub fn resource<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.resources.get::<T>()
    }

    pub fn resource_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.resources.get_mut::<T>()
    }

    pub fn remove_resource<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.resources.remove::<T>()
    }

    // ── Events ──

    pub fn send_event<T: Send + Sync + 'static>(&mut self, event: T) {
        self.events.send(event);
    }

    pub fn read_events<T: Send + Sync + 'static>(&self) -> impl Iterator<Item = &T> {
        self.events.read::<T>()
    }

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
}
