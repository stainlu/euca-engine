//! Sparse set storage for rarely-used ECS components.
//!
//! Components marked as sparse are stored in a HashMap<Entity, data> instead of
//! archetype columns. This avoids archetype explosion when attaching rare
//! components (tags, network IDs, debug markers) to a few entities.
//!
//! # Safety invariants
//! - `data` pointer is valid for `len` elements of `item_layout` size
//! - `entities` Vec is parallel to data slots (same length, same indexing)
//! - `change_ticks` Vec is parallel to data slots
//! - `entity_to_slot` maps entities to valid slot indices in [0, len)

use std::alloc::{self, Layout};
use std::collections::HashMap;
use std::ptr;

use crate::entity::Entity;

/// Type-erased sparse storage for a single component type.
///
/// Entities are mapped to dense packed slots. Insertion is O(1) amortized,
/// lookup is O(1) via HashMap, removal is O(1) via swap-remove.
pub struct SparseSet {
    entity_to_slot: HashMap<Entity, usize>,
    entities: Vec<Entity>,
    data: *mut u8,
    item_layout: Layout,
    len: usize,
    capacity: usize,
    drop_fn: Option<unsafe fn(*mut u8)>,
    change_ticks: Vec<u32>,
}

// SAFETY: SparseSet manages its own memory. Access is controlled by the World.
unsafe impl Send for SparseSet {}
unsafe impl Sync for SparseSet {}

impl SparseSet {
    /// Create a new sparse set for a component with the given layout and drop function.
    pub fn new(item_layout: Layout, drop_fn: Option<unsafe fn(*mut u8)>) -> Self {
        Self {
            entity_to_slot: HashMap::new(),
            entities: Vec::new(),
            data: ptr::null_mut(),
            item_layout,
            len: 0,
            capacity: 0,
            drop_fn,
            change_ticks: Vec::new(),
        }
    }

    /// Number of entities in this sparse set.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this sparse set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if an entity has a value in this sparse set.
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.entity_to_slot.contains_key(&entity)
    }

    /// Insert or overwrite a component value for an entity.
    ///
    /// # Safety
    /// `src` must point to a valid value of the component type matching `item_layout`.
    pub unsafe fn insert(&mut self, entity: Entity, src: *const u8, tick: u32) {
        if let Some(&slot) = self.entity_to_slot.get(&entity) {
            // Overwrite existing slot
            let size = self.item_layout.size();
            if size > 0 {
                let dst = self.data.add(slot * size);
                if let Some(drop_fn) = self.drop_fn {
                    // SAFETY: Dropping the old value before overwriting.
                    (drop_fn)(dst);
                }
                // SAFETY: src is valid, dst is within allocated bounds.
                ptr::copy_nonoverlapping(src, dst, size);
            }
            self.change_ticks[slot] = tick;
        } else {
            // New slot — grow if needed
            self.grow_if_needed();

            let slot = self.len;
            let size = self.item_layout.size();
            if size > 0 {
                // SAFETY: slot is within capacity, src is valid.
                let dst = self.data.add(slot * size);
                ptr::copy_nonoverlapping(src, dst, size);
            }

            self.entities.push(entity);
            self.change_ticks.push(tick);
            self.entity_to_slot.insert(entity, slot);
            self.len += 1;
        }
    }

    /// Get an immutable reference to the component for an entity.
    ///
    /// # Safety
    /// `T` must match the component type this sparse set was created for.
    #[inline]
    pub unsafe fn get<T: 'static>(&self, entity: Entity) -> Option<&T> {
        let &slot = self.entity_to_slot.get(&entity)?;
        let size = self.item_layout.size();
        if size == 0 {
            // SAFETY: Zero-sized type — any aligned pointer is valid.
            Some(&*(ptr::NonNull::<T>::dangling().as_ptr()))
        } else {
            // SAFETY: slot is valid, T matches item_layout.
            Some(&*(self.data.add(slot * size) as *const T))
        }
    }

    /// Get a mutable reference to the component for an entity, updating the change tick.
    ///
    /// # Safety
    /// `T` must match the component type this sparse set was created for.
    /// Caller must ensure exclusive access.
    #[inline]
    pub unsafe fn get_mut<T: 'static>(&mut self, entity: Entity, tick: u32) -> Option<&mut T> {
        let &slot = self.entity_to_slot.get(&entity)?;
        self.change_ticks[slot] = tick;
        let size = self.item_layout.size();
        if size == 0 {
            // SAFETY: Zero-sized type — any aligned pointer is valid.
            Some(unsafe { &mut *(ptr::NonNull::<T>::dangling().as_ptr()) })
        } else {
            // SAFETY: slot is valid, T matches item_layout, we have &mut self.
            Some(unsafe { &mut *(self.data.add(slot * size) as *mut T) })
        }
    }

    /// Get the change tick for an entity's component.
    #[inline]
    pub fn get_change_tick(&self, entity: Entity) -> Option<u32> {
        let &slot = self.entity_to_slot.get(&entity)?;
        Some(self.change_ticks[slot])
    }

    /// Remove an entity's component, returning true if it was present.
    /// Uses swap-remove for O(1) performance.
    pub fn remove(&mut self, entity: Entity) -> bool {
        let slot = match self.entity_to_slot.remove(&entity) {
            Some(s) => s,
            None => return false,
        };

        let last = self.len - 1;
        let size = self.item_layout.size();

        // Drop the removed value
        if size > 0
            && let Some(drop_fn) = self.drop_fn
        {
            // SAFETY: slot is valid, component needs dropping.
            unsafe { (drop_fn)(self.data.add(slot * size)) };
        }

        if slot != last {
            // Swap-remove: move last element into the vacated slot
            if size > 0 {
                unsafe {
                    let src = self.data.add(last * size);
                    let dst = self.data.add(slot * size);
                    ptr::copy_nonoverlapping(src, dst, size);
                }
            }
            let swapped_entity = self.entities[last];
            self.entities[slot] = swapped_entity;
            self.change_ticks[slot] = self.change_ticks[last];
            self.entity_to_slot.insert(swapped_entity, slot);
        }

        self.entities.pop();
        self.change_ticks.pop();
        self.len -= 1;
        true
    }

    fn grow_if_needed(&mut self) {
        if self.len >= self.capacity {
            let new_capacity = if self.capacity == 0 {
                4
            } else {
                self.capacity * 2
            };
            let size = self.item_layout.size();
            if size > 0 {
                let new_layout =
                    Layout::from_size_align(size * new_capacity, self.item_layout.align())
                        .expect("invalid layout");

                let new_data = if self.data.is_null() {
                    // SAFETY: new_layout has valid size and alignment.
                    unsafe { alloc::alloc(new_layout) }
                } else {
                    let old_layout =
                        Layout::from_size_align(size * self.capacity, self.item_layout.align())
                            .expect("old sparse set layout invalid");
                    // SAFETY: self.data was allocated with old_layout.
                    unsafe { alloc::realloc(self.data, old_layout, new_layout.size()) }
                };

                if new_data.is_null() {
                    alloc::handle_alloc_error(new_layout);
                }
                self.data = new_data;
            }
            self.capacity = new_capacity;
        }
    }

    /// Deep-clone this sparse set using the component's type-erased clone
    /// function. Produces a new `SparseSet` with the same entity mapping
    /// and cloned component data, owning its memory independently.
    ///
    /// `new.len` is incremented after each successful clone so that a
    /// panic inside `clone_fn` leaves the new sparse set in a consistent
    /// state for its Drop impl.
    ///
    /// # Safety
    /// `clone_fn` must match the component type this sparse set stores.
    pub(crate) unsafe fn clone_with(
        &self,
        clone_fn: unsafe fn(*const u8, *mut u8),
    ) -> SparseSet {
        let mut new = SparseSet {
            entity_to_slot: self.entity_to_slot.clone(),
            entities: self.entities.clone(),
            data: ptr::null_mut(),
            item_layout: self.item_layout,
            len: 0,
            capacity: 0,
            drop_fn: self.drop_fn,
            change_ticks: Vec::with_capacity(self.len),
        };
        let size = self.item_layout.size();
        if self.capacity > 0 && size > 0 {
            let layout = Layout::from_size_align(size * self.capacity, self.item_layout.align())
                .expect("sparse set clone layout invalid");
            // SAFETY: layout has valid size and alignment.
            let new_data = unsafe { alloc::alloc(layout) };
            if new_data.is_null() {
                alloc::handle_alloc_error(layout);
            }
            new.data = new_data;
            new.capacity = self.capacity;
        } else if size == 0 {
            new.capacity = self.capacity;
        }
        for i in 0..self.len {
            if size > 0 {
                // SAFETY: i < self.len <= self.capacity, data pointers valid.
                unsafe {
                    let src = self.data.add(i * size);
                    let dst = new.data.add(i * size);
                    clone_fn(src, dst);
                }
            } else {
                let dangling = ptr::NonNull::<u8>::dangling().as_ptr();
                unsafe {
                    clone_fn(dangling as *const u8, dangling);
                }
            }
            new.change_ticks.push(self.change_ticks[i]);
            new.len += 1;
        }
        new
    }
}

impl Drop for SparseSet {
    fn drop(&mut self) {
        let size = self.item_layout.size();
        if let Some(drop_fn) = self.drop_fn {
            for i in 0..self.len {
                // SAFETY: Each slot contains a valid value that needs dropping.
                unsafe { (drop_fn)(self.data.add(i * size)) };
            }
        }
        if size > 0 && !self.data.is_null() {
            let layout = Layout::from_size_align(size * self.capacity, self.item_layout.align())
                .expect("sparse set dealloc layout invalid");
            // SAFETY: self.data was allocated with this layout.
            unsafe { alloc::dealloc(self.data, layout) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;

    #[derive(Debug, Clone, PartialEq)]
    struct Tag(u32);

    fn tag_layout() -> Layout {
        Layout::new::<Tag>()
    }

    fn make_entity(index: u32) -> Entity {
        Entity::from_raw(index, 0)
    }

    #[test]
    fn insert_and_get() {
        let mut set = SparseSet::new(tag_layout(), None);
        let e = make_entity(0);
        let tag = Tag(42);

        unsafe { set.insert(e, &tag as *const Tag as *const u8, 0) };
        std::mem::forget(tag);

        let result: &Tag = unsafe { set.get(e).unwrap() };
        assert_eq!(result.0, 42);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn overwrite() {
        let mut set = SparseSet::new(tag_layout(), None);
        let e = make_entity(0);

        let tag1 = Tag(1);
        unsafe { set.insert(e, &tag1 as *const Tag as *const u8, 0) };
        std::mem::forget(tag1);

        let tag2 = Tag(2);
        unsafe { set.insert(e, &tag2 as *const Tag as *const u8, 1) };
        std::mem::forget(tag2);

        let result: &Tag = unsafe { set.get(e).unwrap() };
        assert_eq!(result.0, 2);
        assert_eq!(set.len(), 1); // Still 1 entity
    }

    #[test]
    fn remove() {
        let mut set = SparseSet::new(tag_layout(), None);
        let e = make_entity(0);
        let tag = Tag(42);
        unsafe { set.insert(e, &tag as *const Tag as *const u8, 0) };
        std::mem::forget(tag);

        assert!(set.remove(e));
        assert!(!set.contains(e));
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn swap_remove_preserves_other() {
        let mut set = SparseSet::new(tag_layout(), None);
        let e0 = make_entity(0);
        let e1 = make_entity(1);
        let e2 = make_entity(2);

        for (e, v) in [(e0, 10), (e1, 20), (e2, 30)] {
            let tag = Tag(v);
            unsafe { set.insert(e, &tag as *const Tag as *const u8, 0) };
            std::mem::forget(tag);
        }

        // Remove e0 (first) — e2 should swap into slot 0
        set.remove(e0);
        assert_eq!(set.len(), 2);
        assert!(!set.contains(e0));

        let v1: &Tag = unsafe { set.get(e1).unwrap() };
        assert_eq!(v1.0, 20);
        let v2: &Tag = unsafe { set.get(e2).unwrap() };
        assert_eq!(v2.0, 30);
    }

    #[test]
    fn change_tick() {
        let mut set = SparseSet::new(tag_layout(), None);
        let e = make_entity(0);
        let tag = Tag(42);
        unsafe { set.insert(e, &tag as *const Tag as *const u8, 5) };
        std::mem::forget(tag);

        assert_eq!(set.get_change_tick(e), Some(5));

        // Mutable access should update tick
        let _: &mut Tag = unsafe { set.get_mut(e, 10).unwrap() };
        assert_eq!(set.get_change_tick(e), Some(10));
    }
}
