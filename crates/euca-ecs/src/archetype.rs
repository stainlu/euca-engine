use std::alloc::{self, Layout};
use std::collections::HashMap;
use std::ptr;

use crate::component::{ComponentId, ComponentInfo};
use crate::entity::Entity;

/// Unique identifier for an archetype.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ArchetypeId(pub(crate) u32);

/// A column of component data stored contiguously (SoA layout).
struct Column {
    data: *mut u8,
    item_layout: Layout,
    len: usize,
    capacity: usize,
    drop_fn: Option<unsafe fn(*mut u8)>,
}

unsafe impl Send for Column {}
unsafe impl Sync for Column {}

impl Column {
    fn new(info: &ComponentInfo) -> Self {
        Self {
            data: ptr::null_mut(),
            item_layout: info.layout,
            len: 0,
            capacity: 0,
            drop_fn: info.drop_fn,
        }
    }

    fn grow_if_needed(&mut self) {
        if self.len < self.capacity {
            return;
        }
        let new_cap = if self.capacity == 0 { 8 } else { self.capacity * 2 };
        self.realloc(new_cap);
    }

    fn realloc(&mut self, new_capacity: usize) {
        if self.item_layout.size() == 0 {
            self.capacity = new_capacity;
            return;
        }
        let new_layout = Layout::from_size_align(
            self.item_layout.size() * new_capacity,
            self.item_layout.align(),
        )
        .expect("invalid layout");

        let new_data = if self.data.is_null() {
            unsafe { alloc::alloc(new_layout) }
        } else {
            let old_layout = Layout::from_size_align(
                self.item_layout.size() * self.capacity,
                self.item_layout.align(),
            )
            .unwrap();
            unsafe { alloc::realloc(self.data, old_layout, new_layout.size()) }
        };

        if new_data.is_null() {
            alloc::handle_alloc_error(new_layout);
        }
        self.data = new_data;
        self.capacity = new_capacity;
    }

    /// # Safety
    /// `src` must point to a valid value of the component type.
    unsafe fn push(&mut self, src: *const u8) {
        let size = self.item_layout.size();
        if size > 0 {
            self.grow_if_needed();
            unsafe {
                let dst = self.data.add(self.len * size);
                ptr::copy_nonoverlapping(src, dst, size);
            }
        }
        self.len += 1;
    }

    /// Swap-remove element at `index`, dropping it.
    /// Returns `true` if a swap occurred.
    ///
    /// # Safety
    /// `index` must be < `self.len`.
    unsafe fn swap_remove(&mut self, index: usize) -> bool {
        let size = self.item_layout.size();
        self.len -= 1;
        let swapped = index != self.len;
        unsafe {
            if size > 0 {
                let removed = self.data.add(index * size);
                if let Some(drop_fn) = self.drop_fn {
                    (drop_fn)(removed);
                }
                if swapped {
                    let last = self.data.add(self.len * size);
                    ptr::copy_nonoverlapping(last, removed, size);
                }
            } else if let Some(drop_fn) = self.drop_fn {
                // ZST with Drop impl: call drop on a dangling but aligned pointer
                let ptr = std::ptr::NonNull::<u8>::dangling().as_ptr();
                (drop_fn)(ptr);
            }
        }
        swapped
    }

    /// Swap-remove element at `index` WITHOUT dropping it (caller takes ownership of data).
    /// Returns `true` if a swap occurred.
    ///
    /// # Safety
    /// `index` must be < `self.len`. Caller must ensure the removed data is properly handled.
    unsafe fn swap_remove_no_drop(&mut self, index: usize) -> bool {
        let size = self.item_layout.size();
        self.len -= 1;
        let swapped = index != self.len;
        unsafe {
            if swapped && size > 0 {
                let removed = self.data.add(index * size);
                let last = self.data.add(self.len * size);
                ptr::copy_nonoverlapping(last, removed, size);
            }
        }
        swapped
    }

    /// # Safety
    /// `index` must be < `self.len`.
    #[inline]
    unsafe fn get(&self, index: usize) -> *const u8 {
        let size = self.item_layout.size();
        if size == 0 {
            std::ptr::NonNull::<u8>::dangling().as_ptr() as *const u8
        } else {
            unsafe { self.data.add(index * size) }
        }
    }

    /// # Safety
    /// `index` must be < `self.len`.
    #[inline]
    unsafe fn get_mut(&self, index: usize) -> *mut u8 {
        let size = self.item_layout.size();
        if size == 0 {
            std::ptr::NonNull::<u8>::dangling().as_ptr()
        } else {
            unsafe { self.data.add(index * size) }
        }
    }

    /// Copy raw bytes of element at `index` into `dst`.
    ///
    /// # Safety
    /// `index` must be valid. `dst` must have room for `item_layout.size()` bytes.
    unsafe fn read_raw(&self, index: usize, dst: *mut u8) {
        let size = self.item_layout.size();
        if size > 0 {
            unsafe {
                let src = self.data.add(index * size);
                ptr::copy_nonoverlapping(src, dst, size);
            }
        }
    }
}

impl Drop for Column {
    fn drop(&mut self) {
        if let Some(drop_fn) = self.drop_fn {
            for i in 0..self.len {
                unsafe {
                    let ptr = self.data.add(i * self.item_layout.size());
                    (drop_fn)(ptr);
                }
            }
        }
        if !self.data.is_null() && self.item_layout.size() > 0 {
            let layout = Layout::from_size_align(
                self.item_layout.size() * self.capacity,
                self.item_layout.align(),
            )
            .unwrap();
            unsafe { alloc::dealloc(self.data, layout) };
        }
    }
}

/// An archetype stores all entities that have the exact same set of component types.
/// Components are stored in struct-of-arrays (SoA) layout for cache-friendly iteration.
pub struct Archetype {
    pub(crate) id: ArchetypeId,
    pub(crate) component_ids: Vec<ComponentId>,
    pub(crate) entities: Vec<Entity>,
    columns: HashMap<ComponentId, Column>,
}

impl Archetype {
    pub(crate) fn new(id: ArchetypeId, component_infos: &[(ComponentId, &ComponentInfo)]) -> Self {
        let mut component_ids: Vec<ComponentId> =
            component_infos.iter().map(|(id, _)| *id).collect();
        component_ids.sort();

        let columns = component_infos
            .iter()
            .map(|(id, info)| (*id, Column::new(info)))
            .collect();

        Self {
            id,
            component_ids,
            entities: Vec::new(),
            columns,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    #[inline]
    pub fn has_component(&self, id: ComponentId) -> bool {
        self.component_ids.binary_search(&id).is_ok()
    }

    /// # Safety
    /// Component data pointers must be valid and match the archetype's component types.
    pub(crate) unsafe fn push(
        &mut self,
        entity: Entity,
        component_data: &[(ComponentId, *const u8)],
    ) -> usize {
        let row = self.entities.len();
        self.entities.push(entity);
        for (id, data) in component_data {
            unsafe { self.columns.get_mut(id).unwrap().push(*data) };
        }
        row
    }

    /// Remove entity at `row` using swap-remove, dropping component data.
    pub(crate) fn swap_remove(&mut self, row: usize) -> Option<Entity> {
        self.entities.swap_remove(row);
        let mut swapped = false;
        for column in self.columns.values_mut() {
            unsafe {
                swapped = column.swap_remove(row);
            }
        }
        if swapped {
            Some(self.entities[row])
        } else {
            None
        }
    }

    /// Remove entity at `row` using swap-remove WITHOUT dropping component data.
    /// Used when moving entities between archetypes (data is copied first).
    pub(crate) fn swap_remove_no_drop(&mut self, row: usize) -> Option<Entity> {
        self.entities.swap_remove(row);
        let mut swapped = false;
        for column in self.columns.values_mut() {
            unsafe {
                swapped = column.swap_remove_no_drop(row);
            }
        }
        if swapped {
            Some(self.entities[row])
        } else {
            None
        }
    }

    /// Read raw component bytes at `row` for the given component.
    /// # Safety
    /// Component must exist in this archetype and row must be valid.
    pub(crate) unsafe fn read_component_raw(
        &self,
        component_id: ComponentId,
        row: usize,
        dst: *mut u8,
    ) {
        unsafe {
            self.columns
                .get(&component_id)
                .unwrap()
                .read_raw(row, dst);
        }
    }

    /// # Safety
    /// `T` must match the component type for `component_id`.
    #[inline]
    pub(crate) unsafe fn get<T: 'static>(
        &self,
        component_id: ComponentId,
        row: usize,
    ) -> &T {
        let column = self.columns.get(&component_id).unwrap();
        unsafe { &*(column.get(row) as *const T) }
    }

    /// # Safety
    /// `T` must match the component type for `component_id`.
    #[inline]
    pub(crate) unsafe fn get_mut<T: 'static>(
        &self,
        component_id: ComponentId,
        row: usize,
    ) -> &mut T {
        let column = self.columns.get(&component_id).unwrap();
        unsafe { &mut *(column.get_mut(row) as *mut T) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::ComponentStorage;

    #[derive(Debug, Clone, PartialEq)]
    struct Pos { x: f32, y: f32 }

    #[derive(Debug, Clone, PartialEq)]
    struct Vel { dx: f32, dy: f32 }

    fn setup() -> (ComponentStorage, ComponentId, ComponentId) {
        let mut cs = ComponentStorage::new();
        let p = cs.register::<Pos>();
        let v = cs.register::<Vel>();
        (cs, p, v)
    }

    #[test]
    fn push_and_read() {
        let (cs, pid, vid) = setup();
        let mut arch = Archetype::new(ArchetypeId(0), &[(pid, cs.info(pid)), (vid, cs.info(vid))]);

        let entity = Entity::from_raw(0, 0);
        let pos = Pos { x: 1.0, y: 2.0 };
        let vel = Vel { dx: 3.0, dy: 4.0 };

        unsafe {
            arch.push(entity, &[
                (pid, &pos as *const Pos as *const u8),
                (vid, &vel as *const Vel as *const u8),
            ]);
            assert_eq!(arch.get::<Pos>(pid, 0), &Pos { x: 1.0, y: 2.0 });
            assert_eq!(arch.get::<Vel>(vid, 0), &Vel { dx: 3.0, dy: 4.0 });
        }
        assert_eq!(arch.len(), 1);
    }

    #[test]
    fn swap_remove_middle() {
        let (cs, pid, _) = setup();
        let mut arch = Archetype::new(ArchetypeId(0), &[(pid, cs.info(pid))]);

        let e0 = Entity::from_raw(0, 0);
        let e1 = Entity::from_raw(1, 0);
        let e2 = Entity::from_raw(2, 0);

        unsafe {
            let p0 = Pos { x: 0.0, y: 0.0 };
            let p1 = Pos { x: 1.0, y: 1.0 };
            let p2 = Pos { x: 2.0, y: 2.0 };
            arch.push(e0, &[(pid, &p0 as *const _ as *const u8)]);
            arch.push(e1, &[(pid, &p1 as *const _ as *const u8)]);
            arch.push(e2, &[(pid, &p2 as *const _ as *const u8)]);
        }

        let swapped = arch.swap_remove(1);
        assert_eq!(swapped, Some(e2));
        assert_eq!(arch.len(), 2);
        unsafe {
            assert_eq!(arch.get::<Pos>(pid, 0), &Pos { x: 0.0, y: 0.0 });
            assert_eq!(arch.get::<Pos>(pid, 1), &Pos { x: 2.0, y: 2.0 });
        }
    }

    #[test]
    fn has_component() {
        let (cs, pid, vid) = setup();
        let arch = Archetype::new(ArchetypeId(0), &[(pid, cs.info(pid))]);
        assert!(arch.has_component(pid));
        assert!(!arch.has_component(vid));
    }
}
