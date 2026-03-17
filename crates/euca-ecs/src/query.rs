use std::collections::HashMap;
use std::marker::PhantomData;

use crate::archetype::Archetype;
use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::world::World;

// ── Component access tracking ──

/// Describes how a query element accesses a specific component.
#[derive(Clone, Debug)]
pub struct ComponentAccess {
    pub component_id: ComponentId,
    pub mutable: bool,
}

// ── Query filters ──

/// A filter that requires an entity to have a specific component.
pub struct With<T: Component>(PhantomData<T>);

/// A filter that requires an entity to NOT have a specific component.
pub struct Without<T: Component>(PhantomData<T>);

/// Trait for query filters.
pub trait QueryFilter {
    /// Check if an archetype matches this filter.
    fn matches(world: &World, archetype: &Archetype) -> bool;
}

impl QueryFilter for () {
    #[inline]
    fn matches(_world: &World, _archetype: &Archetype) -> bool {
        true
    }
}

impl<T: Component> QueryFilter for With<T> {
    fn matches(world: &World, archetype: &Archetype) -> bool {
        world
            .component_id::<T>()
            .is_some_and(|id| archetype.has_component(id))
    }
}

impl<T: Component> QueryFilter for Without<T> {
    fn matches(world: &World, archetype: &Archetype) -> bool {
        !world
            .component_id::<T>()
            .is_some_and(|id| archetype.has_component(id))
    }
}

// ── Query ──

/// Type-safe query over the world.
///
/// Supports both immutable (`&T`) and mutable (`&mut T`) component access.
/// Panics at creation time if the same component is accessed both mutably and immutably.
pub struct Query<'w, Q: WorldQuery, F: QueryFilter = ()> {
    world: &'w World,
    _marker: PhantomData<(Q, F)>,
}

impl<'w, Q: WorldQuery, F: QueryFilter> Query<'w, Q, F> {
    /// Create a new query bound to the given world.
    ///
    /// # Panics
    /// Panics if the query contains conflicting access to the same component
    /// (e.g., both `&T` and `&mut T`).
    pub fn new(world: &'w World) -> Self {
        // Validate no aliasing within this query's component accesses
        let accesses = Q::component_access(world);
        let mut seen: HashMap<ComponentId, bool> = HashMap::new();
        for access in &accesses {
            if let Some(&prev_mutable) = seen.get(&access.component_id)
                && (access.mutable || prev_mutable)
            {
                panic!(
                    "Query has conflicting access to the same component: \
                     cannot have both mutable and immutable (or two mutable) accesses"
                );
            }
            seen.insert(access.component_id, access.mutable);
        }

        Self {
            world,
            _marker: PhantomData,
        }
    }

    /// Iterate over all matching entities.
    pub fn iter(&self) -> QueryIter<'w, Q, F> {
        QueryIter {
            world: self.world,
            archetype_index: 0,
            row_index: 0,
            _marker: PhantomData,
        }
    }

    /// Count matching entities without iterating component data.
    pub fn count(&self) -> usize {
        let mut total = 0;
        for archetype in &self.world.archetypes {
            if Q::matches_archetype(self.world, archetype) && F::matches(self.world, archetype) {
                total += archetype.len();
            }
        }
        total
    }
}

/// Iterator over query results.
pub struct QueryIter<'w, Q: WorldQuery, F: QueryFilter> {
    world: &'w World,
    archetype_index: usize,
    row_index: usize,
    _marker: PhantomData<(Q, F)>,
}

impl<'w, Q: WorldQuery, F: QueryFilter> Iterator for QueryIter<'w, Q, F> {
    type Item = Q::Item<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.archetype_index >= self.world.archetypes.len() {
                return None;
            }

            let archetype = &self.world.archetypes[self.archetype_index];

            if !Q::matches_archetype(self.world, archetype)
                || !F::matches(self.world, archetype)
                || archetype.is_empty()
            {
                self.archetype_index += 1;
                self.row_index = 0;
                continue;
            }

            if self.row_index >= archetype.len() {
                self.archetype_index += 1;
                self.row_index = 0;
                continue;
            }

            let row = self.row_index;
            self.row_index += 1;

            // SAFETY: We verified the archetype matches and the row is valid.
            let item = unsafe { Q::fetch(self.world, archetype, row) };
            return Some(item);
        }
    }
}

// ── WorldQuery trait ──

/// Trait for query fetch parameters (what data to extract from matching archetypes).
///
/// # Safety
/// Implementations must correctly fetch data matching the component type.
pub unsafe trait WorldQuery {
    type Item<'w>;

    fn matches_archetype(world: &World, archetype: &Archetype) -> bool;

    /// Return the component accesses for this query element.
    /// Used for aliasing validation at query creation time.
    fn component_access(world: &World) -> Vec<ComponentAccess>;

    /// # Safety
    /// Caller must ensure the archetype matches and the row is valid.
    unsafe fn fetch<'w>(world: &'w World, archetype: &'w Archetype, row: usize) -> Self::Item<'w>;
}

// ── WorldQuery: Entity ──

unsafe impl WorldQuery for Entity {
    type Item<'w> = Entity;

    fn matches_archetype(_world: &World, _archetype: &Archetype) -> bool {
        true
    }

    fn component_access(_world: &World) -> Vec<ComponentAccess> {
        vec![]
    }

    unsafe fn fetch<'w>(_world: &'w World, archetype: &'w Archetype, row: usize) -> Self::Item<'w> {
        archetype.entities[row]
    }
}

// ── WorldQuery: &T (immutable) ──

unsafe impl<T: Component> WorldQuery for &T {
    type Item<'w> = &'w T;

    fn matches_archetype(world: &World, archetype: &Archetype) -> bool {
        world
            .component_id::<T>()
            .is_some_and(|id| archetype.has_component(id))
    }

    fn component_access(world: &World) -> Vec<ComponentAccess> {
        match world.component_id::<T>() {
            Some(id) => vec![ComponentAccess {
                component_id: id,
                mutable: false,
            }],
            None => vec![],
        }
    }

    unsafe fn fetch<'w>(world: &'w World, archetype: &'w Archetype, row: usize) -> Self::Item<'w> {
        let comp_id = world.components.id_of::<T>().unwrap();
        unsafe { archetype.get::<T>(comp_id, row) }
    }
}

// ── WorldQuery: &mut T (mutable) ──

unsafe impl<T: Component> WorldQuery for &mut T {
    type Item<'w> = &'w mut T;

    fn matches_archetype(world: &World, archetype: &Archetype) -> bool {
        world
            .component_id::<T>()
            .is_some_and(|id| archetype.has_component(id))
    }

    fn component_access(world: &World) -> Vec<ComponentAccess> {
        match world.component_id::<T>() {
            Some(id) => vec![ComponentAccess {
                component_id: id,
                mutable: true,
            }],
            None => vec![],
        }
    }

    unsafe fn fetch<'w>(world: &'w World, archetype: &'w Archetype, row: usize) -> Self::Item<'w> {
        let comp_id = world.components.id_of::<T>().unwrap();
        // SAFETY: Aliasing is validated at Query::new() — no other access to this component
        // in the same query. Change tick updated for change detection.
        unsafe { archetype.set_change_tick_unchecked(comp_id, row, world.tick as u32) };
        unsafe { archetype.get_mut::<T>(comp_id, row) }
    }
}

// ── WorldQuery: tuple impls (1 through 8) ──

macro_rules! impl_world_query_tuple {
    ($($name:ident),+) => {
        unsafe impl<$($name: WorldQuery),+> WorldQuery for ($($name,)+) {
            type Item<'w> = ($($name::Item<'w>,)+);

            #[inline]
            fn matches_archetype(world: &World, archetype: &Archetype) -> bool {
                $($name::matches_archetype(world, archetype))&&+
            }

            fn component_access(world: &World) -> Vec<ComponentAccess> {
                let mut accesses = Vec::new();
                $(accesses.extend($name::component_access(world));)+
                accesses
            }

            #[inline]
            unsafe fn fetch<'w>(world: &'w World, archetype: &'w Archetype, row: usize) -> Self::Item<'w> {
                unsafe {
                    ($($name::fetch(world, archetype, row),)+)
                }
            }
        }
    };
}

impl_world_query_tuple!(A);
impl_world_query_tuple!(A, B);
impl_world_query_tuple!(A, B, C);
impl_world_query_tuple!(A, B, C, D);
impl_world_query_tuple!(A, B, C, D, E);
impl_world_query_tuple!(A, B, C, D, E, F2);
impl_world_query_tuple!(A, B, C, D, E, F2, G);
impl_world_query_tuple!(A, B, C, D, E, F2, G, H);

// ── QueryFilter: tuple impls (1 through 8) ──

macro_rules! impl_query_filter_tuple {
    ($($name:ident),+) => {
        impl<$($name: QueryFilter),+> QueryFilter for ($($name,)+) {
            #[inline]
            fn matches(world: &World, archetype: &Archetype) -> bool {
                $($name::matches(world, archetype))&&+
            }
        }
    };
}

impl_query_filter_tuple!(A);
impl_query_filter_tuple!(A, B);
impl_query_filter_tuple!(A, B, C);
impl_query_filter_tuple!(A, B, C, D);
impl_query_filter_tuple!(A, B, C, D, E);
impl_query_filter_tuple!(A, B, C, D, E, F2);
impl_query_filter_tuple!(A, B, C, D, E, F2, G);
impl_query_filter_tuple!(A, B, C, D, E, F2, G, H);

// ── Tests ──

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
    struct Health(f32);

    #[derive(Debug, Clone, PartialEq)]
    struct Damage(f32);

    #[derive(Debug, Clone, PartialEq)]
    struct Static;

    // ── Existing tests (immutable queries) ──

    #[test]
    fn query_single_component() {
        let mut world = World::new();
        world.spawn(Position { x: 1.0, y: 1.0 });
        world.spawn(Position { x: 2.0, y: 2.0 });
        world.spawn(Position { x: 3.0, y: 3.0 });

        let query = Query::<&Position>::new(&world);
        let positions: Vec<_> = query.iter().collect();
        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0].x, 1.0);
        assert_eq!(positions[1].x, 2.0);
        assert_eq!(positions[2].x, 3.0);
    }

    #[test]
    fn query_tuple() {
        let mut world = World::new();

        let e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        world.insert(e1, Velocity { dx: 10.0, dy: 10.0 });

        world.spawn(Position { x: 2.0, y: 2.0 });

        let query = Query::<(&Position, &Velocity)>::new(&world);
        let results: Vec<_> = query.iter().collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.x, 1.0);
        assert_eq!(results[0].1.dx, 10.0);
    }

    #[test]
    fn query_with_entity() {
        let mut world = World::new();
        let e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        let e2 = world.spawn(Position { x: 2.0, y: 2.0 });

        let query = Query::<(Entity, &Position)>::new(&world);
        let results: Vec<_> = query.iter().collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, e1);
        assert_eq!(results[1].0, e2);
    }

    #[test]
    fn query_with_filter() {
        let mut world = World::new();

        let e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        world.insert(e1, Velocity { dx: 10.0, dy: 10.0 });

        let e2 = world.spawn(Position { x: 2.0, y: 2.0 });
        world.insert(e2, Static);

        let e3 = world.spawn(Position { x: 3.0, y: 3.0 });
        world.insert(e3, Velocity { dx: 30.0, dy: 30.0 });

        let query = Query::<&Position, Without<Static>>::new(&world);
        let results: Vec<_> = query.iter().collect();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn query_count() {
        let mut world = World::new();
        world.spawn(Position { x: 1.0, y: 1.0 });
        world.spawn(Position { x: 2.0, y: 2.0 });
        world.spawn(Position { x: 3.0, y: 3.0 });

        let query = Query::<&Position>::new(&world);
        assert_eq!(query.count(), 3);
    }

    // ── New tests: mutable queries ──

    #[test]
    fn query_mut_single_component() {
        let mut world = World::new();
        world.spawn(Position { x: 1.0, y: 1.0 });
        world.spawn(Position { x: 2.0, y: 2.0 });

        {
            let query = Query::<&mut Position>::new(&world);
            for pos in query.iter() {
                pos.x += 10.0;
            }
        }

        let query = Query::<&Position>::new(&world);
        let positions: Vec<_> = query.iter().collect();
        assert_eq!(positions[0].x, 11.0);
        assert_eq!(positions[1].x, 12.0);
    }

    #[test]
    fn query_mut_with_immutable() {
        let mut world = World::new();
        let e = world.spawn(Position { x: 1.0, y: 1.0 });
        world.insert(e, Velocity { dx: 10.0, dy: 20.0 });

        {
            let query = Query::<(&Velocity, &mut Position)>::new(&world);
            for (vel, pos) in query.iter() {
                pos.x += vel.dx;
                pos.y += vel.dy;
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 11.0);
        assert_eq!(world.get::<Position>(e).unwrap().y, 21.0);
    }

    #[test]
    fn query_mut_updates_change_tick() {
        let mut world = World::new();
        let e = world.spawn(Position { x: 1.0, y: 1.0 });
        assert_eq!(world.get_change_tick::<Position>(e), Some(0));

        world.tick(); // tick = 1

        {
            let query = Query::<&mut Position>::new(&world);
            for pos in query.iter() {
                pos.x = 99.0;
            }
        }

        assert_eq!(world.get_change_tick::<Position>(e), Some(1));
        assert_eq!(world.get::<Position>(e).unwrap().x, 99.0);
    }

    #[test]
    #[should_panic(expected = "conflicting access")]
    fn query_aliasing_panics() {
        let mut world = World::new();
        world.spawn(Position { x: 1.0, y: 1.0 });

        let _query = Query::<(&Position, &mut Position)>::new(&world);
    }

    #[test]
    fn query_two_mut_different_types() {
        let mut world = World::new();
        let e = world.spawn(Position { x: 1.0, y: 1.0 });
        world.insert(e, Velocity { dx: 5.0, dy: 5.0 });

        {
            let query = Query::<(&mut Position, &mut Velocity)>::new(&world);
            for (pos, vel) in query.iter() {
                pos.x += 1.0;
                vel.dx += 1.0;
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 2.0);
        assert_eq!(world.get::<Velocity>(e).unwrap().dx, 6.0);
    }

    #[test]
    fn query_entity_with_mut() {
        let mut world = World::new();
        let e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        let e2 = world.spawn(Position { x: 2.0, y: 2.0 });

        {
            let query = Query::<(Entity, &mut Position)>::new(&world);
            for (entity, pos) in query.iter() {
                if entity == e1 {
                    pos.x = 100.0;
                } else {
                    pos.x = 200.0;
                }
            }
        }

        assert_eq!(world.get::<Position>(e1).unwrap().x, 100.0);
        assert_eq!(world.get::<Position>(e2).unwrap().x, 200.0);
    }

    #[test]
    fn query_mut_with_filter() {
        let mut world = World::new();
        let e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        let e2 = world.spawn(Position { x: 2.0, y: 2.0 });
        world.insert(e2, Static);

        {
            let query = Query::<&mut Position, Without<Static>>::new(&world);
            for pos in query.iter() {
                pos.x += 100.0;
            }
        }

        assert_eq!(world.get::<Position>(e1).unwrap().x, 101.0);
        assert_eq!(world.get::<Position>(e2).unwrap().x, 2.0);
    }

    #[test]
    fn query_4_tuple() {
        let mut world = World::new();
        let e = world.spawn(Position { x: 1.0, y: 1.0 });
        world.insert(e, Velocity { dx: 2.0, dy: 2.0 });
        world.insert(e, Health(100.0));
        world.insert(e, Damage(25.0));

        let query = Query::<(&Position, &Velocity, &Health, &Damage)>::new(&world);
        let results: Vec<_> = query.iter().collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.x, 1.0);
        assert_eq!(results[0].1.dx, 2.0);
        assert_eq!(results[0].2.0, 100.0);
        assert_eq!(results[0].3.0, 25.0);
    }
}
