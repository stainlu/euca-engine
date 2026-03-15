use std::marker::PhantomData;

use crate::archetype::Archetype;
use crate::component::Component;
use crate::world::World;

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

// Tuple filter combinators
impl<A: QueryFilter, B: QueryFilter> QueryFilter for (A, B) {
    #[inline]
    fn matches(world: &World, archetype: &Archetype) -> bool {
        A::matches(world, archetype) && B::matches(world, archetype)
    }
}

impl<A: QueryFilter, B: QueryFilter, C: QueryFilter> QueryFilter for (A, B, C) {
    #[inline]
    fn matches(world: &World, archetype: &Archetype) -> bool {
        A::matches(world, archetype) && B::matches(world, archetype) && C::matches(world, archetype)
    }
}

/// Type-safe query over the world.
///
/// Fetches components matching the query parameters from all matching archetypes.
pub struct Query<'w, Q: WorldQuery, F: QueryFilter = ()> {
    world: &'w World,
    _marker: PhantomData<(Q, F)>,
}

impl<'w, Q: WorldQuery, F: QueryFilter> Query<'w, Q, F> {
    pub fn new(world: &'w World) -> Self {
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

            // Check if this archetype matches
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

/// Trait for query fetch parameters (what data to extract from matching archetypes).
///
/// # Safety
/// Implementations must correctly fetch data matching the component type.
pub unsafe trait WorldQuery {
    type Item<'w>;

    fn matches_archetype(world: &World, archetype: &Archetype) -> bool;

    /// # Safety
    /// Caller must ensure the archetype matches and the row is valid.
    unsafe fn fetch<'w>(world: &'w World, archetype: &'w Archetype, row: usize) -> Self::Item<'w>;
}

// ── WorldQuery implementations for single component references ──

// &T - immutable component access
unsafe impl<T: Component> WorldQuery for &T {
    type Item<'w> = &'w T;

    fn matches_archetype(world: &World, archetype: &Archetype) -> bool {
        world
            .component_id::<T>()
            .is_some_and(|id| archetype.has_component(id))
    }

    unsafe fn fetch<'w>(world: &'w World, archetype: &'w Archetype, row: usize) -> Self::Item<'w> {
        let comp_id = world.components.id_of::<T>().unwrap();
        unsafe { archetype.get::<T>(comp_id, row) }
    }
}

// ── WorldQuery for tuples ──

unsafe impl<A: WorldQuery, B: WorldQuery> WorldQuery for (A, B) {
    type Item<'w> = (A::Item<'w>, B::Item<'w>);

    fn matches_archetype(world: &World, archetype: &Archetype) -> bool {
        A::matches_archetype(world, archetype) && B::matches_archetype(world, archetype)
    }

    unsafe fn fetch<'w>(world: &'w World, archetype: &'w Archetype, row: usize) -> Self::Item<'w> {
        unsafe {
            (
                A::fetch(world, archetype, row),
                B::fetch(world, archetype, row),
            )
        }
    }
}

unsafe impl<A: WorldQuery, B: WorldQuery, C: WorldQuery> WorldQuery for (A, B, C) {
    type Item<'w> = (A::Item<'w>, B::Item<'w>, C::Item<'w>);

    fn matches_archetype(world: &World, archetype: &Archetype) -> bool {
        A::matches_archetype(world, archetype)
            && B::matches_archetype(world, archetype)
            && C::matches_archetype(world, archetype)
    }

    unsafe fn fetch<'w>(world: &'w World, archetype: &'w Archetype, row: usize) -> Self::Item<'w> {
        unsafe {
            (
                A::fetch(world, archetype, row),
                B::fetch(world, archetype, row),
                C::fetch(world, archetype, row),
            )
        }
    }
}

// ── Entity fetch (get the entity ID alongside components) ──

use crate::entity::Entity;

unsafe impl WorldQuery for Entity {
    type Item<'w> = Entity;

    fn matches_archetype(_world: &World, _archetype: &Archetype) -> bool {
        true // Entity is always available
    }

    unsafe fn fetch<'w>(_world: &'w World, archetype: &'w Archetype, row: usize) -> Self::Item<'w> {
        archetype.entities[row]
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
    struct Static;

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

        // Entity with both Position and Velocity
        let e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        world.insert(e1, Velocity { dx: 10.0, dy: 10.0 });

        // Entity with only Position
        world.spawn(Position { x: 2.0, y: 2.0 });

        let query = Query::<(&Position, &Velocity)>::new(&world);
        let results: Vec<_> = query.iter().collect();

        // Only e1 has both components
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

        // Query Position without Static
        let query = Query::<&Position, Without<Static>>::new(&world);
        let results: Vec<_> = query.iter().collect();

        // e1 (Pos+Vel) and e3 (Pos+Vel) match, but not e2 (Pos+Static)
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
}
