use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::RwLock;

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

// ── Query cache ──

/// Unique key for a `(Q, F)` query type combination. We use the address of a
/// monomorphized function as a type-erased identity -- each `(Q, F)` pair gets
/// its own instantiation with a distinct address.
fn query_cache_key<Q: WorldQuery, F: QueryFilter>() {}
type CacheKeyFn = fn();

/// Shared query cache stored in [`World`].
///
/// Maps each query type to its list of matching archetype indices.
/// The cache is invalidated when new archetypes are created (tracked via a
/// generation counter). Hot-path systems running 60+ times per second benefit
/// from skipping the archetype scan on every [`Query`] construction.
pub struct QueryCache {
    /// Function-pointer key -> matching archetype indices.
    cache: HashMap<CacheKeyFn, Vec<usize>>,
    /// Global generation: incremented when archetypes change in the world.
    generation: u64,
    /// Last-seen generation per query type. If this differs from `generation`,
    /// the entry is stale and must be recomputed.
    query_generations: HashMap<CacheKeyFn, u64>,
}

impl QueryCache {
    /// Create an empty query cache at generation 0.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            generation: 0,
            query_generations: HashMap::new(),
        }
    }

    /// Notify the cache that a new archetype has been created.
    /// All cached entries become potentially stale.
    pub(crate) fn increment_generation(&mut self) {
        self.generation += 1;
    }

    /// Returns the current cache generation.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Look up the cached archetype list for a query type.
    /// Returns `Some(&[usize])` if the cache entry exists **and** is fresh
    /// (i.e. no new archetypes have been created since the entry was stored).
    fn get<Q: WorldQuery, F: QueryFilter>(&self) -> Option<&Vec<usize>> {
        let key: CacheKeyFn = query_cache_key::<Q, F>;
        let entry_gen = self.query_generations.get(&key)?;
        if *entry_gen == self.generation {
            self.cache.get(&key)
        } else {
            None
        }
    }

    /// Store (or update) the cached archetype list for a query type at the
    /// current generation.
    fn insert<Q: WorldQuery, F: QueryFilter>(&mut self, archetypes: Vec<usize>) {
        let key: CacheKeyFn = query_cache_key::<Q, F>;
        self.query_generations.insert(key, self.generation);
        self.cache.insert(key, archetypes);
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a new `RwLock<QueryCache>` for embedding in [`World`].
pub(crate) fn new_query_cache_lock() -> RwLock<QueryCache> {
    RwLock::new(QueryCache::new())
}

// ── Query filters ──

/// A filter that requires an entity to have a specific component.
pub struct With<T: Component>(PhantomData<T>);

/// A filter that requires an entity to NOT have a specific component.
pub struct Without<T: Component>(PhantomData<T>);

/// Marker for change-detection queries. Use with `World::changed_entities::<T>(since_tick)`
/// for efficient iteration of only modified entities.
///
/// Per-entity change detection is available via `world.get_change_tick::<T>(entity)`.
/// Field-level granularity requires wrapper types (future work).
#[allow(dead_code)]
pub struct Changed<T: Component>(PhantomData<T>);

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
    /// Cached matching archetype indices, valid when `cache_generation` matches world.
    cached_archetypes: Vec<usize>,
    cache_generation: u64,
    _marker: PhantomData<(Q, F)>,
}

impl<'w, Q: WorldQuery, F: QueryFilter> Query<'w, Q, F> {
    /// Create a new query bound to the given world.
    ///
    /// # Panics
    /// Panics if the query contains conflicting access to the same component
    /// (e.g., both `&T` and `&mut T`).
    pub fn new(world: &'w World) -> Self {
        Self::validate_no_aliasing(world);

        let cached_archetypes = Self::compute_matching(world);
        let cache_generation = world.archetype_generation;

        Self {
            world,
            cached_archetypes,
            cache_generation,
            _marker: PhantomData,
        }
    }

    /// Create a query using the world's shared [`QueryCache`].
    ///
    /// On cache hit (same query type, no new archetypes since last call), this
    /// skips the archetype scan entirely -- O(1) instead of O(archetypes).
    /// On cache miss, the scan runs once and the result is stored for future calls.
    ///
    /// # Panics
    /// Same aliasing rules as [`Query::new`].
    pub fn new_cached(world: &'w World) -> Self {
        Self::validate_no_aliasing(world);

        // Try the shared cache first (read lock)
        {
            let cache = world.query_cache.read().expect("query cache lock poisoned");
            if let Some(cached) = cache.get::<Q, F>() {
                return Self {
                    world,
                    cached_archetypes: cached.clone(),
                    cache_generation: world.archetype_generation,
                    _marker: PhantomData,
                };
            }
        }

        // Cache miss: compute and store (write lock)
        let cached_archetypes = Self::compute_matching(world);
        {
            let mut cache = world
                .query_cache
                .write()
                .expect("query cache lock poisoned");
            cache.insert::<Q, F>(cached_archetypes.clone());
        }

        Self {
            world,
            cached_archetypes,
            cache_generation: world.archetype_generation,
            _marker: PhantomData,
        }
    }

    /// Validate that no component is accessed both mutably and immutably
    /// (or mutably twice).
    fn validate_no_aliasing(world: &World) {
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
    }

    /// Compute matching archetype indices (includes empty -- they may gain entities later in the same borrow).
    fn compute_matching(world: &World) -> Vec<usize> {
        world
            .archetypes
            .iter()
            .enumerate()
            .filter(|(_, arch)| Q::matches_archetype(world, arch) && F::matches(world, arch))
            .map(|(i, _)| i)
            .collect()
    }

    /// Iterate over all matching entities.
    ///
    /// Uses cached archetype indices. The cache is valid as long as no new
    /// archetypes were created since the Query was built (asserted in debug).
    pub fn iter(&self) -> QueryIter<'w, Q, F> {
        debug_assert!(
            self.cache_generation == self.world.archetype_generation,
            "Query cache stale: create a new Query after structural world changes"
        );
        QueryIter {
            world: self.world,
            matching_archetypes: self.cached_archetypes.clone(),
            arch_cursor: 0,
            row_index: 0,
            _marker: PhantomData,
        }
    }

    /// Count matching entities without iterating component data.
    pub fn count(&self) -> usize {
        self.cached_archetypes
            .iter()
            .map(|&i| self.world.archetypes[i].len())
            .sum()
    }
}

/// Iterator over query results. Uses cached archetype indices to skip non-matching archetypes.
pub struct QueryIter<'w, Q: WorldQuery, F: QueryFilter> {
    world: &'w World,
    matching_archetypes: Vec<usize>,
    arch_cursor: usize,
    row_index: usize,
    _marker: PhantomData<(Q, F)>,
}

impl<'w, Q: WorldQuery, F: QueryFilter> Iterator for QueryIter<'w, Q, F> {
    type Item = Q::Item<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.arch_cursor >= self.matching_archetypes.len() {
                return None;
            }

            let arch_idx = self.matching_archetypes[self.arch_cursor];
            let archetype = &self.world.archetypes[arch_idx];

            if archetype.is_empty() || self.row_index >= archetype.len() {
                self.arch_cursor += 1;
                self.row_index = 0;
                continue;
            }

            let row = self.row_index;
            self.row_index += 1;

            // SAFETY: Archetype matching was verified at cache build time.
            // Row is within bounds (checked above).
            let item = unsafe { Q::fetch(self.world, archetype, row) };
            return Some(item);
        }
    }
}

// ── WorldQuery trait ──

/// Trait for query fetch parameters (what data to extract from matching archetypes).
///
/// # Safety
/// Implementations must correctly fetch data matching the component type and
/// respect aliasing rules (no overlapping mutable references).
pub unsafe trait WorldQuery {
    type Item<'w>;

    fn matches_archetype(world: &World, archetype: &Archetype) -> bool;

    /// Return the component accesses for this query element.
    /// Used for aliasing validation at query creation time.
    fn component_access(world: &World) -> Vec<ComponentAccess>;

    /// Fetch data for a single row in a matching archetype.
    ///
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
        // SAFETY: Aliasing is validated at Query::new() -- no other access to this component
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
    struct Position { x: f32, y: f32 }
    #[derive(Debug, Clone, PartialEq)]
    struct Velocity { dx: f32, dy: f32 }
    #[derive(Debug, Clone, PartialEq)]
    struct Health(f32);
    #[derive(Debug, Clone, PartialEq)]
    struct Damage(f32);
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

    #[test]
    fn query_mut_single_component() {
        let mut world = World::new();
        world.spawn(Position { x: 1.0, y: 1.0 });
        world.spawn(Position { x: 2.0, y: 2.0 });
        { let query = Query::<&mut Position>::new(&world); for pos in query.iter() { pos.x += 10.0; } }
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
        { let query = Query::<(&Velocity, &mut Position)>::new(&world); for (vel, pos) in query.iter() { pos.x += vel.dx; pos.y += vel.dy; } }
        assert_eq!(world.get::<Position>(e).unwrap().x, 11.0);
        assert_eq!(world.get::<Position>(e).unwrap().y, 21.0);
    }

    #[test]
    fn query_mut_updates_change_tick() {
        let mut world = World::new();
        let e = world.spawn(Position { x: 1.0, y: 1.0 });
        assert_eq!(world.get_change_tick::<Position>(e), Some(0));
        world.tick();
        { let query = Query::<&mut Position>::new(&world); for pos in query.iter() { pos.x = 99.0; } }
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
        { let query = Query::<(&mut Position, &mut Velocity)>::new(&world); for (pos, vel) in query.iter() { pos.x += 1.0; vel.dx += 1.0; } }
        assert_eq!(world.get::<Position>(e).unwrap().x, 2.0);
        assert_eq!(world.get::<Velocity>(e).unwrap().dx, 6.0);
    }

    #[test]
    fn query_entity_with_mut() {
        let mut world = World::new();
        let e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        let e2 = world.spawn(Position { x: 2.0, y: 2.0 });
        { let query = Query::<(Entity, &mut Position)>::new(&world); for (entity, pos) in query.iter() { if entity == e1 { pos.x = 100.0; } else { pos.x = 200.0; } } }
        assert_eq!(world.get::<Position>(e1).unwrap().x, 100.0);
        assert_eq!(world.get::<Position>(e2).unwrap().x, 200.0);
    }

    #[test]
    fn query_mut_with_filter() {
        let mut world = World::new();
        let e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        let e2 = world.spawn(Position { x: 2.0, y: 2.0 });
        world.insert(e2, Static);
        { let query = Query::<&mut Position, Without<Static>>::new(&world); for pos in query.iter() { pos.x += 100.0; } }
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

    // ── Query cache tests ──

    #[test]
    fn query_cache_hit_returns_same_results() {
        let mut world = World::new();
        world.spawn(Position { x: 1.0, y: 1.0 });
        world.spawn(Position { x: 2.0, y: 2.0 });
        let q1 = Query::<&Position>::new_cached(&world);
        assert_eq!(q1.count(), 2);
        let q2 = Query::<&Position>::new_cached(&world);
        assert_eq!(q2.count(), 2);
        let positions: Vec<_> = q2.iter().collect();
        assert_eq!(positions[0].x, 1.0);
        assert_eq!(positions[1].x, 2.0);
    }

    #[test]
    fn query_cache_invalidated_on_new_archetype() {
        let mut world = World::new();
        world.spawn(Position { x: 1.0, y: 1.0 });
        let q1 = Query::<&Position>::new_cached(&world);
        assert_eq!(q1.count(), 1);
        drop(q1);
        let e2 = world.spawn(Position { x: 2.0, y: 2.0 });
        world.insert(e2, Velocity { dx: 5.0, dy: 5.0 });
        let q2 = Query::<&Position>::new_cached(&world);
        assert_eq!(q2.count(), 2);
    }

    #[test]
    fn query_cache_different_types_cached_separately() {
        let mut world = World::new();
        let e = world.spawn(Position { x: 1.0, y: 1.0 });
        world.insert(e, Velocity { dx: 5.0, dy: 5.0 });
        world.spawn(Position { x: 2.0, y: 2.0 });
        let q_pos = Query::<&Position>::new_cached(&world);
        let q_vel = Query::<&Velocity>::new_cached(&world);
        assert_eq!(q_pos.count(), 2);
        assert_eq!(q_vel.count(), 1);
        drop(q_pos); drop(q_vel);
        let q_pos2 = Query::<&Position>::new_cached(&world);
        let q_vel2 = Query::<&Velocity>::new_cached(&world);
        assert_eq!(q_pos2.count(), 2);
        assert_eq!(q_vel2.count(), 1);
    }

    #[test]
    fn query_cache_with_filter_cached_separately_from_no_filter() {
        let mut world = World::new();
        let _e1 = world.spawn(Position { x: 1.0, y: 1.0 });
        let e2 = world.spawn(Position { x: 2.0, y: 2.0 });
        world.insert(e2, Static);
        let q_all = Query::<&Position>::new_cached(&world);
        let q_filtered = Query::<&Position, Without<Static>>::new_cached(&world);
        assert_eq!(q_all.count(), 2);
        assert_eq!(q_filtered.count(), 1);
        drop(q_all); drop(q_filtered);
        let e3 = world.spawn(Position { x: 3.0, y: 3.0 });
        world.insert(e3, Health(100.0));
        let q_all3 = Query::<&Position>::new_cached(&world);
        let q_filtered3 = Query::<&Position, Without<Static>>::new_cached(&world);
        assert_eq!(q_all3.count(), 3);
        assert_eq!(q_filtered3.count(), 2);
    }

    #[test]
    fn query_cache_mutable_query_works() {
        let mut world = World::new();
        world.spawn(Position { x: 1.0, y: 1.0 });
        world.spawn(Position { x: 2.0, y: 2.0 });
        { let query = Query::<&mut Position>::new_cached(&world); for pos in query.iter() { pos.x += 100.0; } }
        let query = Query::<&Position>::new_cached(&world);
        let positions: Vec<_> = query.iter().collect();
        assert_eq!(positions[0].x, 101.0);
        assert_eq!(positions[1].x, 102.0);
    }

    #[test]
    fn query_cache_generation_increments_on_archetype_creation() {
        let mut world = World::new();
        let gen0 = world.query_cache.read().expect("lock").generation();
        assert_eq!(gen0, 0);
        world.spawn(Position { x: 1.0, y: 1.0 });
        let gen1 = world.query_cache.read().expect("lock").generation();
        assert_eq!(gen1, 1);
        world.spawn(Position { x: 2.0, y: 2.0 });
        let gen2 = world.query_cache.read().expect("lock").generation();
        assert_eq!(gen2, 1);
        world.spawn(Velocity { dx: 1.0, dy: 1.0 });
        let gen3 = world.query_cache.read().expect("lock").generation();
        assert_eq!(gen3, 2);
    }
}
