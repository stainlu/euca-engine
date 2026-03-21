//! Uniform-grid spatial index for fast proximity queries.
//!
//! Entities are bucketed into grid cells based on their [`GlobalTransform`]
//! translation. Queries scan only the cells that overlap the search region,
//! making them O(k) in the number of results rather than O(n) in total
//! entity count.
//!
//! Rebuild the index every frame (or whenever positions change) by running
//! [`spatial_index_update_system`].

use std::collections::HashMap;

use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;

use crate::GlobalTransform;

/// A 3D uniform-grid spatial index for fast proximity queries.
///
/// Uses a `HashMap<(i32, i32, i32), Vec<Entity>>` where each key is a grid
/// cell coordinate. Smaller cells give more precise culling but use more memory
/// when entities are spread across a large area. A good default is roughly
/// 2x the typical query radius.
pub struct SpatialIndex {
    /// Side length of each cubic grid cell.
    cell_size: f32,
    /// Reciprocal of cell_size, cached to avoid division in hot paths.
    inv_cell_size: f32,
    /// Maps cell coordinates to the entities contained within.
    cells: HashMap<(i32, i32, i32), Vec<Entity>>,
    /// Maps entity to its position, for radius queries that need distance checks.
    positions: HashMap<Entity, Vec3>,
}

impl SpatialIndex {
    /// Create a new spatial index with the given cell size.
    ///
    /// # Panics
    /// Panics if `cell_size` is not positive.
    pub fn new(cell_size: f32) -> Self {
        assert!(cell_size > 0.0, "cell_size must be positive");
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
            positions: HashMap::new(),
        }
    }

    /// The cell size this index was configured with.
    #[inline]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Clear all entries (called before rebuilding).
    pub(crate) fn clear(&mut self) {
        self.cells.clear();
        self.positions.clear();
    }

    /// Insert an entity at the given position.
    pub(crate) fn insert(&mut self, entity: Entity, position: Vec3) {
        let key = self.cell_key(position);
        self.cells.entry(key).or_default().push(entity);
        self.positions.insert(entity, position);
    }

    /// Compute the cell key for a world-space position.
    #[inline]
    fn cell_key(&self, pos: Vec3) -> (i32, i32, i32) {
        (
            (pos.x * self.inv_cell_size).floor() as i32,
            (pos.y * self.inv_cell_size).floor() as i32,
            (pos.z * self.inv_cell_size).floor() as i32,
        )
    }

    /// Find all entities within `radius` of `center`.
    ///
    /// Uses squared-distance comparison internally to avoid square roots.
    /// The returned entities are in no particular order.
    pub fn query_radius(&self, center: Vec3, radius: f32) -> Vec<Entity> {
        let radius_sq = radius * radius;

        let min = Vec3::new(center.x - radius, center.y - radius, center.z - radius);
        let max = Vec3::new(center.x + radius, center.y + radius, center.z + radius);
        let min_key = self.cell_key(min);
        let max_key = self.cell_key(max);

        let mut result = Vec::new();

        for cx in min_key.0..=max_key.0 {
            for cy in min_key.1..=max_key.1 {
                for cz in min_key.2..=max_key.2 {
                    if let Some(entities) = self.cells.get(&(cx, cy, cz)) {
                        for &entity in entities {
                            if let Some(&pos) = self.positions.get(&entity) {
                                let diff = pos - center;
                                if diff.length_squared() <= radius_sq {
                                    result.push(entity);
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Find all entities whose position falls within the axis-aligned bounding
    /// box defined by `min` and `max` (inclusive).
    pub fn query_aabb(&self, min: Vec3, max: Vec3) -> Vec<Entity> {
        let min_key = self.cell_key(min);
        let max_key = self.cell_key(max);

        let mut result = Vec::new();

        for cx in min_key.0..=max_key.0 {
            for cy in min_key.1..=max_key.1 {
                for cz in min_key.2..=max_key.2 {
                    if let Some(entities) = self.cells.get(&(cx, cy, cz)) {
                        for &entity in entities {
                            if let Some(&pos) = self.positions.get(&entity)
                                && pos.x >= min.x
                                && pos.x <= max.x
                                && pos.y >= min.y
                                && pos.y <= max.y
                                && pos.z >= min.z
                                && pos.z <= max.z
                            {
                                result.push(entity);
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// The total number of entities currently indexed.
    pub fn entity_count(&self) -> usize {
        self.positions.len()
    }

    /// The number of non-empty grid cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }
}

impl Default for SpatialIndex {
    /// Default cell size of 16.0 world units.
    fn default() -> Self {
        Self::new(16.0)
    }
}

/// System that rebuilds the [`SpatialIndex`] from all entities with a
/// [`GlobalTransform`] component.
///
/// Run this every frame after [`crate::transform_propagation_system`] so
/// that the index reflects up-to-date world positions.
pub fn spatial_index_update_system(world: &mut World) {
    // Ensure the resource exists.
    if world.resource::<SpatialIndex>().is_none() {
        world.insert_resource(SpatialIndex::default());
    }

    // Collect positions first to avoid borrow conflicts.
    let entries: Vec<(Entity, Vec3)> = {
        let query = Query::<(Entity, &GlobalTransform)>::new(world);
        query.iter().map(|(e, gt)| (e, gt.0.translation)).collect()
    };

    // Rebuild the index.
    // SAFETY: resource was just ensured above.
    let index = world
        .resource_mut::<SpatialIndex>()
        .expect("SpatialIndex resource missing");
    index.clear();
    for (entity, pos) in entries {
        index.insert(entity, pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    use crate::LocalTransform;

    /// Helper: spawn an entity with both Local and Global transforms at the given position.
    fn spawn_at(world: &mut World, pos: Vec3) -> Entity {
        let entity = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(entity, GlobalTransform(Transform::from_translation(pos)));
        entity
    }

    #[test]
    fn spatial_index_update_populates_from_global_transforms() {
        let mut world = World::new();
        let e1 = spawn_at(&mut world, Vec3::new(0.0, 0.0, 0.0));
        let e2 = spawn_at(&mut world, Vec3::new(10.0, 0.0, 0.0));

        spatial_index_update_system(&mut world);

        let index = world.resource::<SpatialIndex>().unwrap();
        assert_eq!(index.entity_count(), 2);

        let near_origin = index.query_radius(Vec3::ZERO, 1.0);
        assert!(near_origin.contains(&e1));
        assert!(!near_origin.contains(&e2));
    }

    #[test]
    fn query_radius_finds_entities_within_range() {
        let mut index = SpatialIndex::new(10.0);
        let e1 = Entity::from_raw(0, 0);
        let e2 = Entity::from_raw(1, 0);
        let e3 = Entity::from_raw(2, 0);

        index.insert(e1, Vec3::new(0.0, 0.0, 0.0));
        index.insert(e2, Vec3::new(5.0, 0.0, 0.0));
        index.insert(e3, Vec3::new(100.0, 0.0, 0.0));

        let result = index.query_radius(Vec3::ZERO, 6.0);
        assert!(result.contains(&e1));
        assert!(result.contains(&e2));
        assert!(!result.contains(&e3));
    }

    #[test]
    fn query_aabb_finds_entities_within_box() {
        let mut index = SpatialIndex::new(10.0);
        let e1 = Entity::from_raw(0, 0);
        let e2 = Entity::from_raw(1, 0);
        let e3 = Entity::from_raw(2, 0);

        index.insert(e1, Vec3::new(5.0, 5.0, 5.0));
        index.insert(e2, Vec3::new(15.0, 5.0, 5.0));
        index.insert(e3, Vec3::new(50.0, 50.0, 50.0));

        let result = index.query_aabb(Vec3::new(0.0, 0.0, 0.0), Vec3::new(20.0, 10.0, 10.0));
        assert!(result.contains(&e1));
        assert!(result.contains(&e2));
        assert!(!result.contains(&e3));
    }

    #[test]
    fn clear_removes_all_entries() {
        let mut index = SpatialIndex::new(10.0);
        index.insert(Entity::from_raw(0, 0), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(index.entity_count(), 1);

        index.clear();
        assert_eq!(index.entity_count(), 0);
        assert_eq!(index.cell_count(), 0);
    }

    #[test]
    fn query_radius_excludes_entities_outside_sphere() {
        let mut index = SpatialIndex::new(5.0);
        // Place an entity just outside the query radius on the diagonal.
        // Distance = sqrt(6^2 + 6^2 + 6^2) = sqrt(108) ~= 10.39
        let entity = Entity::from_raw(0, 0);
        index.insert(entity, Vec3::new(6.0, 6.0, 6.0));

        let result = index.query_radius(Vec3::ZERO, 10.0);
        assert!(
            !result.contains(&entity),
            "entity at distance ~10.39 should be outside radius 10.0"
        );
    }
}
