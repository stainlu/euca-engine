mod hierarchy;
mod prefab;
mod spatial;
pub mod streaming;
mod transform;

pub use hierarchy::{Children, Parent};
pub use prefab::{
    ComponentData, Damage, Health, Name, Prefab, PrefabRegistry, Speed, Team, WorldPrefabExt,
};
pub use spatial::{SpatialIndex, spatial_index_update_system};
pub use transform::{GlobalTransform, LocalTransform};
pub use streaming::{
    CameraPosition, ChunkData, ChunkEntityData, ChunkLoader, NullChunkLoader, StreamingConfig,
    StreamingState, WorldChunk, chunks_in_radius, streaming_update_system, world_to_chunk,
};

use std::collections::VecDeque;

use euca_ecs::{Entity, Query, World};

/// State for dirty-flag optimization. Tracks the last tick at which
/// transform propagation ran so we can skip unchanged entities.
#[derive(Clone, Debug, Default)]
pub struct PropagationState {
    pub last_tick: u32,
}

/// Returns true if component `T` on `entity` was modified since `since_tick`.
fn changed_since<T: 'static + Send + Sync>(world: &World, entity: Entity, since_tick: u32) -> bool {
    world
        .get_change_tick::<T>(entity)
        .is_some_and(|tick| tick >= since_tick)
}

/// Propagate transforms through the parent/child hierarchy (BFS).
///
/// Uses dirty-flag optimization: only recomputes GlobalTransform for entities
/// whose LocalTransform (or hierarchy) changed since the last propagation.
/// On first call (or when `PropagationState` is missing), propagates everything.
///
/// For each entity with a `Parent`, its `GlobalTransform` is computed as:
/// `parent.global_transform * self.local_transform`
///
/// Root entities (no Parent) have `GlobalTransform = LocalTransform`.
pub fn transform_propagation_system(world: &mut World) {
    let last_tick = world
        .resource::<PropagationState>()
        .map(|s| s.last_tick)
        .unwrap_or(0);
    let current_tick = world.current_tick() as u32;

    // First pass: process root entities (no Parent)
    let roots: Vec<(Entity, euca_math::Transform)> = {
        let query = Query::<(Entity, &LocalTransform), euca_ecs::Without<Parent>>::new(world);
        query.iter().map(|(e, lt)| (e, lt.0)).collect()
    };

    // BFS queue: (entity, parent_global_transform, parent_was_dirty)
    let mut queue: VecDeque<(Entity, euca_math::Transform, bool)> = VecDeque::new();

    for (entity, local) in &roots {
        let dirty = changed_since::<LocalTransform>(world, *entity, last_tick)
            || changed_since::<Children>(world, *entity, last_tick);

        if dirty && let Some(gt) = world.get_mut::<GlobalTransform>(*entity) {
            gt.0 = *local;
        }

        // Always enqueue children — they need to check their own dirty state
        if let Some(children) = world.get::<Children>(*entity) {
            for &child in &children.0 {
                queue.push_back((child, *local, dirty));
            }
        }
    }

    // Second pass: BFS through children
    while let Some((entity, parent_global, parent_dirty)) = queue.pop_front() {
        let self_dirty = changed_since::<LocalTransform>(world, entity, last_tick)
            || changed_since::<Parent>(world, entity, last_tick)
            || changed_since::<Children>(world, entity, last_tick);

        let needs_update = parent_dirty || self_dirty;

        let local = world
            .get::<LocalTransform>(entity)
            .map(|lt| lt.0)
            .unwrap_or(euca_math::Transform::IDENTITY);

        let global = parent_global.mul(local);

        if needs_update && let Some(gt) = world.get_mut::<GlobalTransform>(entity) {
            gt.0 = global;
        }

        if let Some(children) = world.get::<Children>(entity) {
            for &child in &children.0 {
                queue.push_back((child, global, needs_update));
            }
        }
    }

    // Update propagation state
    if let Some(state) = world.resource_mut::<PropagationState>() {
        state.last_tick = current_tick;
    } else {
        world.insert_resource(PropagationState {
            last_tick: current_tick,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::{Transform, Vec3};

    fn spawn_with_transform(world: &mut World, translation: Vec3) -> Entity {
        let entity = world.spawn(LocalTransform(Transform::from_translation(translation)));
        world.insert(entity, GlobalTransform(Transform::IDENTITY));
        entity
    }

    #[test]
    fn root_entity_propagation() {
        let mut world = World::new();
        let e = spawn_with_transform(&mut world, Vec3::new(1.0, 2.0, 3.0));

        transform_propagation_system(&mut world);

        let gt = world.get::<GlobalTransform>(e).unwrap();
        assert_eq!(gt.0.translation, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn parent_child_propagation() {
        let mut world = World::new();

        // Parent at (10, 0, 0)
        let parent = spawn_with_transform(&mut world, Vec3::new(10.0, 0.0, 0.0));

        // Child at local (5, 0, 0)
        let child = spawn_with_transform(&mut world, Vec3::new(5.0, 0.0, 0.0));

        // Set up hierarchy
        world.insert(child, Parent(parent));
        world.insert(parent, Children(vec![child]));

        transform_propagation_system(&mut world);

        // Child's global should be parent + child = (15, 0, 0)
        let gt = world.get::<GlobalTransform>(child).unwrap();
        assert!((gt.0.translation.x - 15.0).abs() < 1e-5);
    }

    #[test]
    fn three_level_hierarchy() {
        let mut world = World::new();

        let grandparent = spawn_with_transform(&mut world, Vec3::new(100.0, 0.0, 0.0));
        let parent = spawn_with_transform(&mut world, Vec3::new(10.0, 0.0, 0.0));
        let child = spawn_with_transform(&mut world, Vec3::new(1.0, 0.0, 0.0));

        world.insert(parent, Parent(grandparent));
        world.insert(child, Parent(parent));
        world.insert(grandparent, Children(vec![parent]));
        world.insert(parent, Children(vec![child]));

        transform_propagation_system(&mut world);

        let gt = world.get::<GlobalTransform>(child).unwrap();
        assert!((gt.0.translation.x - 111.0).abs() < 1e-5);
    }

    // ── New tests: dirty flag optimization ──

    #[test]
    fn dirty_flag_skips_unchanged() {
        let mut world = World::new();

        let e1 = spawn_with_transform(&mut world, Vec3::new(1.0, 0.0, 0.0));
        let e2 = spawn_with_transform(&mut world, Vec3::new(2.0, 0.0, 0.0));
        let e3 = spawn_with_transform(&mut world, Vec3::new(3.0, 0.0, 0.0));

        // First propagation: everything gets computed
        transform_propagation_system(&mut world);
        assert_eq!(
            world.get::<GlobalTransform>(e1).unwrap().0.translation.x,
            1.0
        );
        assert_eq!(
            world.get::<GlobalTransform>(e2).unwrap().0.translation.x,
            2.0
        );
        assert_eq!(
            world.get::<GlobalTransform>(e3).unwrap().0.translation.x,
            3.0
        );

        // Advance tick so change detection works
        world.tick();

        // Only modify e2
        world.get_mut::<LocalTransform>(e2).unwrap().0.translation.x = 20.0;

        // Second propagation: only e2 should update
        transform_propagation_system(&mut world);
        assert_eq!(
            world.get::<GlobalTransform>(e1).unwrap().0.translation.x,
            1.0
        ); // unchanged
        assert_eq!(
            world.get::<GlobalTransform>(e2).unwrap().0.translation.x,
            20.0
        ); // updated
        assert_eq!(
            world.get::<GlobalTransform>(e3).unwrap().0.translation.x,
            3.0
        ); // unchanged
    }

    #[test]
    fn parent_change_propagates_to_children() {
        let mut world = World::new();

        let parent = spawn_with_transform(&mut world, Vec3::new(10.0, 0.0, 0.0));
        let child = spawn_with_transform(&mut world, Vec3::new(5.0, 0.0, 0.0));

        world.insert(child, Parent(parent));
        world.insert(parent, Children(vec![child]));

        // First propagation
        transform_propagation_system(&mut world);
        assert!((world.get::<GlobalTransform>(child).unwrap().0.translation.x - 15.0).abs() < 1e-5);

        // Advance tick
        world.tick();

        // Move only the parent
        world
            .get_mut::<LocalTransform>(parent)
            .unwrap()
            .0
            .translation
            .x = 100.0;

        // Second propagation: child should update because parent was dirty
        transform_propagation_system(&mut world);
        assert!(
            (world.get::<GlobalTransform>(child).unwrap().0.translation.x - 105.0).abs() < 1e-5
        );
    }
}
