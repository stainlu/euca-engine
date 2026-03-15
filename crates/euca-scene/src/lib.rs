mod hierarchy;
mod transform;

pub use hierarchy::{Children, Parent};
pub use transform::{GlobalTransform, LocalTransform};

use euca_ecs::{Entity, Query, World};

/// Propagate transforms through the parent/child hierarchy.
///
/// For each entity with a `Parent`, its `GlobalTransform` is computed as:
/// `parent.global_transform * self.local_transform`
///
/// Root entities (no Parent) have `GlobalTransform = LocalTransform`.
pub fn transform_propagation_system(world: &mut World) {
    // First pass: set GlobalTransform for root entities (no Parent)
    let roots: Vec<(Entity, euca_math::Transform)> = {
        let query = Query::<(Entity, &LocalTransform), euca_ecs::Without<Parent>>::new(world);
        query.iter().map(|(e, lt)| (e, lt.0)).collect()
    };

    for (entity, local) in &roots {
        if let Some(gt) = world.get_mut::<GlobalTransform>(*entity) {
            gt.0 = *local;
        }
    }

    // Second pass: propagate through children (BFS)
    let mut queue: Vec<(Entity, euca_math::Transform)> = Vec::new();
    for (entity, local) in roots {
        if let Some(children) = world.get::<Children>(entity) {
            let global = local; // Root's global = local
            for &child in &children.0 {
                queue.push((child, global));
            }
        }
    }

    while let Some((entity, parent_global)) = queue.pop() {
        let local = world
            .get::<LocalTransform>(entity)
            .map(|lt| lt.0)
            .unwrap_or(euca_math::Transform::IDENTITY);

        let global = parent_global.mul(local);

        if let Some(gt) = world.get_mut::<GlobalTransform>(entity) {
            gt.0 = global;
        }

        if let Some(children) = world.get::<Children>(entity) {
            for &child in &children.0 {
                queue.push((child, global));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::{Quat, Transform, Vec3};

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
}
