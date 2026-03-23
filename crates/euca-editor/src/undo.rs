use euca_ecs::World;
use euca_math::Transform;
use euca_physics::Collider;
use euca_render::{MaterialHandle, MaterialRef, MeshHandle, MeshRenderer};
use euca_scene::{GlobalTransform, LocalTransform};

/// Snapshot of an entity's components, stored for undo/redo respawn.
#[derive(Clone, Debug)]
pub struct EntitySnapshot {
    pub entity_index: u32,
    pub transform: Transform,
    pub mesh: Option<MeshHandle>,
    pub material: Option<MaterialHandle>,
    pub collider: Option<Collider>,
}

/// A reversible editor action.
#[derive(Clone, Debug)]
pub enum UndoAction {
    /// Transform was edited (inspector drag or gizmo drag).
    TransformEdit {
        entity_index: u32,
        old: Transform,
        new: Transform,
    },
    /// An entity was spawned.
    SpawnEntity { entity_index: u32 },
    /// An entity was despawned — stores data to re-create it.
    DespawnEntity(EntitySnapshot),
    /// Multiple entities were spawned at once (e.g. paste).
    SpawnMultiple { entity_indices: Vec<u32> },
    /// Multiple entities were despawned at once.
    DespawnMultiple { entities: Vec<EntitySnapshot> },
}

/// Stack-based undo/redo history with drag debouncing.
pub struct UndoHistory {
    undo_stack: Vec<UndoAction>,
    redo_stack: Vec<UndoAction>,
    /// Pending drag: (entity_index, transform_at_drag_start).
    /// Committed as a single UndoAction when the drag ends.
    pending_drag: Option<(u32, Transform)>,
}

impl UndoHistory {
    pub fn new() -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            pending_drag: None,
        }
    }

    /// Record an action. Clears the redo stack.
    pub fn push(&mut self, action: UndoAction) {
        self.undo_stack.push(action);
        self.redo_stack.clear();
    }

    /// Begin tracking a continuous drag (inspector or gizmo).
    pub fn begin_drag(&mut self, entity_index: u32, transform: Transform) {
        if self.pending_drag.is_none() {
            self.pending_drag = Some((entity_index, transform));
        }
    }

    /// End a drag and commit it as a single undo action.
    pub fn end_drag(&mut self, new_transform: Transform) {
        if let Some((entity_index, old_transform)) = self.pending_drag.take() {
            // Only push if something actually changed
            let delta = (old_transform.translation - new_transform.translation).length_squared()
                + (old_transform.scale - new_transform.scale).length_squared();
            if delta > 1e-8 {
                self.push(UndoAction::TransformEdit {
                    entity_index,
                    old: old_transform,
                    new: new_transform,
                });
            }
        }
    }

    /// Cancel a pending drag without recording it.
    pub fn cancel_drag(&mut self) {
        self.pending_drag = None;
    }

    /// Whether a drag is currently in progress.
    pub fn is_dragging(&self) -> bool {
        self.pending_drag.is_some()
    }

    /// Undo the last action.
    pub fn undo(&mut self, world: &mut World) {
        let action = match self.undo_stack.pop() {
            Some(a) => a,
            None => return,
        };
        let inverse = apply_inverse(world, &action);
        self.redo_stack.push(inverse);
    }

    /// Redo the last undone action.
    pub fn redo(&mut self, world: &mut World) {
        let action = match self.redo_stack.pop() {
            Some(a) => a,
            None => return,
        };
        let inverse = apply_inverse(world, &action);
        self.undo_stack.push(inverse);
    }
}

impl Default for UndoHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Capture the restorable state of an entity as an [`EntitySnapshot`].
fn capture_entity_snapshot(
    world: &World,
    entity_index: u32,
    entity: euca_ecs::Entity,
) -> EntitySnapshot {
    EntitySnapshot {
        entity_index,
        transform: world
            .get::<LocalTransform>(entity)
            .map(|lt| lt.0)
            .unwrap_or_default(),
        mesh: world.get::<MeshRenderer>(entity).map(|mr| mr.mesh),
        material: world.get::<MaterialRef>(entity).map(|mr| mr.handle),
        collider: world.get::<Collider>(entity).cloned(),
    }
}

/// Respawn an entity from a snapshot. Returns the new entity.
fn respawn_from_snapshot(world: &mut World, snapshot: &EntitySnapshot) -> euca_ecs::Entity {
    let e = world.spawn(LocalTransform(snapshot.transform));
    world.insert(e, GlobalTransform::default());
    if let Some(m) = snapshot.mesh {
        world.insert(e, MeshRenderer { mesh: m });
    }
    if let Some(m) = snapshot.material {
        world.insert(e, MaterialRef { handle: m });
    }
    if let Some(c) = &snapshot.collider {
        world.insert(e, c.clone());
    }
    e
}

/// Apply the inverse of an action to the world. Returns the action needed to reverse this inverse.
fn apply_inverse(world: &mut World, action: &UndoAction) -> UndoAction {
    match action {
        UndoAction::TransformEdit {
            entity_index,
            old,
            new,
        } => {
            // Reverse: set transform to old, store new as the "old" for redo
            if let Some(entity) = crate::find_alive_entity(world, *entity_index)
                && let Some(lt) = world.get_mut::<LocalTransform>(entity)
            {
                lt.0 = *old;
            }
            UndoAction::TransformEdit {
                entity_index: *entity_index,
                old: *new,
                new: *old,
            }
        }
        UndoAction::SpawnEntity { entity_index } => {
            // Reverse of spawn = despawn. Capture state first.
            if let Some(entity) = crate::find_alive_entity(world, *entity_index) {
                let snapshot = capture_entity_snapshot(world, *entity_index, entity);
                world.despawn(entity);
                UndoAction::DespawnEntity(snapshot)
            } else {
                action.clone()
            }
        }
        UndoAction::DespawnEntity(snapshot) => {
            // Reverse of despawn = respawn with stored data
            let e = respawn_from_snapshot(world, snapshot);
            UndoAction::SpawnEntity {
                entity_index: e.index(),
            }
        }
        UndoAction::SpawnMultiple { entity_indices } => {
            // Reverse of spawn-multiple = despawn all. Capture state first.
            let mut captured = Vec::new();
            for idx in entity_indices {
                if let Some(entity) = crate::find_alive_entity(world, *idx) {
                    captured.push(capture_entity_snapshot(world, *idx, entity));
                    world.despawn(entity);
                }
            }
            UndoAction::DespawnMultiple { entities: captured }
        }
        UndoAction::DespawnMultiple { entities } => {
            // Reverse of despawn-multiple = respawn all with stored data
            let spawned_indices = entities
                .iter()
                .map(|snapshot| respawn_from_snapshot(world, snapshot).index())
                .collect();
            UndoAction::SpawnMultiple {
                entity_indices: spawned_indices,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Vec3;

    #[test]
    fn push_clears_redo() {
        let mut h = UndoHistory::new();
        h.push(UndoAction::SpawnEntity { entity_index: 0 });
        h.push(UndoAction::SpawnEntity { entity_index: 1 });
        assert_eq!(h.undo_stack.len(), 2);
        assert_eq!(h.redo_stack.len(), 0);
    }

    #[test]
    fn drag_debouncing() {
        let mut h = UndoHistory::new();
        let old = Transform::from_translation(Vec3::ZERO);
        let new = Transform::from_translation(Vec3::new(5.0, 0.0, 0.0));

        h.begin_drag(0, old);
        assert!(h.is_dragging());

        h.end_drag(new);
        assert!(!h.is_dragging());
        assert_eq!(h.undo_stack.len(), 1);
    }

    #[test]
    fn drag_no_change_skipped() {
        let mut h = UndoHistory::new();
        let t = Transform::from_translation(Vec3::ZERO);

        h.begin_drag(0, t);
        h.end_drag(t); // same transform = no-op
        assert_eq!(h.undo_stack.len(), 0);
    }

    #[test]
    fn cancel_drag() {
        let mut h = UndoHistory::new();
        h.begin_drag(0, Transform::default());
        h.cancel_drag();
        assert!(!h.is_dragging());
    }
}
