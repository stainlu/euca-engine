//! Retained render extraction layer.
//!
//! Instead of rebuilding the entire `Vec<DrawCommand>` from the ECS every
//! frame, [`RenderExtractor`] maintains a persistent mapping from entities to
//! draw commands. Only entities with changed transforms are re-extracted,
//! dramatically reducing CPU overhead at high entity counts.
//!
//! # Usage
//! ```ignore
//! let mut extractor = RenderExtractor::new();
//!
//! // Each frame:
//! extractor.sync(&world);
//! renderer.draw(gpu, &camera, &light, &ambient, extractor.commands());
//! ```

use std::collections::HashMap;

use euca_ecs::{Entity, Query, World};
use euca_scene::GlobalTransform;

use crate::material::MaterialHandle;
use crate::mesh::{GroundOffset, MeshHandle};
use crate::renderer::DrawCommand;

/// Render-side copy of an entity's mesh and material handles.
/// Stored alongside the [`DrawCommand`] so we can detect mesh/material changes.
struct RenderEntity {
    mesh: MeshHandle,
    material: MaterialHandle,
}

/// A persistent extraction layer that caches [`DrawCommand`]s across frames.
///
/// Call [`sync`](Self::sync) each frame to incrementally update only the
/// entities whose transforms (or mesh/material assignments) have changed.
pub struct RenderExtractor {
    /// Persistent draw command list, indexed by slot.
    commands: Vec<DrawCommand>,
    /// Per-slot metadata for change detection.
    entities: Vec<Option<RenderEntity>>,
    /// Entity → slot index mapping.
    entity_to_slot: HashMap<Entity, usize>,
    /// Free slot indices (from despawned entities).
    free_slots: Vec<usize>,
    /// ECS tick at which we last synced (for change detection).
    last_sync_tick: u64,
}

impl RenderExtractor {
    /// Create a new extractor with no cached entities.
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            entities: Vec::new(),
            entity_to_slot: HashMap::new(),
            free_slots: Vec::new(),
            last_sync_tick: 0,
        }
    }

    /// Synchronize with the ECS world. Only re-extracts entities whose
    /// `GlobalTransform` changed since the last sync, plus any newly
    /// spawned or despawned renderable entities.
    ///
    /// # Components required per entity
    /// - `GlobalTransform` — world-space transform
    /// - `MeshRenderer` — mesh handle (`MeshRenderer { mesh: MeshHandle }`)
    /// - `MaterialRef` — material handle (`MaterialRef { handle: MaterialHandle }`)
    pub fn sync(&mut self, world: &World) {
        // ── Phase 1: Detect despawned entities ──
        // Entities in our map that no longer exist in the world.
        let mut despawned: Vec<Entity> = Vec::new();
        for &entity in self.entity_to_slot.keys() {
            if world.get::<GlobalTransform>(entity).is_none() {
                despawned.push(entity);
            }
        }
        for entity in despawned {
            if let Some(slot) = self.entity_to_slot.remove(&entity) {
                self.entities[slot] = None;
                self.free_slots.push(slot);
            }
        }

        // ── Phase 2: Query all renderable entities ──
        // We need to detect both newly spawned entities and transform changes.
        // Using the full query is simpler than maintaining spawn/despawn events,
        // and at 100K entities the query itself is ~0.1ms.
        let query = Query::<(
            Entity,
            &GlobalTransform,
            &crate::MeshRenderer,
            &crate::MaterialRef,
        )>::new(world);

        let current_tick = world.current_tick();

        for (entity, gt, mesh_renderer, mat_ref) in query.iter() {
            let mut model_matrix = gt.0.to_matrix();
            // Apply visual ground offset: shift the rendered mesh upward so its
            // bottom sits on the ground, without affecting the entity's logical position.
            if let Some(offset) = world.get::<GroundOffset>(entity) {
                model_matrix.cols[3][1] += offset.0;
            }

            if let Some(&slot) = self.entity_to_slot.get(&entity) {
                // Existing entity — check if transform changed.
                // We use the ECS change tick for GlobalTransform.
                let gt_changed = world
                    .get_change_tick::<GlobalTransform>(entity)
                    .is_some_and(|tick| (tick as u64) >= self.last_sync_tick);

                // Also check mesh/material changes.
                let meta_changed = self.entities[slot].as_ref().is_some_and(|re| {
                    re.mesh != mesh_renderer.mesh || re.material != mat_ref.handle
                });

                if gt_changed || meta_changed {
                    self.commands[slot] = DrawCommand {
                        mesh: mesh_renderer.mesh,
                        material: mat_ref.handle,
                        model_matrix,
                        aabb: None,
                    };
                    self.entities[slot] = Some(RenderEntity {
                        mesh: mesh_renderer.mesh,
                        material: mat_ref.handle,
                    });
                }
            } else {
                // New entity — allocate a slot.
                let slot = if let Some(free) = self.free_slots.pop() {
                    self.commands[free] = DrawCommand {
                        mesh: mesh_renderer.mesh,
                        material: mat_ref.handle,
                        model_matrix,
                        aabb: None,
                    };
                    self.entities[free] = Some(RenderEntity {
                        mesh: mesh_renderer.mesh,
                        material: mat_ref.handle,
                    });
                    free
                } else {
                    let slot = self.commands.len();
                    self.commands.push(DrawCommand {
                        mesh: mesh_renderer.mesh,
                        material: mat_ref.handle,
                        model_matrix,
                        aabb: None,
                    });
                    self.entities.push(Some(RenderEntity {
                        mesh: mesh_renderer.mesh,
                        material: mat_ref.handle,
                    }));
                    slot
                };
                self.entity_to_slot.insert(entity, slot);
            }
        }

        // ── Phase 3: Auto-compact on excessive fragmentation ──
        // If holes exceed 25% of total slots, compact to reclaim memory and
        // keep the command buffer dense for GPU submission.
        if self.free_slots.len() * 4 > self.commands.len() {
            self.compact();
        }

        self.last_sync_tick = current_tick;
    }

    /// Return the current draw command list for rendering.
    ///
    /// Note: contains holes (despawned entity slots) with stale data. The
    /// renderer's material/mesh check will handle these gracefully, but for
    /// best performance, call [`compact`] periodically.
    pub fn commands(&self) -> &[DrawCommand] {
        &self.commands
    }

    /// Return only active (non-despawned) draw commands as a lazy iterator.
    /// No allocation — the caller can iterate, collect, or chain as needed.
    pub fn active_commands(&self) -> impl Iterator<Item = &DrawCommand> {
        self.commands
            .iter()
            .zip(self.entities.iter())
            .filter_map(|(cmd, meta)| meta.as_ref().map(|_| cmd))
    }

    /// Number of active (non-despawned) renderable entities.
    pub fn active_count(&self) -> usize {
        self.entity_to_slot.len()
    }

    /// Total slot count (including holes from despawned entities).
    pub fn slot_count(&self) -> usize {
        self.commands.len()
    }

    /// Remove holes left by despawned entities, compacting the command list.
    /// Call periodically (e.g. every few seconds) to prevent unbounded growth.
    pub fn compact(&mut self) {
        if self.free_slots.is_empty() {
            return;
        }

        // Rebuild compacted arrays.
        let mut new_commands = Vec::with_capacity(self.entity_to_slot.len());
        let mut new_entities = Vec::with_capacity(self.entity_to_slot.len());
        let mut new_map = HashMap::with_capacity(self.entity_to_slot.len());

        for (&entity, &old_slot) in &self.entity_to_slot {
            let new_slot = new_commands.len();
            new_commands.push(self.commands[old_slot].clone());
            new_entities.push(self.entities[old_slot].take());
            new_map.insert(entity, new_slot);
        }

        self.commands = new_commands;
        self.entities = new_entities;
        self.entity_to_slot = new_map;
        self.free_slots.clear();
    }
}

impl Default for RenderExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extractor_starts_empty() {
        let ext = RenderExtractor::new();
        assert_eq!(ext.active_count(), 0);
        assert_eq!(ext.slot_count(), 0);
        assert!(ext.commands().is_empty());
    }
}
