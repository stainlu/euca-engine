use euca_ecs::{Entity, Query, World};
use euca_physics::{PhysicsBody, PhysicsCollider};
use euca_render::{MaterialRef, MeshRenderer};
use euca_scene::{GlobalTransform, LocalTransform};

use crate::EditorState;

/// Top toolbar: Play/Pause/Step controls + info.
pub fn toolbar_panel(ctx: &egui::Context, state: &mut EditorState, world: &World) {
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if state.playing {
                if ui.button("⏸ Pause").clicked() {
                    state.playing = false;
                }
            } else if ui.button("▶ Play").clicked() {
                state.playing = true;
            }

            if ui.button("⏭ Step").clicked() {
                state.step_once = true;
            }

            if ui.button("⏹ Stop").clicked() {
                state.playing = false;
                state.reset_requested = true;
            }

            ui.separator();
            ui.label(format!(
                "Entities: {} | Tick: {} | Archetypes: {}",
                world.entity_count(),
                world.current_tick(),
                world.archetype_count(),
            ));
        });
    });
}

/// Left panel: entity hierarchy list.
pub fn hierarchy_panel(ctx: &egui::Context, state: &mut EditorState, world: &World) {
    egui::SidePanel::left("hierarchy")
        .default_width(200.0)
        .show(ctx, |ui| {
            ui.heading("Hierarchy");
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                // List all entities with GlobalTransform (scene entities)
                let entities: Vec<(Entity, bool, bool, bool)> = {
                    let query = Query::<Entity>::new(world);
                    query
                        .iter()
                        .map(|e| {
                            let has_mesh = world.get::<MeshRenderer>(e).is_some();
                            let has_transform = world.get::<LocalTransform>(e).is_some();
                            let has_physics = world.get::<PhysicsBody>(e).is_some();
                            (e, has_mesh, has_transform, has_physics)
                        })
                        .collect()
                };

                for (entity, has_mesh, has_transform, has_physics) in entities {
                    let selected = state.selected_entity == Some(entity.index());

                    let mut label = format!("Entity {}", entity);
                    if has_mesh {
                        label.push_str(" 🎨");
                    }
                    if has_physics {
                        label.push_str(" ⚡");
                    }
                    if !has_transform {
                        label.push_str(" (no transform)");
                    }

                    if ui.selectable_label(selected, &label).clicked() {
                        state.selected_entity = Some(entity.index());
                    }
                }
            });
        });
}

/// Right panel: inspector for the selected entity.
pub fn inspector_panel(ctx: &egui::Context, state: &mut EditorState, world: &mut World) {
    egui::SidePanel::right("inspector")
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.heading("Inspector");
            ui.separator();

            let entity_idx = match state.selected_entity {
                Some(idx) => idx,
                None => {
                    ui.label("No entity selected");
                    return;
                }
            };

            // Find the entity (try generation 0 first, then search)
            let entity = find_alive_entity(world, entity_idx);
            let entity = match entity {
                Some(e) => e,
                None => {
                    ui.label(format!("Entity {} not found", entity_idx));
                    state.selected_entity = None;
                    return;
                }
            };

            ui.label(format!("Entity: {}", entity));
            ui.separator();

            // Transform
            if let Some(lt) = world.get::<LocalTransform>(entity) {
                let t = lt.0;
                ui.collapsing("LocalTransform", |ui| {
                    ui.label(format!(
                        "Position: ({:.2}, {:.2}, {:.2})",
                        t.translation.x, t.translation.y, t.translation.z
                    ));
                    ui.label(format!(
                        "Scale: ({:.2}, {:.2}, {:.2})",
                        t.scale.x, t.scale.y, t.scale.z
                    ));
                });
            }

            if let Some(gt) = world.get::<GlobalTransform>(entity) {
                let t = gt.0;
                ui.collapsing("GlobalTransform", |ui| {
                    ui.label(format!(
                        "Position: ({:.2}, {:.2}, {:.2})",
                        t.translation.x, t.translation.y, t.translation.z
                    ));
                });
            }

            // Mesh
            if let Some(mr) = world.get::<MeshRenderer>(entity) {
                ui.collapsing("MeshRenderer", |ui| {
                    ui.label(format!("Mesh: #{}", mr.mesh.0));
                });
            }

            // Material
            if let Some(mat) = world.get::<MaterialRef>(entity) {
                ui.collapsing("MaterialRef", |ui| {
                    ui.label(format!("Material: #{}", mat.handle.0));
                });
            }

            // Physics
            if let Some(body) = world.get::<PhysicsBody>(entity) {
                ui.collapsing("PhysicsBody", |ui| {
                    ui.label(format!("Type: {:?}", body.body_type));
                });
            }

            if let Some(col) = world.get::<PhysicsCollider>(entity) {
                ui.collapsing("PhysicsCollider", |ui| {
                    ui.label(format!("Shape: {:?}", col.shape));
                    ui.label(format!("Restitution: {:.2}", col.restitution));
                    ui.label(format!("Friction: {:.2}", col.friction));
                });
            }
        });
}

/// Try to find a living entity with the given index.
fn find_alive_entity(world: &World, index: u32) -> Option<Entity> {
    // Try common generations
    for g in 0..16 {
        let e = Entity::from_raw(index, g);
        if world.is_alive(e) {
            return Some(e);
        }
    }
    None
}
