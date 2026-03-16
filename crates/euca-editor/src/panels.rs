use euca_ecs::{Entity, Query, World};
use euca_physics::{Collider, PhysicsBody};
use euca_render::{MaterialRef, MeshRenderer};
use euca_scene::{GlobalTransform, LocalTransform};

use crate::EditorState;

/// Actions returned from the toolbar.
#[derive(Clone, Debug)]
pub enum ToolbarAction {
    SaveScene,
    LoadScene,
}

/// Top toolbar: Play/Pause/Step controls + Save/Load + info + FPS.
pub fn toolbar_panel(
    ctx: &egui::Context,
    state: &mut EditorState,
    world: &World,
    delta_time: f32,
) -> Option<ToolbarAction> {
    let mut action = None;
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

            if ui.button("Save").clicked() {
                action = Some(ToolbarAction::SaveScene);
            }
            if ui.button("Load").clicked() {
                action = Some(ToolbarAction::LoadScene);
            }

            ui.separator();

            let fps = if delta_time > 0.0 {
                (1.0 / delta_time) as u32
            } else {
                0
            };
            ui.label(format!(
                "FPS: {} | Entities: {} | Tick: {} | Archetypes: {}",
                fps,
                world.entity_count(),
                world.current_tick(),
                world.archetype_count(),
            ));
        });
    });
    action
}

/// Entity spawn request from the hierarchy panel.
#[derive(Clone, Debug)]
pub enum SpawnRequest {
    Empty,
    Cube,
    Sphere,
}

/// Left panel: entity hierarchy list.
pub fn hierarchy_panel(
    ctx: &egui::Context,
    state: &mut EditorState,
    world: &World,
) -> Option<SpawnRequest> {
    let mut spawn = None;
    egui::SidePanel::left("hierarchy")
        .default_width(200.0)
        .show(ctx, |ui| {
            ui.heading("Hierarchy");

            // Add Entity buttons
            ui.horizontal(|ui| {
                if ui.button("+ Empty").clicked() {
                    spawn = Some(SpawnRequest::Empty);
                }
                if ui.button("+ Cube").clicked() {
                    spawn = Some(SpawnRequest::Cube);
                }
                if ui.button("+ Sphere").clicked() {
                    spawn = Some(SpawnRequest::Sphere);
                }
            });

            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
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
    spawn
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

            // Editable Transform
            if let Some(lt) = world.get::<LocalTransform>(entity) {
                let mut pos = [lt.0.translation.x, lt.0.translation.y, lt.0.translation.z];
                let mut scl = [lt.0.scale.x, lt.0.scale.y, lt.0.scale.z];
                let mut changed = false;

                ui.collapsing("LocalTransform", |ui| {
                    ui.label("Position:");
                    ui.horizontal(|ui| {
                        ui.label("X");
                        changed |= ui
                            .add(egui::DragValue::new(&mut pos[0]).speed(0.1))
                            .changed();
                        ui.label("Y");
                        changed |= ui
                            .add(egui::DragValue::new(&mut pos[1]).speed(0.1))
                            .changed();
                        ui.label("Z");
                        changed |= ui
                            .add(egui::DragValue::new(&mut pos[2]).speed(0.1))
                            .changed();
                    });
                    ui.label("Scale:");
                    ui.horizontal(|ui| {
                        ui.label("X");
                        changed |= ui
                            .add(egui::DragValue::new(&mut scl[0]).speed(0.01))
                            .changed();
                        ui.label("Y");
                        changed |= ui
                            .add(egui::DragValue::new(&mut scl[1]).speed(0.01))
                            .changed();
                        ui.label("Z");
                        changed |= ui
                            .add(egui::DragValue::new(&mut scl[2]).speed(0.01))
                            .changed();
                    });
                });

                if changed && let Some(lt) = world.get_mut::<LocalTransform>(entity) {
                    lt.0.translation = euca_math::Vec3::new(pos[0], pos[1], pos[2]);
                    lt.0.scale = euca_math::Vec3::new(scl[0], scl[1], scl[2]);
                }
            }

            if let Some(gt) = world.get::<GlobalTransform>(entity) {
                ui.collapsing("GlobalTransform", |ui| {
                    ui.label(format!(
                        "World Pos: ({:.2}, {:.2}, {:.2})",
                        gt.0.translation.x, gt.0.translation.y, gt.0.translation.z
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

            if let Some(col) = world.get::<Collider>(entity) {
                ui.collapsing("Collider", |ui| {
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
