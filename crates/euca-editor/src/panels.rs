use euca_ecs::{Entity, Query, World};
use euca_physics::{Collider, PhysicsBody, Velocity};
use euca_reflect::Reflect;
use euca_render::{MaterialRef, MeshRenderer};
use euca_scene::{GlobalTransform, LocalTransform};

use crate::EditorState;

/// Display a component using the Reflect trait — generic, no hardcoding.
/// Shows the component name as a collapsible header with field names + values.
fn reflect_component<T: 'static + Send + Sync + Reflect>(
    ui: &mut egui::Ui,
    world: &World,
    entity: Entity,
) {
    if let Some(component) = world.get::<T>(entity) {
        let name = component.type_name();
        let fields = component.fields();
        ui.collapsing(name, |ui| {
            for (field_name, value) in &fields {
                ui.label(format!("{field_name}: {value}"));
            }
        });
    }
}

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
            }

            if !state.playing && ui.button("🔄 Reset").clicked() {
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

            // ── Reflection-driven component display ──
            // All components implementing Reflect are shown automatically
            // via their field names + values. No hardcoding per component.
            reflect_component::<GlobalTransform>(ui, world, entity);
            reflect_component::<MeshRenderer>(ui, world, entity);
            reflect_component::<MaterialRef>(ui, world, entity);
            reflect_component::<PhysicsBody>(ui, world, entity);
            reflect_component::<Collider>(ui, world, entity);
            reflect_component::<Velocity>(ui, world, entity);
        });
}

fn find_alive_entity(world: &World, index: u32) -> Option<Entity> {
    crate::find_alive_entity(world, index)
}
