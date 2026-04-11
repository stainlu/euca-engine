use euca_ecs::{Entity, Query, World};
use euca_physics::{Collider, PhysicsBody, Velocity};
use euca_reflect::Reflect;
use euca_render::{MaterialRef, MeshRenderer};
use euca_scene::{GlobalTransform, LocalTransform};

use crate::EditorState;

/// Display a component using the Reflect trait — generic, no hardcoding.
/// Shows the component name as a collapsible header with field names + values.
fn reflect_component<T: 'static + Send + Sync + Clone + Reflect>(
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

/// Top toolbar: Play/Pause/Step controls + Level selector + Save/Load + info + FPS.
pub fn toolbar_panel(
    ctx: &egui::Context,
    state: &mut EditorState,
    world: &World,
    delta_time: f32,
    available_levels: &[String],
    selected_level: &mut Option<usize>,
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

            // Level selector dropdown (disabled during play)
            ui.add_enabled_ui(!state.playing, |ui| {
                let current_label = selected_level
                    .and_then(|i| available_levels.get(i))
                    .map(|s| s.as_str())
                    .unwrap_or("(No level)");

                egui::ComboBox::from_id_salt("level_select")
                    .selected_text(current_label)
                    .width(160.0)
                    .show_ui(ui, |ui| {
                        // "(No level)" option
                        if ui
                            .selectable_label(selected_level.is_none(), "(No level)")
                            .clicked()
                        {
                            *selected_level = None;
                        }
                        // Available level files
                        for (i, name) in available_levels.iter().enumerate() {
                            let is_selected = *selected_level == Some(i);
                            if ui.selectable_label(is_selected, name).clicked() {
                                *selected_level = Some(i);
                            }
                        }
                    });
            });

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

/// Entity spawn request from the hierarchy or content browser panel.
#[derive(Clone, Debug)]
pub enum SpawnRequest {
    Empty,
    Cube,
    Sphere,
    Plane,
    Cylinder,
    Cone,
}

/// Left panel: entity hierarchy list.
///
/// `shift_held` controls multi-select behaviour: normal click replaces
/// selection, shift-click adds to the existing selection.
pub fn hierarchy_panel(
    ctx: &egui::Context,
    state: &mut EditorState,
    world: &World,
    shift_held: bool,
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
                    let selected = state.is_selected(entity.index());

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
                        if shift_held {
                            state.add_select(entity.index());
                        } else {
                            state.select(entity.index());
                        }
                    }
                }
            });
        });
    spawn
}

/// Right panel: inspector for the selected entity.
/// Returns `true` if a transform was changed via the inspector (for dirty tracking).
pub fn inspector_panel(ctx: &egui::Context, state: &mut EditorState, world: &mut World) -> bool {
    let mut transform_changed = false;
    egui::SidePanel::right("inspector")
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.heading("Inspector");
            ui.separator();

            // Multiple selection: show count label instead of component details
            if state.selected_entities.len() > 1 {
                ui.label(format!(
                    "{} entities selected",
                    state.selected_entities.len()
                ));
                return;
            }

            let entity_idx = match state.primary_selected() {
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
                    state.clear_selection();
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
                    transform_changed = true;
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
    transform_changed
}

/// Bottom panel: content browser showing built-in meshes to spawn.
pub fn content_browser_panel(ctx: &egui::Context, _state: &EditorState) -> Option<SpawnRequest> {
    let mut spawn = None;
    egui::TopBottomPanel::bottom("content_browser")
        .default_height(60.0)
        .show(ctx, |ui| {
            ui.heading("Content Browser");
            ui.horizontal(|ui| {
                let meshes: &[(&str, SpawnRequest)] = &[
                    ("Cube", SpawnRequest::Cube),
                    ("Sphere", SpawnRequest::Sphere),
                    ("Plane", SpawnRequest::Plane),
                    ("Cylinder", SpawnRequest::Cylinder),
                    ("Cone", SpawnRequest::Cone),
                ];
                for (label, request) in meshes {
                    if ui.button(*label).clicked() {
                        spawn = Some(request.clone());
                    }
                }
            });
        });
    spawn
}

/// Which sculpting operation the terrain brush performs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TerrainBrushMode {
    Raise,
    Lower,
    Flatten,
    Smooth,
}

/// Snapshot of the current terrain brush settings, returned each frame the
/// brush is active so the caller can apply the sculpt operation.
#[derive(Clone, Debug)]
pub struct TerrainBrushAction {
    pub mode: TerrainBrushMode,
    pub radius: f32,
    pub strength: f32,
    pub target_height: f32,
}

/// Terrain brush editor panel.
///
/// Renders inside the right-side inspector area as a collapsible section.
/// Returns `Some(TerrainBrushAction)` when the brush is active, `None`
/// otherwise.
pub fn terrain_panel(ctx: &egui::Context, state: &mut EditorState) -> Option<TerrainBrushAction> {
    let mut action = None;

    egui::SidePanel::right("terrain_brush")
        .default_width(260.0)
        .show(ctx, |ui| {
            ui.heading("Terrain Brush");
            ui.separator();

            // Active toggle
            ui.checkbox(&mut state.terrain_brush_active, "Brush Active");
            ui.separator();

            // Mode selection
            ui.label("Mode:");
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(state.terrain_brush_mode == TerrainBrushMode::Raise, "Raise")
                    .clicked()
                {
                    state.terrain_brush_mode = TerrainBrushMode::Raise;
                }
                if ui
                    .selectable_label(state.terrain_brush_mode == TerrainBrushMode::Lower, "Lower")
                    .clicked()
                {
                    state.terrain_brush_mode = TerrainBrushMode::Lower;
                }
                if ui
                    .selectable_label(
                        state.terrain_brush_mode == TerrainBrushMode::Flatten,
                        "Flatten",
                    )
                    .clicked()
                {
                    state.terrain_brush_mode = TerrainBrushMode::Flatten;
                }
                if ui
                    .selectable_label(
                        state.terrain_brush_mode == TerrainBrushMode::Smooth,
                        "Smooth",
                    )
                    .clicked()
                {
                    state.terrain_brush_mode = TerrainBrushMode::Smooth;
                }
            });

            ui.separator();

            // Radius
            ui.horizontal(|ui| {
                ui.label("Radius:");
                ui.add(
                    egui::DragValue::new(&mut state.terrain_brush_radius)
                        .speed(0.1)
                        .range(1.0..=50.0),
                );
            });

            // Strength
            ui.horizontal(|ui| {
                ui.label("Strength:");
                ui.add(
                    egui::DragValue::new(&mut state.terrain_brush_strength)
                        .speed(0.005)
                        .range(0.01..=1.0),
                );
            });

            // Target height (relevant for Flatten mode)
            ui.horizontal(|ui| {
                ui.label("Target Height:");
                ui.add(
                    egui::DragValue::new(&mut state.terrain_brush_target_height)
                        .speed(0.1)
                        .range(0.0..=100.0),
                );
            });

            if state.terrain_brush_active {
                action = Some(TerrainBrushAction {
                    mode: state.terrain_brush_mode.clone(),
                    radius: state.terrain_brush_radius,
                    strength: state.terrain_brush_strength,
                    target_height: state.terrain_brush_target_height,
                });
            }
        });

    action
}

fn find_alive_entity(world: &World, index: u32) -> Option<Entity> {
    crate::find_alive_entity(world, index)
}
