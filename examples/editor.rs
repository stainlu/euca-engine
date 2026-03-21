use euca_agent::{
    AgentServer, CameraOverride, EngineControl, ScreenshotChannel,
    auth::AuthStore,
    hud::{HudCanvas, HudElement, parse_color},
};
use euca_core::{Profiler, Time, profiler_begin, profiler_end};
use euca_ecs::Events;
use euca_ecs::{Query, Schedule, SharedWorld, World};
use euca_editor::{
    EditorState, SceneFile, SpawnRequest, ToolbarAction, find_alive_entity, hierarchy_panel,
    inspector_panel, toolbar_panel,
};
use euca_math::{Transform, Vec3};
use euca_physics::{
    Collider, PhysicsBody, PhysicsConfig, Ray, physics_step_system, raycast_collider,
};
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};

use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

const AGENT_PORT: u16 = 3917;

// ---------------------------------------------------------------------------
// Helper functions extracted from EditorApp to keep methods focused
// ---------------------------------------------------------------------------

/// Insert every default resource the editor world requires at startup.
fn setup_default_resources(world: &mut World) {
    world.insert_resource(Time::new());
    world.insert_resource(Camera::new(
        Vec3::new(8.0, 6.0, 8.0),
        Vec3::new(0.0, 1.0, 0.0),
    ));
    world.insert_resource(PhysicsConfig::new());
    world.insert_resource(AmbientLight {
        color: [1.0, 1.0, 1.0],
        intensity: 0.2,
    });
    world.insert_resource(EngineControl::new());
    world.insert_resource(ScreenshotChannel::new());
    world.insert_resource(AuthStore::new());
    world.insert_resource(CameraOverride::new());
    world.insert_resource(Events::default());
    world.insert_resource(HudCanvas::new());
    world.insert_resource(euca_agent::routes::TemplateRegistry::new());
    world.insert_resource(euca_asset::AnimationLibrary::default());
    world.insert_resource(euca_input::ActionMap::new());
    world.insert_resource(euca_input::InputContextStack::new());
    world.insert_resource(euca_scene::SpatialIndex::new(2.0));
    world.insert_resource(euca_scene::PrefabRegistry::default());
    world.insert_resource(LodSettings::default());
    world.insert_resource(PostProcessSettings::default());
    world.insert_resource(Profiler::default());
    // AudioEngine init may fail on headless systems — log and continue
    match euca_audio::AudioEngine::new() {
        Ok(engine) => world.insert_resource(engine),
        Err(e) => log::warn!("Audio init failed (non-fatal): {e}"),
    }
}

/// Run all gameplay systems for a single simulation tick.
///
/// Called once per frame when the editor is in play mode (or stepping).
/// Order: physics, gameplay, combat, rules, economy, audio/animation.
fn run_gameplay_systems(world: &mut World, dt: f32) {
    if let Some(p) = world.resource_mut::<Profiler>() {
        profiler_begin(p, "physics");
    }
    physics_step_system(world);
    euca_physics::character_controller_system(world, dt);
    euca_physics::vehicle_physics_system(world, dt);
    if let Some(p) = world.resource_mut::<Profiler>() {
        profiler_end(p);
    }

    if let Some(p) = world.resource_mut::<Profiler>() {
        profiler_begin(p, "gameplay");
    }
    euca_gameplay::apply_damage_system(world);
    euca_gameplay::death_check_system(world);
    euca_gameplay::projectile_system(world, dt);
    euca_gameplay::trigger_system(world);
    euca_gameplay::ai_system(world, dt);
    if let Some(p) = world.resource_mut::<Profiler>() {
        profiler_end(p);
    }

    if let Some(p) = world.resource_mut::<Profiler>() {
        profiler_begin(p, "combat");
    }
    euca_gameplay::auto_combat_system(world, dt);
    euca_gameplay::game_state_system(world, dt);
    if let Some(p) = world.resource_mut::<Profiler>() {
        profiler_end(p);
    }

    if let Some(p) = world.resource_mut::<Profiler>() {
        profiler_begin(p, "rules");
    }
    euca_gameplay::on_death_rule_system(world);
    euca_gameplay::timer_rule_system(world, dt);
    euca_gameplay::health_below_rule_system(world);
    euca_gameplay::on_score_rule_system(world);
    euca_gameplay::on_phase_rule_system(world);

    let respawn_delay = world
        .resource::<euca_gameplay::GameState>()
        .map(|s| s.config.respawn_delay);
    if let Some(delay) = respawn_delay {
        euca_gameplay::respawn_system(world, dt);
        euca_gameplay::start_respawn_on_death(world, delay);
    }
    euca_gameplay::corpse_cleanup_system(world, dt);

    // Attach visuals to rule-spawned entities
    let spawn_events: Vec<euca_gameplay::RuleSpawnEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<euca_gameplay::RuleSpawnEvent>().cloned().collect())
        .unwrap_or_default();
    if let Some(assets) = world
        .resource::<euca_agent::routes::DefaultAssets>()
        .cloned()
    {
        for ev in spawn_events {
            if let Some(mesh_handle) = assets.mesh(&ev.mesh) {
                world.insert(ev.entity, euca_render::MeshRenderer { mesh: mesh_handle });
                let mat = ev
                    .color
                    .as_deref()
                    .and_then(|c| assets.material(c))
                    .unwrap_or(assets.default_material);
                world.insert(ev.entity, euca_render::MaterialRef { handle: mat });
            }
        }
    }

    euca_gameplay::gold_on_kill_system(world);
    euca_gameplay::xp_on_kill_system(world);
    euca_gameplay::ability_tick_system(world, dt);
    euca_gameplay::use_ability_system(world);
    if let Some(p) = world.resource_mut::<Profiler>() {
        profiler_end(p);
    }

    if let Some(p) = world.resource_mut::<Profiler>() {
        profiler_begin(p, "audio");
    }
    euca_audio::audio_update_system_mut(world, dt);
    euca_asset::skeletal_animation_system(world, dt);
    euca_particle::emit_particles_system(world, dt);
    euca_particle::particle_update_system(world, dt);
    euca_nav::pathfinding_system(world);
    euca_nav::steering_system(world, dt);
    if let Some(p) = world.resource_mut::<Profiler>() {
        profiler_end(p);
    }

    if let Some(events) = world.resource_mut::<Events>() {
        events.update();
    }

    // Advance tick counter — required for change detection
    world.tick();
}

/// Collect base draw commands for all alive renderable entities.
fn collect_draw_commands(world: &World) -> Vec<DrawCommand> {
    let query = Query::<(
        euca_ecs::Entity,
        &GlobalTransform,
        &MeshRenderer,
        &MaterialRef,
    )>::new(world);
    query
        .iter()
        .filter(|(e, _, _, _)| world.get::<euca_gameplay::Dead>(*e).is_none())
        .map(|(_, gt, mr, mat)| DrawCommand {
            mesh: mr.mesh,
            material: mat.handle,
            model_matrix: gt.0.to_matrix(),
            aabb: None,
        })
        .collect()
}

/// Append a selection outline (1.06x scale, orange material) for the selected entity.
fn append_selection_outline(
    world: &World,
    selected: Option<u32>,
    outline_mat: Option<MaterialHandle>,
    cmds: &mut Vec<DrawCommand>,
) {
    let (Some(sel_idx), Some(mat)) = (selected, outline_mat) else {
        return;
    };
    for g in 0..16u32 {
        let entity = euca_ecs::Entity::from_raw(sel_idx, g);
        if !world.is_alive(entity) {
            continue;
        }
        if let (Some(gt), Some(mr)) = (
            world.get::<GlobalTransform>(entity),
            world.get::<MeshRenderer>(entity),
        ) {
            let max_scale = gt.0.scale.x.max(gt.0.scale.y).max(gt.0.scale.z);
            if max_scale < 5.0 {
                let mut t = gt.0;
                t.scale = t.scale * 1.06;
                cmds.push(DrawCommand {
                    mesh: mr.mesh,
                    material: mat,
                    model_matrix: t.to_matrix(),
                    aabb: None,
                });
            }
        }
        break;
    }
}

/// Append gizmo axis handle draw commands for the selected entity.
fn append_gizmo_commands(world: &World, editor_state: &EditorState, cmds: &mut Vec<DrawCommand>) {
    let Some(sel_idx) = editor_state.selected_entity else {
        return;
    };
    let Some(entity) = find_alive_entity(world, sel_idx) else {
        return;
    };
    if let Some(gt) = world.get::<GlobalTransform>(entity) {
        let camera = world.resource::<Camera>().unwrap();
        cmds.extend(euca_editor::gizmo::gizmo_draw_commands(
            gt.0.translation,
            camera.eye,
            &editor_state.gizmo,
        ));
    }
}

/// Append foliage instancing draw commands from all visible layers.
fn append_foliage_instances(world: &World, gpu: &GpuContext, cmds: &mut Vec<DrawCommand>) {
    let Some(foliage_layers) = world.resource::<FoliageLayers>() else {
        return;
    };
    let camera = world.resource::<Camera>().unwrap();
    let aspect = gpu.surface_config.width as f32 / gpu.surface_config.height as f32;
    let vp = camera.view_projection_matrix(aspect);
    let frustum = Frustum::from_view_projection(&vp);
    for layer in &foliage_layers.layers {
        for model_matrix in FoliageRenderer::collect_visible_instances(layer, camera.eye, &frustum)
        {
            cmds.push(DrawCommand {
                mesh: layer.mesh,
                material: layer.material,
                model_matrix,
                aabb: None,
            });
        }
    }
}

/// Draw world-space health bars above entities that have a `Health` component.
fn draw_health_bars(ctx: &egui::Context, world: &World, aspect: f32) {
    let vp = ctx.available_rect();
    let mut painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("health_bars"),
    ));
    painter.set_clip_rect(vp);

    let Some(cam) = world.resource::<Camera>().cloned() else {
        return;
    };
    let view_proj = cam.view_projection_matrix(aspect);

    let hp_entities: Vec<(Vec3, f32, u8)> = {
        let query =
            Query::<(euca_ecs::Entity, &GlobalTransform, &euca_gameplay::Health)>::new(world);
        query
            .iter()
            .filter(|(e, _, h)| !h.is_dead() && world.get::<euca_gameplay::Dead>(*e).is_none())
            .map(|(e, gt, h)| {
                let team = world
                    .get::<euca_gameplay::Team>(e)
                    .map(|t| t.0)
                    .unwrap_or(0);
                (gt.0.translation, h.fraction(), team)
            })
            .collect()
    };

    for (world_pos, fraction, team) in &hp_entities {
        let offset_pos = *world_pos + Vec3::new(0.0, 1.2, 0.0);
        let clip = view_proj * euca_math::Vec4::new(offset_pos.x, offset_pos.y, offset_pos.z, 1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let screen_x = vp.min.x + (ndc_x * 0.5 + 0.5) * vp.width();
        let screen_y = vp.min.y + (1.0 - (ndc_y * 0.5 + 0.5)) * vp.height();
        let bar_w = 40.0;
        let bar_h = 5.0;
        let bar_rect = egui::Rect::from_min_size(
            egui::pos2(screen_x - bar_w / 2.0, screen_y),
            egui::vec2(bar_w, bar_h),
        );
        painter.rect_filled(
            bar_rect,
            2.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160),
        );
        let fill_rect =
            egui::Rect::from_min_size(bar_rect.min, egui::vec2(bar_w * fraction, bar_h));
        let bar_color = if *team == 1 {
            egui::Color32::from_rgb(220, 50, 50)
        } else {
            egui::Color32::from_rgb(50, 100, 220)
        };
        painter.rect_filled(fill_rect, 2.0, bar_color);
    }
}

/// Draw the in-game HUD overlay (text, bars, rects) inside the 3D viewport.
fn draw_hud_overlay(ctx: &egui::Context, world: &World) {
    let Some(canvas) = world.resource::<HudCanvas>() else {
        return;
    };
    let vp = ctx.available_rect();
    let mut painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("hud"),
    ));
    painter.set_clip_rect(vp);

    let vp_pos =
        |hx: f32, hy: f32| egui::pos2(vp.min.x + hx * vp.width(), vp.min.y + hy * vp.height());
    let vp_size = |hw: f32, hh: f32| egui::vec2(hw * vp.width(), hh * vp.height());
    let to_color = |rgba: [f32; 4]| {
        egui::Color32::from_rgba_unmultiplied(
            (rgba[0] * 255.0) as u8,
            (rgba[1] * 255.0) as u8,
            (rgba[2] * 255.0) as u8,
            (rgba[3] * 255.0) as u8,
        )
    };

    for element in &canvas.elements {
        match element {
            HudElement::Text {
                text,
                x,
                y,
                size,
                color,
            } => {
                painter.text(
                    vp_pos(*x, *y),
                    egui::Align2::CENTER_TOP,
                    text,
                    egui::FontId::proportional(*size),
                    to_color(parse_color(color)),
                );
            }
            HudElement::Bar {
                x,
                y,
                width,
                height,
                fill,
                color,
            } => {
                let rect = egui::Rect::from_min_size(vp_pos(*x, *y), vp_size(*width, *height));
                painter.rect_filled(
                    rect,
                    2.0,
                    egui::Color32::from_rgba_unmultiplied(20, 20, 20, 180),
                );
                let fill_rect = egui::Rect::from_min_size(
                    rect.min,
                    egui::vec2(rect.width() * fill.clamp(0.0, 1.0), rect.height()),
                );
                painter.rect_filled(fill_rect, 2.0, to_color(parse_color(color)));
            }
            HudElement::Rect {
                x,
                y,
                width,
                height,
                color,
            } => {
                let rect = egui::Rect::from_min_size(vp_pos(*x, *y), vp_size(*width, *height));
                painter.rect_filled(rect, 0.0, to_color(parse_color(color)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// EditorApp
// ---------------------------------------------------------------------------

struct EditorApp {
    shared: SharedWorld,
    survey: HardwareSurvey,
    wgpu_instance: wgpu::Instance,
    editor_state: EditorState,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    egui_ctx: egui::Context,
    egui_winit: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    window_attrs: WindowAttributes,
    mouse_pos: [f32; 2],
    mouse_delta: [f32; 2],
    right_mouse_down: bool,
    middle_mouse_down: bool,
    cam_yaw: f32,
    cam_pitch: f32,
    cam_distance: f32,
    cam_target: Vec3,
    outline_material: Option<MaterialHandle>,
    cube_mesh: Option<MeshHandle>,
    sphere_mesh: Option<MeshHandle>,
    default_material: Option<MaterialHandle>,
    ctrl_held: bool,
    shift_held: bool,
    _tokio_rt: Option<tokio::runtime::Runtime>,
}

impl EditorApp {
    fn new() -> Self {
        let (survey, wgpu_instance) = HardwareSurvey::detect();
        let mut world = World::new();
        setup_default_resources(&mut world);
        let shared = SharedWorld::new(world, Schedule::new());

        Self {
            shared,
            survey,
            wgpu_instance,
            editor_state: EditorState::new(),
            window: None,
            gpu: None,
            renderer: None,
            egui_ctx: egui::Context::default(),
            egui_winit: None,
            egui_renderer: None,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — Editor")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 800)),
            mouse_pos: [0.0, 0.0],
            mouse_delta: [0.0, 0.0],
            right_mouse_down: false,
            middle_mouse_down: false,
            cam_yaw: 0.6,
            cam_pitch: 0.35,
            cam_distance: 14.0,
            cam_target: Vec3::new(0.0, 1.5, 0.0),
            outline_material: None,
            cube_mesh: None,
            sphere_mesh: None,
            default_material: None,
            ctrl_held: false,
            shift_held: false,
            _tokio_rt: None,
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        let cube = renderer.upload_mesh(gpu, &Mesh::cube());
        let sphere = renderer.upload_mesh(gpu, &Mesh::sphere(0.5, 16, 32));
        let plane = renderer.upload_mesh(gpu, &Mesh::plane(20.0));
        self.cube_mesh = Some(cube);
        self.sphere_mesh = Some(sphere);
        self.editor_state.gizmo = euca_editor::gizmo::init_gizmo(renderer, gpu, cube);

        let grid_tex = renderer.checkerboard_texture(gpu, 512, 32);
        let grid_mat = renderer.upload_material(
            gpu,
            &Material::new([0.45, 0.45, 0.45, 1.0], 0.0, 0.95).with_texture(grid_tex),
        );

        // Upload material palette for agent use (table-driven)
        let palette: &[(&str, Material)] = &[
            ("blue", Material::blue_plastic()),
            ("red", Material::red_plastic()),
            ("green", Material::green()),
            ("gold", Material::gold()),
            ("silver", Material::silver()),
            ("gray", Material::gray()),
            ("white", Material::new([1.0, 1.0, 1.0, 1.0], 0.0, 0.5)),
            ("black", Material::new([0.05, 0.05, 0.05, 1.0], 0.0, 0.5)),
            ("yellow", Material::new([1.0, 1.0, 0.0, 1.0], 0.0, 0.4)),
            ("cyan", Material::new([0.0, 0.9, 0.9, 1.0], 0.0, 0.4)),
            ("magenta", Material::new([0.9, 0.0, 0.9, 1.0], 0.0, 0.4)),
            ("orange", Material::new([1.0, 0.5, 0.0, 1.0], 0.0, 0.4)),
        ];
        let mut materials = std::collections::HashMap::new();
        let mut blue = None;
        for (name, mat) in palette {
            let h = renderer.upload_material(gpu, mat);
            if *name == "blue" {
                blue = Some(h);
            }
            materials.insert((*name).to_string(), h);
        }
        let blue = blue.expect("blue material must be in palette");
        self.default_material = Some(blue);

        self.outline_material =
            Some(renderer.upload_material(gpu, &Material::new([1.0, 0.6, 0.0, 1.0], 0.0, 1.0)));

        let mut meshes = std::collections::HashMap::new();
        meshes.insert("cube".to_string(), cube);
        meshes.insert("sphere".to_string(), sphere);
        meshes.insert("plane".to_string(), plane);

        let mut pool = self.shared.lock();
        let world = pool.world();
        world.insert_resource(euca_agent::routes::DefaultAssets {
            meshes,
            materials,
            default_material: blue,
        });

        // Ground plane (Persistent — survives reset)
        let g = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(g, GlobalTransform::default());
        world.insert(g, MeshRenderer { mesh: plane });
        world.insert(g, MaterialRef { handle: grid_mat });
        world.insert(g, PhysicsBody::fixed());
        world.insert(g, Collider::aabb(10.0, 0.01, 10.0));
        world.insert(g, euca_agent::Persistent);

        // Directional light — warm sunlight (Persistent)
        let light = world.spawn(DirectionalLight {
            direction: [0.4, -0.9, 0.25],
            color: [1.0, 0.95, 0.88],
            intensity: 2.5,
        });
        world.insert(light, euca_agent::Persistent);
    }

    fn reset_scene(&mut self) {
        {
            let mut pool = self.shared.lock();
            let world = pool.world();
            let entities: Vec<euca_ecs::Entity> = {
                let query = euca_ecs::Query::<euca_ecs::Entity>::new(world);
                query.iter().collect()
            };
            for entity in entities {
                world.despawn(entity);
            }
            world.insert_resource(PhysicsConfig::new());
        }
        self.setup_scene();
        self.editor_state.selected_entity = None;
    }

    fn render_frame(&mut self) {
        if self.editor_state.reset_requested {
            self.editor_state.reset_requested = false;
            self.reset_scene();
        }

        let mut pool = self.shared.lock();
        let world = pool.world();
        world.resource_mut::<Time>().unwrap().update();
        let _elapsed = world.resource::<Time>().unwrap().elapsed as f32;

        if let Some(ctrl) = world.resource::<EngineControl>() {
            self.editor_state.playing = ctrl.is_playing();
            if ctrl.take_step_request() {
                self.editor_state.step_once = true;
            }
        }

        if self.editor_state.should_tick() {
            let dt = world
                .resource::<Time>()
                .map(|t| t.delta as f32)
                .unwrap_or(0.016);
            run_gameplay_systems(world, dt);
        }
        euca_scene::transform_propagation_system(world);
        euca_scene::spatial_index_update_system(world);

        let camera_overridden = world
            .resource::<CameraOverride>()
            .map(|co| co.take())
            .unwrap_or(false);
        if !camera_overridden {
            if self.right_mouse_down {
                self.cam_yaw += self.mouse_delta[0] * 0.005;
                self.cam_pitch = (self.cam_pitch - self.mouse_delta[1] * 0.005).clamp(0.05, 1.5);
            }
            if self.middle_mouse_down {
                let right = Vec3::new(self.cam_yaw.cos(), 0.0, -self.cam_yaw.sin());
                let up = Vec3::Y;
                self.cam_target = self.cam_target
                    + right * (-self.mouse_delta[0] * 0.01 * self.cam_distance * 0.1);
                self.cam_target =
                    self.cam_target + up * (self.mouse_delta[1] * 0.01 * self.cam_distance * 0.1);
            }
            let cam = world.resource_mut::<Camera>().unwrap();
            cam.eye = Vec3::new(
                self.cam_target.x + self.cam_yaw.sin() * self.cam_pitch.cos() * self.cam_distance,
                self.cam_target.y + self.cam_pitch.sin() * self.cam_distance,
                self.cam_target.z + self.cam_yaw.cos() * self.cam_pitch.cos() * self.cam_distance,
            );
            cam.target = self.cam_target;
        }
        self.mouse_delta = [0.0, 0.0];
        lod_select_system(world);

        let gpu = self.gpu.as_ref().unwrap();
        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("editor frame"),
            });

        // === 1. Render 3D scene ===
        if let Some(p) = world.resource_mut::<Profiler>() {
            profiler_begin(p, "render_collect");
        }
        let mut draw_commands = collect_draw_commands(world);
        append_selection_outline(
            world,
            self.editor_state.selected_entity,
            self.outline_material,
            &mut draw_commands,
        );
        append_gizmo_commands(world, &self.editor_state, &mut draw_commands);
        append_foliage_instances(world, gpu, &mut draw_commands);

        let light = {
            let query = Query::<&DirectionalLight>::new(world);
            query.iter().next().cloned().unwrap_or_default()
        };
        let ambient = world
            .resource::<AmbientLight>()
            .cloned()
            .unwrap_or_default();
        let camera = world.resource::<Camera>().unwrap().clone();
        if let Some(p) = world.resource_mut::<Profiler>() {
            profiler_end(p);
        }

        if let Some(p) = world.resource_mut::<Profiler>() {
            profiler_begin(p, "render_draw");
        }
        let renderer = self.renderer.as_mut().unwrap();
        renderer.render_to_view(
            gpu,
            &camera,
            &light,
            &ambient,
            &draw_commands,
            &view,
            &mut encoder,
        );
        if let Some(p) = world.resource_mut::<Profiler>() {
            profiler_end(p);
        }

        let screenshot_tx = world
            .resource::<ScreenshotChannel>()
            .and_then(|ch| ch.take());

        // === 2. Render egui on top ===
        let window = self.window.as_ref().unwrap();
        let egui_winit = self.egui_winit.as_mut().unwrap();
        let raw_input = egui_winit.take_egui_input(window);
        let aspect = gpu.surface_config.width as f32 / gpu.surface_config.height as f32;

        let mut spawn_request = None;
        let mut toolbar_action = None;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            let dt = world.resource::<Time>().map(|t| t.delta).unwrap_or(0.0);
            toolbar_action = toolbar_panel(ctx, &mut self.editor_state, world, dt);
            spawn_request = hierarchy_panel(ctx, &mut self.editor_state, world);
            inspector_panel(ctx, &mut self.editor_state, world);
            draw_health_bars(ctx, world, aspect);
            draw_hud_overlay(ctx, world);
        });

        if let Some(ctrl) = world.resource::<EngineControl>() {
            ctrl.set_playing(self.editor_state.playing);
        }

        if let Some(action) = toolbar_action {
            match action {
                ToolbarAction::SaveScene => {
                    let scene = SceneFile::capture(world);
                    if let Err(e) = scene.save("scene.json") {
                        log::error!("Save failed: {e}");
                    } else {
                        log::info!("Scene saved to scene.json");
                    }
                }
                ToolbarAction::LoadScene => match SceneFile::load("scene.json") {
                    Ok(scene) => {
                        log::info!(
                            "Scene loaded: {} entities from scene.json",
                            scene.entities.len()
                        );
                        let entities: Vec<euca_ecs::Entity> = {
                            let query = Query::<euca_ecs::Entity>::new(world);
                            query.iter().collect()
                        };
                        for entity in entities {
                            world.despawn(entity);
                        }
                        let cube_mesh = self.cube_mesh;
                        let sphere_mesh = self.sphere_mesh;
                        euca_editor::load_scene_into_world(
                            world,
                            &scene,
                            &|name| match name {
                                n if n.contains("0") => cube_mesh,
                                n if n.contains("1") => sphere_mesh,
                                _ => cube_mesh,
                            },
                            6,
                        );
                        world.spawn(DirectionalLight {
                            direction: [0.5, -1.0, 0.3],
                            color: [1.0, 0.98, 0.95],
                            intensity: 2.0,
                        });
                        self.editor_state.selected_entity = None;
                    }
                    Err(e) => log::error!("Load failed: {e}"),
                },
            }
        }

        if let Some(req) = spawn_request {
            let pos = Vec3::new(0.0, 2.0, 0.0);
            let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
            world.insert(e, GlobalTransform::default());
            match req {
                SpawnRequest::Cube => {
                    if let Some(mesh) = self.cube_mesh {
                        world.insert(e, MeshRenderer { mesh });
                    }
                    if let Some(mat) = self.default_material {
                        world.insert(e, MaterialRef { handle: mat });
                    }
                    world.insert(e, Collider::aabb(0.5, 0.5, 0.5));
                }
                SpawnRequest::Sphere => {
                    if let Some(mesh) = self.sphere_mesh {
                        world.insert(e, MeshRenderer { mesh });
                    }
                    if let Some(mat) = self.default_material {
                        world.insert(e, MaterialRef { handle: mat });
                    }
                    world.insert(e, Collider::sphere(0.5));
                }
                SpawnRequest::Empty => {}
            }
            self.editor_state.selected_entity = Some(e.index());
            self.editor_state
                .undo
                .push(euca_editor::undo::UndoAction::SpawnEntity {
                    entity_index: e.index(),
                });
        }

        if let Some(p) = world.resource_mut::<Profiler>() {
            p.end_frame();
        }
        drop(pool);

        egui_winit.handle_platform_output(window, full_output.platform_output);
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.surface_config.width, gpu.surface_config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        let egui_renderer = self.egui_renderer.as_mut().unwrap();
        for (id, delta) in &full_output.textures_delta.set {
            egui_renderer.update_texture(&gpu.device, &gpu.queue, *id, delta);
        }
        let user_bufs = egui_renderer.update_buffers(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &paint_jobs,
            &screen_desc,
        );
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            egui_renderer.render(&mut pass.forget_lifetime(), &paint_jobs, &screen_desc);
        }
        let mut cmds: Vec<wgpu::CommandBuffer> = vec![encoder.finish()];
        cmds.extend(user_bufs);
        gpu.queue.submit(cmds);
        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }

        if let Some(tx) = screenshot_tx {
            let read_pool = self.shared.lock_read();
            let w = read_pool.world();
            let draw_cmds: Vec<DrawCommand> = {
                let query = Query::<(&GlobalTransform, &MeshRenderer, &MaterialRef)>::new(w);
                query
                    .iter()
                    .map(|(gt, mr, mat)| DrawCommand {
                        mesh: mr.mesh,
                        material: mat.handle,
                        model_matrix: gt.0.to_matrix(),
                        aabb: None,
                    })
                    .collect()
            };
            let light = {
                let query = Query::<&DirectionalLight>::new(w);
                query.iter().next().cloned().unwrap_or_default()
            };
            let ambient = w.resource::<AmbientLight>().cloned().unwrap_or_default();
            let camera = w.resource::<Camera>().unwrap().clone();
            drop(read_pool);
            let renderer = self.renderer.as_mut().unwrap();
            capture_screenshot(gpu, renderer, &camera, &light, &ambient, &draw_cmds, tx);
        }
        output.present();
    }
}

impl ApplicationHandler for EditorApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop.create_window(self.window_attrs.clone()).unwrap();
            let gpu = GpuContext::new(window, &self.survey, &self.wgpu_instance);
            let renderer = Renderer::new(&gpu);
            let egui_winit = egui_winit::State::new(
                self.egui_ctx.clone(),
                egui::ViewportId::ROOT,
                &*gpu.window,
                Some(gpu.window.scale_factor() as f32),
                None,
                None,
            );
            let egui_renderer = egui_wgpu::Renderer::new(
                &gpu.device,
                gpu.surface_config.format,
                egui_wgpu::RendererOptions::default(),
            );
            self.window = Some(gpu.window.clone());
            self.gpu = Some(gpu);
            self.renderer = Some(renderer);
            self.egui_winit = Some(egui_winit);
            self.egui_renderer = Some(egui_renderer);
            self.setup_scene();

            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime");
            let server = AgentServer::from_shared(self.shared.clone(), AGENT_PORT);
            rt.spawn(async move {
                server.run().await;
            });
            log::info!("Agent server started on port {AGENT_PORT}");
            self._tokio_rt = Some(rt);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if let Some(egui_winit) = &mut self.egui_winit {
            let resp = egui_winit.on_window_event(self.window.as_ref().unwrap(), &event);
            if resp.consumed {
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Delete),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(idx) = self.editor_state.selected_entity {
                    let mut pool = self.shared.lock();
                    let world = pool.world();
                    if let Some(e) = find_alive_entity(world, idx) {
                        let transform = world
                            .get::<LocalTransform>(e)
                            .map(|lt| lt.0)
                            .unwrap_or_default();
                        let mesh = world.get::<MeshRenderer>(e).map(|mr| mr.mesh);
                        let material = world.get::<MaterialRef>(e).map(|mr| mr.handle);
                        let collider = world.get::<Collider>(e).cloned();
                        world.despawn(e);
                        self.editor_state
                            .undo
                            .push(euca_editor::undo::UndoAction::DespawnEntity {
                                entity_index: idx,
                                transform,
                                mesh,
                                material,
                                collider,
                            });
                    }
                    self.editor_state.selected_entity = None;
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref ch),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if ch.as_str() == "f" || ch.as_str() == "F" => {
                if let Some(idx) = self.editor_state.selected_entity {
                    let pool = self.shared.lock_read();
                    let world = pool.world();
                    if let Some(e) = find_alive_entity(world, idx) {
                        if let Some(gt) = world.get::<GlobalTransform>(e) {
                            self.cam_target = gt.0.translation;
                            self.cam_distance = 5.0;
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref ch),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if ch.as_str() == "z" && self.ctrl_held && !self.shift_held => {
                let mut pool = self.shared.lock();
                self.editor_state.undo.undo(pool.world());
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref ch),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if (ch.as_str() == "y" && self.ctrl_held)
                || (ch.as_str() == "z" && self.ctrl_held && self.shift_held) =>
            {
                let mut pool = self.shared.lock();
                self.editor_state.undo.redo(pool.world());
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.ctrl_held = modifiers.state().control_key();
                self.shift_held = modifiers.state().shift_key();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                    if let Some(r) = &mut self.renderer {
                        r.resize(gpu);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = [position.x as f32, position.y as f32];
                self.mouse_delta = [
                    new_pos[0] - self.mouse_pos[0],
                    new_pos[1] - self.mouse_pos[1],
                ];
                self.mouse_pos = new_pos;
                if self.editor_state.gizmo.active_drag.is_some() {
                    self.update_gizmo_drag();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    winit::event::MouseButton::Left => {
                        if pressed {
                            if !self.try_begin_gizmo_drag() {
                                self.pick_entity_at_cursor();
                            }
                        } else {
                            self.end_gizmo_drag();
                        }
                    }
                    winit::event::MouseButton::Right => self.right_mouse_down = pressed,
                    winit::event::MouseButton::Middle => self.middle_mouse_down = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                self.cam_distance = (self.cam_distance - scroll * 0.5).clamp(1.0, 50.0);
            }
            _ => {}
        }
    }
}

impl EditorApp {
    fn pick_entity_at_cursor(&mut self) {
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };
        let pool = self.shared.lock_read();
        let world = pool.world();
        let camera = match world.resource::<Camera>() {
            Some(c) => c.clone(),
            None => return,
        };
        let screen_w = gpu.surface_config.width as f32;
        let screen_h = gpu.surface_config.height as f32;
        let (ray_origin, ray_dir) =
            camera.screen_to_ray(self.mouse_pos[0], self.mouse_pos[1], screen_w, screen_h);
        let ray = Ray::new(ray_origin, ray_dir);
        let mut closest: Option<(euca_ecs::Entity, f32)> = None;
        let candidates: Vec<(euca_ecs::Entity, Vec3, Collider)> = {
            let query = Query::<(euca_ecs::Entity, &GlobalTransform, &Collider)>::new(world);
            query
                .iter()
                .map(|(e, gt, col)| (e, gt.0.translation, col.clone()))
                .collect()
        };
        for (entity, pos, collider) in &candidates {
            if let Some(hit) = raycast_collider(&ray, *pos, collider) {
                if hit.t >= 0.0 && (closest.is_none() || hit.t < closest.unwrap().1) {
                    closest = Some((*entity, hit.t));
                }
            }
        }
        self.editor_state.selected_entity = closest.map(|(e, _)| e.index());
    }

    fn try_begin_gizmo_drag(&mut self) -> bool {
        let sel_idx = match self.editor_state.selected_entity {
            Some(i) => i,
            None => return false,
        };
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return false,
        };
        let pool = self.shared.lock_read();
        let world = pool.world();
        let entity = match find_alive_entity(world, sel_idx) {
            Some(e) => e,
            None => return false,
        };
        let entity_pos = match world.get::<GlobalTransform>(entity) {
            Some(gt) => gt.0.translation,
            None => return false,
        };
        let camera = match world.resource::<Camera>() {
            Some(c) => c.clone(),
            None => return false,
        };
        let screen_w = gpu.surface_config.width as f32;
        let screen_h = gpu.surface_config.height as f32;
        let (ray_origin, ray_dir) =
            camera.screen_to_ray(self.mouse_pos[0], self.mouse_pos[1], screen_w, screen_h);
        let ray = Ray::new(ray_origin, ray_dir);

        if let Some((axis, _t)) = euca_editor::gizmo::pick_gizmo_axis(&ray, entity_pos, camera.eye)
        {
            let axis_dir = axis.direction();
            let grab_t =
                Vec3::closest_line_param(entity_pos, axis_dir, ray_origin, ray_dir.normalize());
            let grab_point = entity_pos + axis_dir * grab_t;
            let current_transform = world
                .get::<LocalTransform>(entity)
                .map(|lt| lt.0)
                .unwrap_or_default();
            drop(pool);
            self.editor_state.gizmo.active_drag = Some(euca_editor::gizmo::GizmoDrag {
                axis,
                entity_index: sel_idx,
                start_position: entity_pos,
                grab_point,
            });
            self.editor_state
                .undo
                .begin_drag(sel_idx, current_transform);
            return true;
        }
        false
    }

    fn end_gizmo_drag(&mut self) {
        if let Some(drag) = self.editor_state.gizmo.active_drag.take() {
            let pool = self.shared.lock_read();
            let world = pool.world();
            if let Some(entity) = find_alive_entity(world, drag.entity_index) {
                let current = world
                    .get::<LocalTransform>(entity)
                    .map(|lt| lt.0)
                    .unwrap_or_default();
                drop(pool);
                self.editor_state.undo.end_drag(current);
            } else {
                drop(pool);
                self.editor_state.undo.cancel_drag();
            }
        }
    }

    fn update_gizmo_drag(&mut self) {
        let drag = match &self.editor_state.gizmo.active_drag {
            Some(d) => d.clone(),
            None => return,
        };
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };
        let (camera, screen_w, screen_h) = {
            let pool = self.shared.lock_read();
            let world = pool.world();
            let camera = match world.resource::<Camera>() {
                Some(c) => c.clone(),
                None => return,
            };
            let screen_w = gpu.surface_config.width as f32;
            let screen_h = gpu.surface_config.height as f32;
            (camera, screen_w, screen_h)
        };
        let (ray_origin, ray_dir) =
            camera.screen_to_ray(self.mouse_pos[0], self.mouse_pos[1], screen_w, screen_h);
        let new_pos = euca_editor::gizmo::update_gizmo_drag(&drag, ray_origin, ray_dir.normalize());
        let mut pool = self.shared.lock();
        let world = pool.world();
        if let Some(entity) = find_alive_entity(world, drag.entity_index) {
            if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
                lt.0.translation = new_pos;
            }
        }
    }
}

/// Render the scene to an offscreen texture, read it back, encode PNG, and send via oneshot.
fn capture_screenshot(
    gpu: &GpuContext,
    renderer: &mut Renderer,
    camera: &Camera,
    light: &DirectionalLight,
    ambient: &AmbientLight,
    draw_commands: &[DrawCommand],
    tx: tokio::sync::oneshot::Sender<Vec<u8>>,
) {
    let width = gpu.surface_config.width;
    let height = gpu.surface_config.height;
    let format = gpu.surface_config.format;
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;

    let offscreen = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let offscreen_view = offscreen.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("screenshot render"),
        });
    renderer.render_to_view(
        gpu,
        camera,
        light,
        ambient,
        draw_commands,
        &offscreen_view,
        &mut encoder,
    );

    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot buffer"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &offscreen,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(std::iter::once(encoder.finish()));
    let device = gpu.device.clone();

    std::thread::spawn(move || {
        let slice = buffer.slice(..);
        let (map_tx, map_rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = map_tx.send(result);
        });
        loop {
            match device.poll(wgpu::PollType::Poll) {
                Ok(status) if status.is_queue_empty() => break,
                Err(_) => break,
                _ => std::thread::yield_now(),
            }
        }
        if map_rx.recv().ok().and_then(|r| r.ok()).is_none() {
            log::error!("Screenshot buffer map failed");
            return;
        }

        let data = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        let is_bgra = matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        for y in 0..height {
            let row_start = (y * padded_bytes_per_row) as usize;
            let row_end = row_start + (width * bytes_per_pixel) as usize;
            let row = &data[row_start..row_end];
            if is_bgra {
                for pixel in row.chunks_exact(4) {
                    rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
                }
            } else {
                rgba.extend_from_slice(row);
            }
        }
        drop(data);
        buffer.unmap();

        let mut png_buf = Vec::new();
        {
            let mut enc = png::Encoder::new(std::io::Cursor::new(&mut png_buf), width, height);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            match enc.write_header() {
                Ok(mut writer) => {
                    if let Err(e) = writer.write_image_data(&rgba) {
                        log::error!("PNG write failed: {e}");
                        return;
                    }
                }
                Err(e) => {
                    log::error!("PNG header failed: {e}");
                    return;
                }
            }
        }
        let _ = tx.send(png_buf);
    });
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = EditorApp::new();
    event_loop.run_app(&mut app).unwrap();
}
