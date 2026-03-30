use euca_agent::{
    AgentServer, CameraOverride, EngineControl, ScreenshotChannel,
    auth::AuthStore,
    hud::{HudCanvas, HudElement, parse_color},
};
use euca_core::{Profiler, Time, profiler_begin, profiler_end};
use euca_ecs::Events;
use euca_ecs::{Query, Schedule, SharedWorld, World};
use euca_editor::{
    EditorState, SceneEntity, SceneFile, SpawnRequest, ToolbarAction, content_browser_panel,
    find_alive_entity, hierarchy_panel, inspector_panel, toolbar_panel,
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
const AUTOSAVE_FILE: &str = ".euca_autosave.json";

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
    world.insert_resource(euca_input::InputState::new());
    world.insert_resource(setup_moba_action_map());
    world.insert_resource(euca_input::InputContextStack::new());
    world.insert_resource(euca_scene::SpatialIndex::new(2.0));
    world.insert_resource(euca_scene::PrefabRegistry::default());
    world.insert_resource(LodSettings::default());
    world.insert_resource(PostProcessSettings::default());
    world.insert_resource(Profiler::default());
    world.insert_resource(euca_gameplay::camera::MobaCamera::default());
    world.insert_resource(euca_gameplay::player_input::ViewportSize {
        width: 1280.0,
        height: 720.0,
    });
    // AudioEngine init may fail on headless systems — log and continue
    match euca_audio::AudioEngine::new() {
        Ok(engine) => world.insert_resource(engine),
        Err(e) => log::warn!("Audio init failed (non-fatal): {e}"),
    }
}

/// Convert a winit `Key` to the string name used by `InputKey::Key`.
///
/// Character keys are uppercased (e.g. "q" -> "Q").  Named keys use their
/// standard name (e.g. "Space", "Escape", "Tab").  Returns `None` for keys
/// we don't map.
fn winit_key_to_string(key: &Key) -> Option<String> {
    match key {
        Key::Character(ch) => Some(ch.to_uppercase()),
        Key::Named(named) => {
            let s = match named {
                NamedKey::Space => "Space",
                NamedKey::Escape => "Escape",
                NamedKey::Enter => "Enter",
                NamedKey::Tab => "Tab",
                NamedKey::Delete => "Delete",
                NamedKey::Backspace => "Backspace",
                NamedKey::Shift => "Shift",
                NamedKey::Control => "Control",
                NamedKey::Alt => "Alt",
                NamedKey::ArrowUp => "ArrowUp",
                NamedKey::ArrowDown => "ArrowDown",
                NamedKey::ArrowLeft => "ArrowLeft",
                NamedKey::ArrowRight => "ArrowRight",
                NamedKey::F1 => "F1",
                NamedKey::F2 => "F2",
                NamedKey::F3 => "F3",
                NamedKey::F4 => "F4",
                NamedKey::F5 => "F5",
                _ => return None,
            };
            Some(s.to_string())
        }
        _ => None,
    }
}

/// Create the default MOBA action map with standard keybindings.
fn setup_moba_action_map() -> euca_input::ActionMap {
    use euca_input::InputKey;
    let mut map = euca_input::ActionMap::new();
    map.bind(InputKey::MouseRight, "move_or_attack");
    map.bind(InputKey::Key("Q".into()), "ability_q");
    map.bind(InputKey::Key("W".into()), "ability_w");
    map.bind(InputKey::Key("E".into()), "ability_e");
    map.bind(InputKey::Key("R".into()), "ability_r");
    map.bind(InputKey::Key("S".into()), "stop");
    map.bind(InputKey::Key("A".into()), "attack_move");
    map.bind(InputKey::Key("Space".into()), "center_camera");
    map
}

/// Helper: read delta time from the `Time` resource in the world.
fn world_dt(world: &World) -> f32 {
    world
        .resource::<Time>()
        .map(|t| t.delta as f32)
        .unwrap_or(0.016)
}

/// Build the parallel gameplay schedule.
///
/// Systems are grouped into stages with `after()` dependencies. Within each
/// stage, the scheduler automatically batches non-conflicting systems for
/// parallel execution via `std::thread::scope`.
fn build_gameplay_schedule() -> euca_ecs::ParallelSchedule {
    use euca_ecs::{ParallelSchedule, ParallelSystemAccess};

    let mut sched = ParallelSchedule::new();

    // ── Stage 1: Physics ─────────────────────────────────────────────────
    sched.add_system(
        "physics_step",
        |w: &mut World| physics_step_system(w),
        ParallelSystemAccess::new()
            .write::<euca_physics::Velocity>()
            .write::<LocalTransform>(),
    );
    sched
        .add_system(
            "character_controller",
            |w: &mut World| euca_physics::character_controller_system(w, world_dt(w)),
            ParallelSystemAccess::new()
                .write::<LocalTransform>()
                .write::<euca_physics::Velocity>(),
        )
        .after("physics_step");
    sched
        .add_system(
            "vehicle_physics",
            |w: &mut World| euca_physics::vehicle_physics_system(w, world_dt(w)),
            ParallelSystemAccess::new()
                .write::<LocalTransform>()
                .write::<euca_physics::Velocity>(),
        )
        .after("physics_step");

    // ── Stage 2: Gameplay (after physics) ────────────────────────────────
    sched
        .add_system(
            "apply_damage",
            euca_gameplay::apply_damage_system,
            ParallelSystemAccess::new().write::<euca_gameplay::Health>(),
        )
        .after("character_controller");
    sched
        .add_system(
            "death_check",
            euca_gameplay::death_check_system,
            ParallelSystemAccess::new().write::<euca_gameplay::Health>(),
        )
        .after("apply_damage");
    sched
        .add_system(
            "projectiles",
            |w: &mut World| euca_gameplay::projectile_system(w, world_dt(w)),
            ParallelSystemAccess::new()
                .write::<LocalTransform>()
                .write::<euca_gameplay::Health>(),
        )
        .after("character_controller");
    sched
        .add_system(
            "triggers",
            euca_gameplay::trigger_system,
            ParallelSystemAccess::new()
                .read::<LocalTransform>()
                .write::<euca_gameplay::Health>(),
        )
        .after("character_controller");
    sched
        .add_system(
            "ai",
            |w: &mut World| euca_gameplay::ai_system(w, world_dt(w)),
            ParallelSystemAccess::new()
                .read::<LocalTransform>()
                .write::<euca_physics::Velocity>(),
        )
        .after("character_controller");

    // ── Stage 3: Player control (after gameplay) ─────────────────────────
    sched
        .add_system(
            "player_input",
            euca_gameplay::player_input::player_input_system,
            ParallelSystemAccess::new()
                .write::<euca_gameplay::PlayerCommandQueue>()
                .resource_read::<euca_input::InputState>(),
        )
        .after("death_check");
    sched
        .add_system(
            "player_commands",
            |w: &mut World| euca_gameplay::player::player_command_system(w, world_dt(w)),
            ParallelSystemAccess::new()
                .write::<euca_gameplay::PlayerCommandQueue>()
                .write::<euca_physics::Velocity>()
                .write::<LocalTransform>(),
        )
        .after("player_input");

    // ── Stage 4: Combat (after player) ───────────────────────────────────
    sched
        .add_system(
            "auto_combat",
            |w: &mut World| euca_gameplay::auto_combat_system(w, world_dt(w)),
            ParallelSystemAccess::new()
                .read::<euca_gameplay::Health>()
                .write::<euca_physics::Velocity>(),
        )
        .after("player_commands");
    sched
        .add_system(
            "game_state",
            |w: &mut World| euca_gameplay::game_state_system(w, world_dt(w)),
            ParallelSystemAccess::new().resource_write::<euca_gameplay::GameState>(),
        )
        .after("player_commands");

    // ── Stage 5: Rules (after combat) ────────────────────────────────────
    // Rule systems are mostly independent — scheduler can parallelize them.
    sched
        .add_system(
            "on_death_rules",
            euca_gameplay::on_death_rule_system,
            ParallelSystemAccess::new().read::<euca_gameplay::Health>(),
        )
        .after("auto_combat");
    sched
        .add_system(
            "timer_rules",
            |w: &mut World| euca_gameplay::timer_rule_system(w, world_dt(w)),
            ParallelSystemAccess::new(),
        )
        .after("auto_combat");
    sched
        .add_system(
            "health_below_rules",
            euca_gameplay::health_below_rule_system,
            ParallelSystemAccess::new().read::<euca_gameplay::Health>(),
        )
        .after("auto_combat");
    sched
        .add_system(
            "on_score_rules",
            euca_gameplay::on_score_rule_system,
            ParallelSystemAccess::new().resource_read::<euca_gameplay::GameState>(),
        )
        .after("auto_combat");
    sched
        .add_system(
            "on_phase_rules",
            euca_gameplay::on_phase_rule_system,
            ParallelSystemAccess::new().resource_read::<euca_gameplay::GameState>(),
        )
        .after("auto_combat");

    // ── Stage 6: Respawn & cleanup (after rules) ─────────────────────────
    sched
        .add_system(
            "respawn",
            |w: &mut World| {
                let delay = w
                    .resource::<euca_gameplay::GameState>()
                    .map(|s| s.config.respawn_delay);
                let dt = world_dt(w);
                if delay.is_some() {
                    euca_gameplay::respawn_system(w, dt);
                }
            },
            ParallelSystemAccess::new().write::<LocalTransform>(),
        )
        .after("on_death_rules");
    sched
        .add_system(
            "start_respawn",
            |w: &mut World| {
                let delay = w
                    .resource::<euca_gameplay::GameState>()
                    .map(|s| s.config.respawn_delay);
                if let Some(d) = delay {
                    euca_gameplay::start_respawn_on_death(w, d);
                }
            },
            ParallelSystemAccess::new().write::<euca_gameplay::RespawnTimer>(),
        )
        .after("on_death_rules");
    sched
        .add_system(
            "corpse_cleanup",
            |w: &mut World| euca_gameplay::corpse_cleanup_system(w, world_dt(w)),
            ParallelSystemAccess::new(),
        )
        .after("on_death_rules");

    // ── Stage 7: Economy & abilities ─────────────────────────────────────
    sched
        .add_system(
            "gold_on_kill",
            euca_gameplay::gold_on_kill_system,
            ParallelSystemAccess::new().write::<euca_gameplay::Gold>(),
        )
        .after("respawn");
    sched
        .add_system(
            "xp_on_kill",
            euca_gameplay::xp_on_kill_system,
            ParallelSystemAccess::new().write::<euca_gameplay::Level>(),
        )
        .after("respawn");
    sched
        .add_system(
            "ability_tick",
            |w: &mut World| euca_gameplay::ability_tick_system(w, world_dt(w)),
            ParallelSystemAccess::new(),
        )
        .after("respawn");
    sched
        .add_system(
            "use_ability",
            euca_gameplay::use_ability_system,
            ParallelSystemAccess::new(),
        )
        .after("ability_tick");

    // ── Stage 8: Audio, animation, particles, nav ────────────────────────
    // These are independent subsystems — can run in parallel.
    sched
        .add_system(
            "audio",
            |w: &mut World| euca_audio::audio_update_system_mut(w, world_dt(w)),
            ParallelSystemAccess::new(),
        )
        .after("use_ability");
    sched
        .add_system(
            "skeletal_animation",
            |w: &mut World| euca_asset::skeletal_animation_system(w, world_dt(w)),
            ParallelSystemAccess::new(),
        )
        .after("use_ability");
    sched
        .add_system(
            "particle_emit",
            |w: &mut World| euca_particle::emit_particles_system(w, world_dt(w)),
            ParallelSystemAccess::new(),
        )
        .after("use_ability");
    sched
        .add_system(
            "particle_update",
            |w: &mut World| euca_particle::particle_update_system(w, world_dt(w)),
            ParallelSystemAccess::new(),
        )
        .after("particle_emit");
    sched
        .add_system(
            "pathfinding",
            euca_nav::pathfinding_system,
            ParallelSystemAccess::new(),
        )
        .after("use_ability");
    sched
        .add_system(
            "steering",
            |w: &mut World| euca_nav::steering_system(w, world_dt(w)),
            ParallelSystemAccess::new().write::<euca_physics::Velocity>(),
        )
        .after("pathfinding");

    // ── Stage 9: Network prediction correction ──────────────────────────
    sched
        .add_system(
            "prediction_correction",
            euca_net::apply_prediction_system,
            ParallelSystemAccess::new().write::<LocalTransform>(),
        )
        .after("steering");

    // ── Finalize: event flush + tick advance ─────────────────────────────
    sched
        .add_system(
            "event_flush",
            |w: &mut World| {
                if let Some(events) = w.resource_mut::<Events>() {
                    events.update();
                }
                w.tick();
            },
            ParallelSystemAccess::new(), // exclusive — no other system in this batch
        )
        .after("audio")
        .after("skeletal_animation")
        .after("particle_update")
        .after("steering")
        .after("prediction_correction");

    sched.build();

    let batches = sched.batches();
    log::info!(
        "Parallel schedule: {} systems in {} batches",
        sched.len(),
        batches.len(),
    );
    for (i, batch) in batches.iter().enumerate() {
        log::info!("  Batch {i}: {} systems (parallel)", batch.len());
    }

    sched
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
        .map(|(e, gt, mr, mat)| {
            let mut model_matrix = gt.0.to_matrix();
            if let Some(offset) = world.get::<GroundOffset>(e) {
                model_matrix.cols[3][1] += offset.0;
            }
            DrawCommand {
                mesh: mr.mesh,
                material: mat.handle,
                model_matrix,
                aabb: None,
            }
        })
        .collect()
}

/// Append selection outlines (slightly scaled, orange material) for all selected entities.
/// Skips the outline for the ground plane mesh to avoid z-fighting on flat geometry.
fn append_selection_outline(
    world: &World,
    selected: &[u32],
    outline_mat: Option<MaterialHandle>,
    plane_mesh: Option<MeshHandle>,
    cmds: &mut Vec<DrawCommand>,
) {
    let Some(mat) = outline_mat else {
        return;
    };
    for sel_idx in selected {
        for g in 0..16u32 {
            let entity = euca_ecs::Entity::from_raw(*sel_idx, g);
            if !world.is_alive(entity) {
                continue;
            }
            if let (Some(gt), Some(mr)) = (
                world.get::<GlobalTransform>(entity),
                world.get::<MeshRenderer>(entity),
            ) {
                // Skip outline for ground plane — flat geometry causes z-fighting.
                if let Some(pm) = plane_mesh {
                    if mr.mesh == pm {
                        break;
                    }
                }
                let max_scale = gt.0.scale.x.max(gt.0.scale.y).max(gt.0.scale.z);
                if max_scale < 5.0 {
                    let mut t = gt.0;
                    t.scale = t.scale * 1.03;
                    t.translation.y += 0.002;
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
}

/// Append gizmo axis handle draw commands for the selected entity.
fn append_gizmo_commands(world: &World, editor_state: &EditorState, cmds: &mut Vec<DrawCommand>) {
    let Some(sel_idx) = editor_state.primary_selected() else {
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
    cylinder_mesh: Option<MeshHandle>,
    cone_mesh: Option<MeshHandle>,
    default_material: Option<MaterialHandle>,
    ctrl_held: bool,
    shift_held: bool,
    _tokio_rt: Option<tokio::runtime::Runtime>,
    /// Discovered level files (scanned from current dir + levels/ subdirectory).
    available_levels: Vec<String>,
    /// Index into `available_levels` for the currently selected level.
    selected_level: Option<usize>,
    /// Previously loaded level index (to detect selection changes).
    loaded_level: Option<usize>,
    /// Tracks previous frame's play state to detect play-start transitions.
    was_playing_last_frame: bool,
    /// Ground plane mesh handle — outlines are skipped for this mesh.
    plane_mesh: Option<MeshHandle>,
    /// Parallel gameplay system schedule (built once, run each tick).
    gameplay_schedule: euca_ecs::ParallelSchedule,
    /// File watcher for hot-reloading levels and assets on external changes.
    file_watcher: euca_asset::FileWatcher,
    /// Frame counter for throttling file watcher polls.
    poll_counter: u64,
}

impl EditorApp {
    fn new() -> Self {
        let (survey, wgpu_instance) = HardwareSurvey::detect();
        let mut world = World::new();
        setup_default_resources(&mut world);
        let shared = SharedWorld::new(world, Schedule::new());

        let mut app = Self {
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
            cylinder_mesh: None,
            cone_mesh: None,
            default_material: None,
            ctrl_held: false,
            shift_held: false,
            _tokio_rt: None,
            available_levels: Vec::new(),
            selected_level: None,
            loaded_level: None,
            was_playing_last_frame: false,
            plane_mesh: None,
            gameplay_schedule: build_gameplay_schedule(),
            file_watcher: euca_asset::FileWatcher::new(),
            poll_counter: 0,
        };
        app.available_levels = Self::scan_level_files();
        if !app.available_levels.is_empty() {
            app.selected_level = Some(0);
        }

        // Watch level directories for hot-reload
        app.file_watcher.watch(".");
        if std::path::Path::new("levels").is_dir() {
            app.file_watcher.watch("levels");
        }
        if std::path::Path::new("assets").is_dir() {
            app.file_watcher.watch("assets");
        }
        // Seed initial modification times
        app.file_watcher.poll();

        app
    }

    /// Scan for level files in the current directory and `levels/` subdirectory.
    /// Returns all discovered `.level.json` and `level.json` files.
    fn scan_level_files() -> Vec<String> {
        let mut files = Vec::new();

        // Check current directory
        if std::path::Path::new("level.json").exists() {
            files.push("level.json".to_string());
        }
        if let Ok(entries) = std::fs::read_dir(".") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy().to_string();
                if name.ends_with(".level.json") {
                    files.push(name);
                }
            }
        }

        // Check levels/ subdirectory
        if let Ok(entries) = std::fs::read_dir("levels") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy().to_string();
                if name.ends_with(".json") {
                    files.push(format!("levels/{name}"));
                }
            }
        }

        files.sort();
        files.dedup();
        if !files.is_empty() {
            log::info!("Discovered {} level files: {:?}", files.len(), files);
        }
        files
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        let cube = renderer.upload_mesh(gpu, &Mesh::cube());
        let sphere = renderer.upload_mesh(gpu, &Mesh::sphere(0.5, 16, 32));
        let plane = renderer.upload_mesh(gpu, &Mesh::plane(20.0));
        let cylinder = renderer.upload_mesh(gpu, &Mesh::cylinder(0.5, 1.0, 24));
        let cone = renderer.upload_mesh(gpu, &Mesh::cone(0.5, 1.0, 24));
        self.cube_mesh = Some(cube);
        self.sphere_mesh = Some(sphere);
        self.plane_mesh = Some(plane);
        self.cylinder_mesh = Some(cylinder);
        self.cone_mesh = Some(cone);
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
        meshes.insert("cylinder".to_string(), cylinder);
        meshes.insert("cone".to_string(), cone);

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
            ..Default::default()
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
        // Reload the selected level so entities reset to saved positions
        self.load_selected_level();
        self.editor_state.clear_selection();
    }

    /// Load the currently selected level file into the world.
    /// Called when the level selection changes or on Stop to restore saved state.
    fn load_selected_level(&mut self) {
        let path = match self.selected_level {
            Some(idx) => match self.available_levels.get(idx) {
                Some(p) => p.clone(),
                None => return,
            },
            None => return,
        };

        let mut pool = self.shared.lock();
        let world = pool.world();
        match std::fs::read_to_string(&path) {
            Ok(data) => match serde_json::from_str::<serde_json::Value>(&data) {
                Ok(level) => {
                    let count = euca_agent::load_level_into_world(world, &level);
                    log::info!("Level loaded into editor: {count} entities from {path}");
                }
                Err(e) => log::error!("Invalid level JSON in {path}: {e}"),
            },
            Err(e) => log::error!("Cannot read level file {path}: {e}"),
        }
        self.loaded_level = self.selected_level;
    }

    fn render_frame(&mut self) {
        if self.editor_state.reset_requested {
            self.editor_state.reset_requested = false;
            self.reset_scene();
        }

        // Hot-reload: poll for external file changes every ~60 frames (~1 second at 60fps).
        self.poll_counter += 1;
        if self.poll_counter % 60 == 0 && !self.editor_state.playing {
            let changed_files = self.file_watcher.poll().to_vec();
            for path in &changed_files {
                // Check if the changed file is the currently loaded level
                if let Some(level_idx) = self.selected_level {
                    if let Some(level_path) = self.available_levels.get(level_idx) {
                        let level_canonical = std::path::Path::new(level_path).canonicalize().ok();
                        let changed_canonical = path.canonicalize().ok();
                        if level_canonical.is_some() && level_canonical == changed_canonical {
                            log::info!("Level file changed externally, reloading...");
                            self.load_selected_level();
                            break;
                        }
                    }
                }

                // Log other asset changes (future: trigger asset hot-reload)
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if matches!(ext, "glb" | "gltf" | "png" | "jpg" | "wav" | "ogg") {
                    log::info!("Asset changed: {}", path.display());
                }
            }
        }

        // Detect level selection change — reload entities in editor viewport.
        if !self.editor_state.playing && self.selected_level != self.loaded_level {
            // Clear existing non-persistent entities and reload
            {
                let mut pool = self.shared.lock();
                let world = pool.world();
                let entities: Vec<euca_ecs::Entity> = {
                    let query = euca_ecs::Query::<euca_ecs::Entity>::new(world);
                    query.iter().collect()
                };
                for entity in entities {
                    // Keep persistent entities (ground, light)
                    if world.get::<euca_agent::Persistent>(entity).is_none() {
                        world.despawn(entity);
                    }
                }
            }
            if self.selected_level.is_some() {
                self.load_selected_level();
            } else {
                self.loaded_level = None;
            }
            self.editor_state.clear_selection();
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

        // Detect play-start transition: load level file and auto-follow hero.
        if self.editor_state.playing && !self.was_playing_last_frame {
            // Level is already loaded in the editor viewport — no need to load again.
            // Just set up play-time resources (camera follow, navmesh).

            // Auto-detect PlayerHero and set camera follow
            let hero = {
                let q = Query::<(euca_ecs::Entity, &euca_gameplay::player::PlayerHero)>::new(world);
                q.iter().map(|(e, _)| e).next()
            };
            if let Some(hero) = hero {
                if let Some(cam) = world.resource_mut::<euca_gameplay::camera::MobaCamera>() {
                    if cam.follow_entity.is_none() {
                        cam.follow_entity = Some(hero);
                    }
                }
            }
            // Auto-initialize navmesh from world geometry if none exists
            if world.resource::<euca_nav::NavMesh>().is_none() {
                let config = euca_nav::GridConfig {
                    min: [-12.0, -12.0],
                    max: [12.0, 12.0],
                    cell_size: 0.5,
                    ground_y: 0.0,
                };
                let mesh = euca_nav::build_navmesh_from_world_with_radius(world, config, 0.5);
                world.insert_resource(mesh);
            }
            // Clear editor selection for clean play mode
            self.editor_state.clear_selection();
        }
        self.was_playing_last_frame = self.editor_state.playing;

        if self.editor_state.should_tick() {
            // Attach visuals to rule-spawned entities (must run before schedule
            // clears events, and needs access to DefaultAssets which isn't in ECS).
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

            self.gameplay_schedule.run(world);
            // Clear per-frame input AFTER gameplay systems have consumed it.
            if let Some(input) = world.resource_mut::<euca_input::InputState>() {
                input.begin_frame();
            }
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

        // MOBA camera: follow player hero (overrides editor camera when playing)
        if self.editor_state.playing {
            euca_gameplay::camera::moba_camera_system(world);
        }

        lod_select_system(world);

        // Upload GLB meshes that were loaded by the spawn handler.
        {
            let gpu = self.gpu.as_ref().unwrap();
            let renderer = self.renderer.as_mut().unwrap();
            euca_agent::routes::drain_pending_mesh_uploads(world, renderer, gpu);
        }

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
        if !self.editor_state.playing {
            append_selection_outline(
                world,
                &self.editor_state.selected_entities,
                self.outline_material,
                self.plane_mesh,
                &mut draw_commands,
            );
            append_gizmo_commands(world, &self.editor_state, &mut draw_commands);
        }
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

        // Set up light probe for indirect lighting (uniform probe from ambient light).
        {
            let probe = euca_render::LightProbe::uniform(
                Vec3::ZERO,
                [
                    ambient.color[0] * ambient.intensity,
                    ambient.color[1] * ambient.intensity,
                    ambient.color[2] * ambient.intensity,
                ],
            );
            let mut sh_gpu = [[0.0f32; 4]; 9];
            for (i, coeffs) in probe.sh.iter().enumerate() {
                sh_gpu[i] = [coeffs[0], coeffs[1], coeffs[2], 0.0];
            }
            renderer.set_probe_sh(sh_gpu);
        }

        // Collect and render CPU particle emitters as billboard meshes.
        {
            let batches = euca_particle::render::collect_particle_render_data(world, camera.eye);
            let axes = euca_particle::render::BillboardAxes::from_camera(
                camera.eye,
                camera.target,
                Vec3::new(0.0, 1.0, 0.0),
            );
            for batch in &batches {
                if batch.is_empty() {
                    continue;
                }
                let (billboard_verts, indices) = batch.build_billboard_geometry(&axes);
                let vertices: Vec<Vertex> = billboard_verts
                    .iter()
                    .map(|bv| Vertex {
                        position: bv.position,
                        normal: [0.0, 0.0, 1.0],
                        tangent: [1.0, 0.0, 0.0],
                        uv: bv.uv,
                    })
                    .collect();
                let mesh = Mesh { vertices, indices };
                let mesh_handle = renderer.upload_mesh(gpu, &mesh);
                let mat =
                    Material::new([1.0, 1.0, 1.0, 1.0], 0.0, 1.0).with_emissive([1.0, 1.0, 1.0]);
                let mat_handle = renderer.upload_material(gpu, &mat);
                draw_commands.push(DrawCommand {
                    mesh: mesh_handle,
                    material: mat_handle,
                    model_matrix: euca_math::Mat4::IDENTITY,
                    aabb: None,
                });
            }
        }

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
        let playing = self.editor_state.playing;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            let dt = world.resource::<Time>().map(|t| t.delta).unwrap_or(0.0);
            toolbar_action = toolbar_panel(
                ctx,
                &mut self.editor_state,
                world,
                dt,
                &self.available_levels,
                &mut self.selected_level,
            );
            if !playing {
                let shift = self.shift_held;
                spawn_request = hierarchy_panel(ctx, &mut self.editor_state, world, shift);
                let browser_spawn = content_browser_panel(ctx, &self.editor_state);
                if spawn_request.is_none() {
                    spawn_request = browser_spawn;
                }
                let inspector_changed = inspector_panel(ctx, &mut self.editor_state, world);
                if inspector_changed {
                    let elapsed = world.resource::<Time>().map(|t| t.elapsed).unwrap_or(0.0);
                    self.editor_state.mark_dirty(elapsed);
                }
            }
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
                        self.editor_state.dirty = false;
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
                            ..Default::default()
                        });
                        self.editor_state.clear_selection();
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
                SpawnRequest::Plane => {
                    if let Some(mesh) = self.plane_mesh {
                        world.insert(e, MeshRenderer { mesh });
                    }
                    if let Some(mat) = self.default_material {
                        world.insert(e, MaterialRef { handle: mat });
                    }
                    world.insert(e, Collider::aabb(10.0, 0.01, 10.0));
                }
                SpawnRequest::Cylinder => {
                    if let Some(mesh) = self.cylinder_mesh {
                        world.insert(e, MeshRenderer { mesh });
                    }
                    if let Some(mat) = self.default_material {
                        world.insert(e, MaterialRef { handle: mat });
                    }
                    world.insert(e, Collider::aabb(0.5, 0.5, 0.5));
                }
                SpawnRequest::Cone => {
                    if let Some(mesh) = self.cone_mesh {
                        world.insert(e, MeshRenderer { mesh });
                    }
                    if let Some(mat) = self.default_material {
                        world.insert(e, MaterialRef { handle: mat });
                    }
                    world.insert(e, Collider::aabb(0.5, 0.5, 0.5));
                }
                SpawnRequest::Empty => {}
            }
            self.editor_state.select(e.index());
            self.editor_state
                .undo
                .push(euca_editor::undo::UndoAction::SpawnEntity {
                    entity_index: e.index(),
                });
            let elapsed = world.resource::<Time>().map(|t| t.elapsed).unwrap_or(0.0);
            self.editor_state.mark_dirty(elapsed);
        }

        // Auto-save: debounced, 5 seconds after last change (only when not playing)
        if self.editor_state.dirty && !self.editor_state.playing {
            let elapsed = world.resource::<Time>().map(|t| t.elapsed).unwrap_or(0.0);
            if elapsed - self.editor_state.last_dirty_time > 5.0 {
                let scene = SceneFile::capture(world);
                if let Err(e) = scene.save(AUTOSAVE_FILE) {
                    log::error!("Auto-save failed: {e}");
                } else {
                    log::info!("Auto-saved to {AUTOSAVE_FILE}");
                }
                self.editor_state.dirty = false;
            }
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
                let query = Query::<(
                    euca_ecs::Entity,
                    &GlobalTransform,
                    &MeshRenderer,
                    &MaterialRef,
                )>::new(w);
                query
                    .iter()
                    .map(|(e, gt, mr, mat)| {
                        let mut model_matrix = gt.0.to_matrix();
                        if let Some(offset) = w.get::<GroundOffset>(e) {
                            model_matrix.cols[3][1] += offset.0;
                        }
                        DrawCommand {
                            mesh: mr.mesh,
                            material: mat.handle,
                            model_matrix,
                            aabb: None,
                        }
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
            self.load_selected_level();

            // Check for auto-save recovery
            if std::path::Path::new(AUTOSAVE_FILE).exists() {
                log::warn!(
                    "Auto-save file found ({AUTOSAVE_FILE}). Use File > Load to recover unsaved work."
                );
            }

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

        // Forward input events to InputState when the simulation is playing.
        if self.editor_state.playing {
            self.forward_input_event(&event);
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
                let indices: Vec<u32> = self.editor_state.selected_entities.clone();
                if !indices.is_empty() {
                    let mut pool = self.shared.lock();
                    let world = pool.world();
                    let mut despawned = Vec::new();
                    let mut elapsed = 0.0;
                    for idx in &indices {
                        if let Some(e) = find_alive_entity(world, *idx) {
                            let transform = world
                                .get::<LocalTransform>(e)
                                .map(|lt| lt.0)
                                .unwrap_or_default();
                            let mesh = world.get::<MeshRenderer>(e).map(|mr| mr.mesh);
                            let material = world.get::<MaterialRef>(e).map(|mr| mr.handle);
                            let collider = world.get::<Collider>(e).cloned();
                            elapsed = world.resource::<Time>().map(|t| t.elapsed).unwrap_or(0.0);
                            world.despawn(e);
                            despawned.push(euca_editor::undo::EntitySnapshot {
                                entity_index: *idx,
                                transform,
                                mesh,
                                material,
                                collider,
                            });
                        }
                    }
                    if despawned.len() == 1 {
                        self.editor_state
                            .undo
                            .push(euca_editor::undo::UndoAction::DespawnEntity(
                                despawned.remove(0),
                            ));
                    } else if !despawned.is_empty() {
                        self.editor_state.undo.push(
                            euca_editor::undo::UndoAction::DespawnMultiple {
                                entities: despawned,
                            },
                        );
                    }
                    self.editor_state.mark_dirty(elapsed);
                    self.editor_state.clear_selection();
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
                if let Some(idx) = self.editor_state.primary_selected() {
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
            // Gizmo mode shortcuts (Unreal convention: W=Translate, E=Rotate, R=Scale)
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref ch),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if !self.editor_state.playing
                && (ch.as_str() == "w"
                    || ch.as_str() == "W"
                    || ch.as_str() == "e"
                    || ch.as_str() == "E"
                    || ch.as_str() == "r"
                    || ch.as_str() == "R") =>
            {
                // Only switch mode if no active drag and not in play mode
                if self.editor_state.gizmo.active_drag.is_none() {
                    let mode = match ch.as_str() {
                        "w" | "W" => euca_editor::gizmo::GizmoMode::Translate,
                        "e" | "E" => euca_editor::gizmo::GizmoMode::Rotate,
                        "r" | "R" => euca_editor::gizmo::GizmoMode::Scale,
                        _ => unreachable!(),
                    };
                    self.editor_state.gizmo.mode = mode;
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
            // Ctrl+C: copy selected entities to clipboard
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref ch),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if ch.as_str() == "c" && self.ctrl_held => {
                let indices: Vec<u32> = self.editor_state.selected_entities.clone();
                if !indices.is_empty() {
                    let pool = self.shared.lock_read();
                    let world = pool.world();
                    let mut clipboard = Vec::new();
                    for idx in &indices {
                        if let Some(entity) = find_alive_entity(world, *idx) {
                            let t = world
                                .get::<LocalTransform>(entity)
                                .map(|lt| lt.0)
                                .unwrap_or_default();
                            let mesh_str = world
                                .get::<MeshRenderer>(entity)
                                .map_or("none".to_string(), |m| format!("mesh_{}", m.mesh.0));
                            let material =
                                world.get::<MaterialRef>(entity).map_or(0, |m| m.handle.0);
                            clipboard.push(SceneEntity {
                                position: [t.translation.x, t.translation.y, t.translation.z],
                                scale: [t.scale.x, t.scale.y, t.scale.z],
                                rotation: [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w],
                                mesh: mesh_str,
                                material,
                                health: None,
                                team: None,
                                physics_body: None,
                                combat: false,
                            });
                        }
                    }
                    self.editor_state.clipboard = clipboard;
                    log::info!("Copied {} entities to clipboard", indices.len());
                }
            }
            // Ctrl+V: paste from clipboard, offset by +1.0 X
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref ch),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if ch.as_str() == "v" && self.ctrl_held => {
                let clipboard = self.editor_state.clipboard.clone();
                if !clipboard.is_empty() {
                    let mut pool = self.shared.lock();
                    let world = pool.world();
                    let mut new_indices = Vec::new();
                    for se in &clipboard {
                        let pos = Vec3::new(se.position[0] + 1.0, se.position[1], se.position[2]);
                        let scl = Vec3::new(se.scale[0], se.scale[1], se.scale[2]);
                        let rot = euca_math::Quat::from_xyzw(
                            se.rotation[0],
                            se.rotation[1],
                            se.rotation[2],
                            se.rotation[3],
                        );
                        let mut transform = Transform::from_translation(pos);
                        transform.scale = scl;
                        transform.rotation = rot;
                        let e = world.spawn(LocalTransform(transform));
                        world.insert(e, GlobalTransform::default());
                        // Resolve mesh handle from the stored mesh string
                        if se.mesh != "none" {
                            if let Some(assets) = world
                                .resource::<euca_agent::routes::DefaultAssets>()
                                .cloned()
                            {
                                // Try by name first, then by raw handle
                                let mesh_handle = assets.mesh(&se.mesh).or_else(|| {
                                    se.mesh
                                        .strip_prefix("mesh_")
                                        .and_then(|n| n.parse::<u32>().ok())
                                        .map(MeshHandle)
                                });
                                if let Some(mh) = mesh_handle {
                                    world.insert(e, MeshRenderer { mesh: mh });
                                }
                            }
                        }
                        if se.material > 0 {
                            world.insert(
                                e,
                                MaterialRef {
                                    handle: MaterialHandle(se.material),
                                },
                            );
                        } else if let Some(mat) = self.default_material {
                            world.insert(e, MaterialRef { handle: mat });
                        }
                        new_indices.push(e.index());
                    }
                    let elapsed = world.resource::<Time>().map(|t| t.elapsed).unwrap_or(0.0);
                    self.editor_state.mark_dirty(elapsed);
                    self.editor_state
                        .undo
                        .push(euca_editor::undo::UndoAction::SpawnMultiple {
                            entity_indices: new_indices.clone(),
                        });
                    // Select the newly pasted entities
                    self.editor_state.selected_entities = new_indices;
                    log::info!("Pasted {} entities from clipboard", clipboard.len());
                }
            }
            // G key: toggle snap-to-grid
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref ch),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if (ch.as_str() == "g" || ch.as_str() == "G") && !self.ctrl_held => {
                self.editor_state.snap_to_grid = !self.editor_state.snap_to_grid;
                log::info!(
                    "Snap to grid: {} (size: {})",
                    if self.editor_state.snap_to_grid {
                        "ON"
                    } else {
                        "OFF"
                    },
                    self.editor_state.grid_size,
                );
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
                // Keep ViewportSize in sync so player input ray calculations stay correct.
                let mut pool = self.shared.lock();
                let world = pool.world();
                if let Some(vp) = world.resource_mut::<euca_gameplay::player_input::ViewportSize>()
                {
                    vp.width = size.width as f32;
                    vp.height = size.height as f32;
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
                if !self.editor_state.playing {
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
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !self.editor_state.playing {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                    };
                    self.cam_distance = (self.cam_distance - scroll * 0.5).clamp(1.0, 50.0);
                }
            }
            _ => {}
        }
    }
}

impl EditorApp {
    /// Forward a winit event to the gameplay `InputState` resource.
    ///
    /// Called only when the simulation is playing so editor-only mode is not
    /// affected.
    fn forward_input_event(&mut self, event: &WindowEvent) {
        use euca_input::InputKey;

        let mut pool = self.shared.lock();
        let world = pool.world();
        let Some(input) = world.resource_mut::<euca_input::InputState>() else {
            return;
        };

        match event {
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let Some(key_name) = winit_key_to_string(&key_event.logical_key) {
                    match key_event.state {
                        ElementState::Pressed => input.press(InputKey::Key(key_name)),
                        ElementState::Released => input.release(InputKey::Key(key_name)),
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let input_key = match button {
                    winit::event::MouseButton::Left => Some(InputKey::MouseLeft),
                    winit::event::MouseButton::Right => Some(InputKey::MouseRight),
                    winit::event::MouseButton::Middle => Some(InputKey::MouseMiddle),
                    _ => None,
                };
                if let Some(key) = input_key {
                    match state {
                        ElementState::Pressed => input.press(key),
                        ElementState::Released => input.release(key),
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                input.set_mouse_position(position.x as f32, position.y as f32);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                input.set_scroll(scroll);
            }
            _ => {}
        }
    }

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
        if let Some((entity, _)) = closest {
            if self.shift_held {
                self.editor_state.add_select(entity.index());
            } else {
                self.editor_state.select(entity.index());
            }
        } else if !self.shift_held {
            self.editor_state.clear_selection();
        }
    }

    fn try_begin_gizmo_drag(&mut self) -> bool {
        let sel_idx = match self.editor_state.primary_selected() {
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

        let mode = self.editor_state.gizmo.mode;
        if let Some((axis, _t)) =
            euca_editor::gizmo::pick_gizmo_axis(&ray, entity_pos, camera.eye, mode)
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
                mode,
                axis,
                entity_index: sel_idx,
                start_position: entity_pos,
                grab_point,
                start_rotation: current_transform.rotation,
                start_scale: current_transform.scale,
                accumulated_angle: 0.0,
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
                let elapsed = world.resource::<Time>().map(|t| t.elapsed).unwrap_or(0.0);
                drop(pool);
                self.editor_state.undo.end_drag(current);
                self.editor_state.mark_dirty(elapsed);
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
        let ray_dir_n = ray_dir.normalize();
        let mut pool = self.shared.lock();
        let world = pool.world();
        if let Some(entity) = find_alive_entity(world, drag.entity_index) {
            // Read old position before applying gizmo, to compute delta for other selections.
            let old_translation = world
                .get::<LocalTransform>(entity)
                .map(|lt| lt.0.translation)
                .unwrap_or_default();
            if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
                match drag.mode {
                    euca_editor::gizmo::GizmoMode::Translate => {
                        let mut new_pos =
                            euca_editor::gizmo::update_translate_drag(&drag, ray_origin, ray_dir_n);
                        if self.editor_state.snap_to_grid {
                            let gs = self.editor_state.grid_size;
                            new_pos.x = (new_pos.x / gs).round() * gs;
                            new_pos.y = (new_pos.y / gs).round() * gs;
                            new_pos.z = (new_pos.z / gs).round() * gs;
                        }
                        lt.0.translation = new_pos;
                    }
                    euca_editor::gizmo::GizmoMode::Rotate => {
                        lt.0.rotation =
                            euca_editor::gizmo::update_rotate_drag(&drag, ray_origin, ray_dir_n);
                    }
                    euca_editor::gizmo::GizmoMode::Scale => {
                        lt.0.scale =
                            euca_editor::gizmo::update_scale_drag(&drag, ray_origin, ray_dir_n);
                    }
                }
            }

            // Apply the same translation delta to other selected entities
            if matches!(drag.mode, euca_editor::gizmo::GizmoMode::Translate) {
                let new_translation = world
                    .get::<LocalTransform>(entity)
                    .map(|lt| lt.0.translation)
                    .unwrap_or_default();
                let delta = new_translation - old_translation;
                let other_indices: Vec<u32> = self
                    .editor_state
                    .selected_entities
                    .iter()
                    .filter(|&&idx| idx != drag.entity_index)
                    .copied()
                    .collect();
                for idx in other_indices {
                    if let Some(other) = find_alive_entity(world, idx) {
                        if let Some(lt) = world.get_mut::<LocalTransform>(other) {
                            lt.0.translation = lt.0.translation + delta;
                        }
                    }
                }
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
