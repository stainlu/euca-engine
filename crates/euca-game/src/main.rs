//! Standalone game runner — runs a game project without the editor.
//!
//! Usage: `euca-game [path/to/.eucaproject.json]`
//!
//! If no path is given, searches the current directory for `.eucaproject.json`.

use euca_core::Time;
use euca_ecs::{Events, Query, World};
use euca_math::{Transform, Vec3};
use euca_physics::{PhysicsConfig, physics_step_system};
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};

use euca_game::project::{PROJECT_FILE_NAME, ProjectConfig};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{WindowAttributes, WindowId};

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .init();

    // Find project file
    let project_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| PROJECT_FILE_NAME.to_string());

    let project = ProjectConfig::load(&project_path).unwrap_or_else(|e| {
        log::error!("Failed to load project: {e}");
        log::error!("Usage: euca-game [path/to/.eucaproject.json]");
        std::process::exit(1);
    });

    log::info!("Starting game: {} v{}", project.name, project.version);

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = GameApp::new(project);
    event_loop.run_app(&mut app).expect("Event loop failed");
}

struct GameApp {
    project: ProjectConfig,
    world: World,
    initialized: bool,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    window_attrs: WindowAttributes,
    level_loaded: bool,
}

impl GameApp {
    fn new(project: ProjectConfig) -> Self {
        let mut world = World::new();
        world.insert_resource(Time::new());
        // KNOWN LIMITATION: Camera position and look-at target are hardcoded for a
        // top-down isometric view. These should be loaded from the project config
        // (e.g. `project.camera.position`, `project.camera.target`) to support
        // different game perspectives (first-person, side-scroller, etc.).
        world.insert_resource(Camera::new(
            Vec3::new(0.0, 12.0, 8.0),
            Vec3::new(0.0, 0.5, 0.0),
        ));
        world.insert_resource(PhysicsConfig::new());
        // KNOWN LIMITATION: Ambient light color and intensity are hardcoded.
        // A project-level lighting config (or per-level override) would allow
        // designers to set mood without code changes.
        world.insert_resource(AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.2,
        });
        world.insert_resource(Events::default());
        world.insert_resource(euca_input::InputState::new());
        world.insert_resource(euca_input::InputContextStack::new());
        world.insert_resource(euca_gameplay::camera::MobaCamera::default());
        world.insert_resource(euca_gameplay::player_input::ViewportSize {
            width: project.window.width as f32,
            height: project.window.height as f32,
        });
        world.insert_resource(PostProcessSettings::default());
        world.insert_resource(euca_core::Profiler::default());

        let attrs = WindowAttributes::default()
            .with_title(&project.window.title)
            .with_inner_size(winit::dpi::LogicalSize::new(
                project.window.width,
                project.window.height,
            ));

        Self {
            project,
            world,
            initialized: false,
            gpu: None,
            renderer: None,
            window_attrs: attrs,
            level_loaded: false,
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        // KNOWN LIMITATION: Default mesh set (plane, cube, sphere) and their
        // parameters (plane size 20.0, sphere radius 0.5 / subdivisions 16x32)
        // are hardcoded. A project config section for default assets or an asset
        // manifest would let projects declare their own starter primitives.
        let plane = renderer.upload_mesh(gpu, &Mesh::plane(20.0));
        let cube = renderer.upload_mesh(gpu, &Mesh::cube());
        let sphere = renderer.upload_mesh(gpu, &Mesh::sphere(0.5, 16, 32));

        // KNOWN LIMITATION: Grid texture resolution (512) and tile count (32) are
        // hardcoded. Ground material color [0.45, 0.45, 0.45] and roughness 0.95
        // are also fixed. These could be configurable per-level or per-project.
        let grid_tex = renderer.checkerboard_texture(gpu, 512, 32);
        let grid_mat = renderer.upload_material(
            gpu,
            &Material::new([0.45, 0.45, 0.45, 1.0], 0.0, 0.95).with_texture(grid_tex),
        );

        // KNOWN LIMITATION: The material palette is a fixed set of 12 named
        // colors. Projects cannot add custom named materials without code changes.
        // A data-driven palette loaded from the project config (or per-level
        // material definitions) would improve flexibility.
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
        let blue = blue.expect("blue material");

        let mut meshes = std::collections::HashMap::new();
        meshes.insert("cube".to_string(), cube);
        meshes.insert("sphere".to_string(), sphere);
        meshes.insert("plane".to_string(), plane);

        self.world
            .insert_resource(euca_agent::routes::DefaultAssets {
                meshes,
                materials,
                default_material: blue,
            });

        // KNOWN LIMITATION: Ground plane position, collider half-extents
        // (10.0, 0.01, 10.0), and mesh are hardcoded. Levels that need different
        // ground geometry or no ground at all currently require code changes.
        // A level-descriptor field for ground configuration would fix this.
        let g = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        self.world.insert(g, GlobalTransform::default());
        self.world.insert(g, MeshRenderer { mesh: plane });
        self.world.insert(g, MaterialRef { handle: grid_mat });
        self.world.insert(g, euca_physics::PhysicsBody::fixed());
        self.world
            .insert(g, euca_physics::Collider::aabb(10.0, 0.01, 10.0));

        // KNOWN LIMITATION: Directional light direction, color, and intensity are
        // hardcoded. A warm-toned sun angle works for the default MOBA-style scene
        // but other genres need different lighting. Per-level light configuration
        // in the level file would be the proper solution.
        self.world.spawn(DirectionalLight {
            direction: [0.4, -0.9, 0.25],
            color: [1.0, 0.95, 0.88],
            intensity: 2.5,
        });
    }

    fn load_level(&mut self) {
        let path = &self.project.default_level;
        match std::fs::read_to_string(path) {
            Ok(data) => match serde_json::from_str::<serde_json::Value>(&data) {
                Ok(level) => {
                    let count = euca_agent::load_level_into_world(&mut self.world, &level);
                    log::info!("Level loaded: {count} entities from {path}");
                }
                Err(e) => log::error!("Invalid level JSON in {path}: {e}"),
            },
            Err(e) => log::error!("Cannot read level file {path}: {e}"),
        }

        // Auto-follow player hero
        let hero = {
            let q =
                Query::<(euca_ecs::Entity, &euca_gameplay::player::PlayerHero)>::new(&self.world);
            q.iter().map(|(e, _)| e).next()
        };
        if let Some(hero) = hero
            && let Some(cam) = self
                .world
                .resource_mut::<euca_gameplay::camera::MobaCamera>()
        {
            cam.follow_entity = Some(hero);
        }

        // KNOWN LIMITATION: Navmesh grid bounds ([-12, -12] to [12, 12]),
        // cell_size (0.5), ground_y (0.0), and agent radius (0.5) are all
        // hardcoded. These should be derived from the level geometry or exposed
        // in the project/level config. In particular:
        //   - cell_size controls pathfinding resolution vs. performance trade-off
        //   - grid bounds should match the actual playable area
        //   - agent radius affects obstacle avoidance clearance
        if self.world.resource::<euca_nav::NavMesh>().is_none() {
            let config = euca_nav::GridConfig {
                min: [-12.0, -12.0],
                max: [12.0, 12.0],
                cell_size: 0.5,
                ground_y: 0.0,
            };
            let mesh = euca_nav::build_navmesh_from_world_with_radius(&self.world, config, 0.5);
            self.world.insert_resource(mesh);
        }
    }

    fn render_frame(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();

        // Run gameplay systems
        // KNOWN LIMITATION: The fallback delta time 0.016 (~60 FPS) is hardcoded.
        // This should come from a target-framerate project setting so games
        // targeting 30 FPS or 120 FPS get correct first-frame behaviour.
        let dt = self
            .world
            .resource::<Time>()
            .map(|t| t.delta)
            .unwrap_or(0.016);

        // Run core systems inline (simplified — no parallel schedule for standalone)
        physics_step_system(&mut self.world);
        euca_physics::character_controller_system(&mut self.world, dt);
        euca_gameplay::apply_damage_system(&mut self.world);
        euca_gameplay::death_check_system(&mut self.world);
        euca_gameplay::projectile_system(&mut self.world, dt);
        euca_gameplay::trigger_system(&mut self.world);
        euca_gameplay::ai_system(&mut self.world, dt);
        euca_gameplay::player_input::player_input_system(&mut self.world);
        euca_gameplay::player::player_command_system(&mut self.world, dt);
        euca_gameplay::auto_combat_system(&mut self.world, dt);
        euca_gameplay::game_state_system(&mut self.world, dt);
        euca_gameplay::on_death_rule_system(&mut self.world);
        euca_gameplay::timer_rule_system(&mut self.world, dt);
        euca_gameplay::health_below_rule_system(&mut self.world);
        euca_gameplay::on_score_rule_system(&mut self.world);
        euca_gameplay::on_phase_rule_system(&mut self.world);

        let respawn_delay = self
            .world
            .resource::<euca_gameplay::GameState>()
            .map(|s| s.config.respawn_delay);
        if let Some(_delay) = respawn_delay {
            euca_gameplay::respawn_system(&mut self.world, dt);
        }
        euca_gameplay::corpse_cleanup_system(&mut self.world, dt);

        // Attach visuals to rule-spawned entities
        let spawn_events: Vec<euca_gameplay::RuleSpawnEvent> = self
            .world
            .resource::<Events>()
            .map(|e| e.read::<euca_gameplay::RuleSpawnEvent>().cloned().collect())
            .unwrap_or_default();
        if let Some(assets) = self
            .world
            .resource::<euca_agent::routes::DefaultAssets>()
            .cloned()
        {
            for ev in spawn_events {
                if let Some(mesh_handle) = assets.mesh(&ev.mesh) {
                    self.world
                        .insert(ev.entity, MeshRenderer { mesh: mesh_handle });
                    let mat = ev
                        .color
                        .as_deref()
                        .and_then(|c| assets.material(c))
                        .unwrap_or(assets.default_material);
                    self.world.insert(ev.entity, MaterialRef { handle: mat });
                }
            }
        }

        euca_gameplay::gold_on_kill_system(&mut self.world);
        euca_gameplay::xp_on_kill_system(&mut self.world);
        euca_gameplay::ability_tick_system(&mut self.world, dt);
        euca_gameplay::use_ability_system(&mut self.world);
        euca_nav::pathfinding_system(&mut self.world);
        euca_nav::steering_system(&mut self.world, dt);

        if let Some(events) = self.world.resource_mut::<Events>() {
            events.update();
        }
        self.world.tick();

        // Input clear (after gameplay consumed it)
        if let Some(input) = self.world.resource_mut::<euca_input::InputState>() {
            input.begin_frame();
        }

        euca_scene::transform_propagation_system(&mut self.world);

        // MOBA camera
        euca_gameplay::camera::moba_camera_system(&mut self.world);

        // Render
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
                label: Some("game frame"),
            });

        let draw_commands = collect_draw_commands(&self.world);
        let light = {
            let query = Query::<&DirectionalLight>::new(&self.world);
            query.iter().next().cloned().unwrap_or_default()
        };
        let ambient = self
            .world
            .resource::<AmbientLight>()
            .cloned()
            .unwrap_or_default();
        let camera = self.world.resource::<Camera>().unwrap().clone();

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

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

/// Collect draw commands for all alive renderable entities.
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

impl ApplicationHandler for GameApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.initialized {
            return;
        }

        let (survey, wgpu_instance) = HardwareSurvey::detect();
        let window = event_loop
            .create_window(self.window_attrs.clone())
            .expect("Failed to create window");
        let gpu = GpuContext::new(window, &survey, &wgpu_instance);
        let renderer = Renderer::new(&gpu);
        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.initialized = true;
        self.setup_scene();

        if !self.level_loaded {
            self.load_level();
            self.level_loaded = true;
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        // Forward input to gameplay
        self.forward_input(&event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.logical_key == Key::Named(NamedKey::Escape)
                    && event.state == ElementState::Pressed =>
            {
                event_loop.exit()
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                    if let Some(r) = &mut self.renderer {
                        r.resize(gpu);
                    }
                }
                if let Some(vp) = self
                    .world
                    .resource_mut::<euca_gameplay::player_input::ViewportSize>()
                {
                    vp.width = size.width as f32;
                    vp.height = size.height as f32;
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl GameApp {
    fn forward_input(&mut self, event: &WindowEvent) {
        use euca_input::InputKey;

        let Some(input) = self.world.resource_mut::<euca_input::InputState>() else {
            return;
        };

        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let key_name = match &event.logical_key {
                    Key::Character(ch) => Some(ch.to_uppercase()),
                    Key::Named(named) => match named {
                        NamedKey::Space => Some("Space".to_string()),
                        NamedKey::Escape => Some("Escape".to_string()),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(name) = key_name {
                    match event.state {
                        ElementState::Pressed => input.press(InputKey::Key(name)),
                        ElementState::Released => input.release(InputKey::Key(name)),
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let key = match button {
                    winit::event::MouseButton::Left => Some(InputKey::MouseLeft),
                    winit::event::MouseButton::Right => Some(InputKey::MouseRight),
                    winit::event::MouseButton::Middle => Some(InputKey::MouseMiddle),
                    _ => None,
                };
                if let Some(k) = key {
                    match state {
                        ElementState::Pressed => input.press(k),
                        ElementState::Released => input.release(k),
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
}
