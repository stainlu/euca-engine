use euca_core::Time;
use euca_ecs::{Query, World};
use euca_editor::{
    EditorState, SceneFile, SpawnRequest, ToolbarAction, find_alive_entity, hierarchy_panel,
    inspector_panel, toolbar_panel,
};
use euca_math::{Transform, Vec3};
use euca_physics::{
    Collider, PhysicsBody, PhysicsConfig, Ray, Velocity, physics_step_system, raycast_collider,
};
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};

use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

struct EditorApp {
    world: World,
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
    // Stored handles for entity creation
    cube_mesh: Option<MeshHandle>,
    sphere_mesh: Option<MeshHandle>,
    default_material: Option<MaterialHandle>,
    // Modifier key tracking
    ctrl_held: bool,
    shift_held: bool,
}

impl EditorApp {
    fn new() -> Self {
        let (survey, wgpu_instance) = HardwareSurvey::detect();

        let mut world = World::new();
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

        Self {
            world,
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

        // Initialize gizmo (reuses cube mesh, uploads bright R/G/B materials)
        self.editor_state.gizmo = euca_editor::gizmo::init_gizmo(renderer, gpu, cube);

        // Grid texture for ground (dark lines on lighter background)
        let grid_tex = renderer.checkerboard_texture(gpu, 512, 32);
        let grid_mat = renderer.upload_material(
            gpu,
            &Material::new([0.45, 0.45, 0.45, 1.0], 0.0, 0.95).with_texture(grid_tex),
        );
        let red = renderer.upload_material(gpu, &Material::red_plastic());
        let blue = renderer.upload_material(gpu, &Material::blue_plastic());
        let gold = renderer.upload_material(gpu, &Material::gold());
        let green = renderer.upload_material(gpu, &Material::green());
        self.default_material = Some(blue);

        // Bright orange outline material for selection highlight
        self.outline_material =
            Some(renderer.upload_material(gpu, &Material::new([1.0, 0.6, 0.0, 1.0], 0.0, 1.0)));

        // Ground with grid texture
        let g = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        self.world.insert(g, GlobalTransform::default());
        self.world.insert(g, MeshRenderer { mesh: plane });
        self.world.insert(g, MaterialRef { handle: grid_mat });
        self.world.insert(g, PhysicsBody::fixed());
        self.world.insert(g, Collider::aabb(10.0, 0.01, 10.0));

        // Cubes
        let spawn = |w: &mut World, pos: Vec3, mesh: MeshHandle, mat: MaterialHandle, half: f32| {
            let e = w.spawn(LocalTransform(Transform::from_translation(pos)));
            w.insert(e, GlobalTransform::default());
            w.insert(e, MeshRenderer { mesh });
            w.insert(e, MaterialRef { handle: mat });
            w.insert(e, PhysicsBody::dynamic());
            w.insert(e, Velocity::default());
            w.insert(e, Collider::aabb(half, half, half).with_restitution(0.4));
        };

        let silver = renderer.upload_material(gpu, &Material::silver());
        let gray = renderer.upload_material(gpu, &Material::gray());

        // ── Static scene objects (arranged for visual appeal) ──

        let spawn_static =
            |w: &mut World, pos: Vec3, mesh: MeshHandle, mat: MaterialHandle, half: f32| {
                let e = w.spawn(LocalTransform(Transform::from_translation(pos)));
                w.insert(e, GlobalTransform::default());
                w.insert(e, MeshRenderer { mesh });
                w.insert(e, MaterialRef { handle: mat });
                w.insert(e, PhysicsBody::fixed());
                w.insert(e, Collider::aabb(half, half, half));
            };

        let spawn_sphere_static = |w: &mut World, pos: Vec3, mat: MaterialHandle| {
            let e = w.spawn(LocalTransform(Transform::from_translation(pos)));
            w.insert(e, GlobalTransform::default());
            w.insert(e, MeshRenderer { mesh: sphere });
            w.insert(e, MaterialRef { handle: mat });
            w.insert(e, PhysicsBody::fixed());
            w.insert(e, Collider::sphere(0.5));
        };

        // Center pedestal (stacked cubes)
        spawn_static(&mut self.world, Vec3::new(0.0, 0.5, 0.0), cube, gray, 0.5);
        spawn_static(&mut self.world, Vec3::new(0.0, 1.5, 0.0), cube, silver, 0.4);
        // Gold sphere on top of pedestal
        spawn_sphere_static(&mut self.world, Vec3::new(0.0, 2.5, 0.0), gold);

        // Four pillars in a square — taller
        for &(x, z) in &[(4.0, 4.0), (-4.0, 4.0), (4.0, -4.0), (-4.0, -4.0)] {
            spawn_static(&mut self.world, Vec3::new(x, 0.5, z), cube, gray, 0.35);
            spawn_static(&mut self.world, Vec3::new(x, 1.5, z), cube, gray, 0.35);
            spawn_static(&mut self.world, Vec3::new(x, 2.5, z), cube, gray, 0.35);
            // Colored sphere caps
            let mat = match (x > 0.0, z > 0.0) {
                (true, true) => red,
                (false, true) => blue,
                (true, false) => green,
                (false, false) => gold,
            };
            spawn_sphere_static(&mut self.world, Vec3::new(x, 3.5, z), mat);
        }

        // Front row — three material showcase cubes on small pedestals
        for (i, mat) in [red, silver, blue].iter().enumerate() {
            let x = (i as f32 - 1.0) * 2.5;
            spawn_static(&mut self.world, Vec3::new(x, 0.3, -3.0), cube, gray, 0.3);
            spawn_static(&mut self.world, Vec3::new(x, 0.9, -3.0), cube, *mat, 0.25);
        }

        // Dynamic objects (will fall when you press Play)
        spawn(&mut self.world, Vec3::new(-1.5, 5.0, 1.0), cube, red, 0.5);
        spawn(&mut self.world, Vec3::new(1.5, 7.0, -0.5), cube, blue, 0.5);
        spawn(&mut self.world, Vec3::new(0.0, 9.0, 0.5), cube, green, 0.5);

        // Floating gold spheres (dynamic — will drop on Play)
        let spawn_sphere_dyn = |w: &mut World, pos: Vec3, mat: MaterialHandle| {
            let e = w.spawn(LocalTransform(Transform::from_translation(pos)));
            w.insert(e, GlobalTransform::default());
            w.insert(e, MeshRenderer { mesh: sphere });
            w.insert(e, MaterialRef { handle: mat });
            w.insert(e, PhysicsBody::dynamic());
            w.insert(e, Velocity::default());
            w.insert(e, Collider::sphere(0.5).with_restitution(0.6));
        };

        spawn_sphere_dyn(&mut self.world, Vec3::new(2.5, 6.0, 2.0), gold);
        spawn_sphere_dyn(&mut self.world, Vec3::new(-2.5, 8.0, -1.5), silver);

        // Directional light — warm sunlight from upper-left
        self.world.spawn(DirectionalLight {
            direction: [0.4, -0.9, 0.25],
            color: [1.0, 0.95, 0.88],
            intensity: 2.5,
        });
    }

    fn reset_scene(&mut self) {
        // Despawn all entities
        let entities: Vec<euca_ecs::Entity> = {
            let query = euca_ecs::Query::<euca_ecs::Entity>::new(&self.world);
            query.iter().collect()
        };
        for entity in entities {
            self.world.despawn(entity);
        }
        // Reset physics world
        self.world.insert_resource(PhysicsConfig::new());
        // Re-create scene
        self.setup_scene();
        self.editor_state.selected_entity = None;
    }

    fn render_frame(&mut self) {
        // Handle reset
        if self.editor_state.reset_requested {
            self.editor_state.reset_requested = false;
            self.reset_scene();
        }

        self.world.resource_mut::<Time>().unwrap().update();
        let _elapsed = self.world.resource::<Time>().unwrap().elapsed as f32;

        // Tick simulation when playing
        if self.editor_state.should_tick() {
            physics_step_system(&mut self.world);
        }
        euca_scene::transform_propagation_system(&mut self.world);

        // User-controlled camera (orbit/pan/zoom)
        if self.right_mouse_down {
            self.cam_yaw += self.mouse_delta[0] * 0.005;
            self.cam_pitch = (self.cam_pitch - self.mouse_delta[1] * 0.005).clamp(0.05, 1.5);
        }
        if self.middle_mouse_down {
            let right = Vec3::new(self.cam_yaw.cos(), 0.0, -self.cam_yaw.sin());
            let up = Vec3::Y;
            self.cam_target =
                self.cam_target + right * (-self.mouse_delta[0] * 0.01 * self.cam_distance * 0.1);
            self.cam_target =
                self.cam_target + up * (self.mouse_delta[1] * 0.01 * self.cam_distance * 0.1);
        }
        self.mouse_delta = [0.0, 0.0];

        let cam = self.world.resource_mut::<Camera>().unwrap();
        cam.eye = Vec3::new(
            self.cam_target.x + self.cam_yaw.sin() * self.cam_pitch.cos() * self.cam_distance,
            self.cam_target.y + self.cam_pitch.sin() * self.cam_distance,
            self.cam_target.z + self.cam_yaw.cos() * self.cam_pitch.cos() * self.cam_distance,
        );
        cam.target = self.cam_target;

        // Get surface texture
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
        {
            let mut draw_commands: Vec<DrawCommand> = {
                let query =
                    Query::<(&GlobalTransform, &MeshRenderer, &MaterialRef)>::new(&self.world);
                query
                    .iter()
                    .map(|(gt, mr, mat)| DrawCommand {
                        mesh: mr.mesh,
                        material: mat.handle,
                        model_matrix: gt.0.to_matrix(),
                    })
                    .collect()
            };

            // Selection outline: draw selected entity again at 1.06x scale with orange material
            if let (Some(sel_idx), Some(outline_mat)) =
                (self.editor_state.selected_entity, self.outline_material)
            {
                // Find the selected entity
                for g in 0..16u32 {
                    let entity = euca_ecs::Entity::from_raw(sel_idx, g);
                    if !self.world.is_alive(entity) {
                        continue;
                    }
                    if let (Some(gt), Some(mr)) = (
                        self.world.get::<GlobalTransform>(entity),
                        self.world.get::<MeshRenderer>(entity),
                    ) {
                        // Skip outline for large objects (avoids z-fighting on ground plane)
                        let max_scale = gt.0.scale.x.max(gt.0.scale.y).max(gt.0.scale.z);
                        if max_scale < 5.0 {
                            let mut t = gt.0;
                            t.scale = t.scale * 1.06;
                            draw_commands.push(DrawCommand {
                                mesh: mr.mesh,
                                material: outline_mat,
                                model_matrix: t.to_matrix(),
                            });
                        }
                    }
                    break;
                }
            }

            // Gizmo: draw axis handles on selected entity
            if let Some(sel_idx) = self.editor_state.selected_entity {
                if let Some(entity) = find_alive_entity(&self.world, sel_idx) {
                    if let Some(gt) = self.world.get::<GlobalTransform>(entity) {
                        let camera = self.world.resource::<Camera>().unwrap();
                        let gizmo_cmds = euca_editor::gizmo::gizmo_draw_commands(
                            gt.0.translation,
                            camera.eye,
                            &self.editor_state.gizmo,
                        );
                        draw_commands.extend(gizmo_cmds);
                    }
                }
            }

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

            let renderer = self.renderer.as_ref().unwrap();
            renderer.render_to_view(
                gpu,
                &camera,
                &light,
                &ambient,
                &draw_commands,
                &view,
                &mut encoder,
            );
        }

        // === 2. Render egui on top ===
        let window = self.window.as_ref().unwrap();
        let egui_winit = self.egui_winit.as_mut().unwrap();
        let raw_input = egui_winit.take_egui_input(window);

        let mut spawn_request = None;
        let mut toolbar_action = None;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            let dt = self
                .world
                .resource::<Time>()
                .map(|t| t.delta)
                .unwrap_or(0.0);
            toolbar_action = toolbar_panel(ctx, &mut self.editor_state, &self.world, dt);
            spawn_request = hierarchy_panel(ctx, &mut self.editor_state, &self.world);
            inspector_panel(ctx, &mut self.editor_state, &mut self.world);
        });

        // Handle save/load
        if let Some(action) = toolbar_action {
            match action {
                ToolbarAction::SaveScene => {
                    let scene = SceneFile::capture(&self.world);
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
                        // Clear existing entities
                        let entities: Vec<euca_ecs::Entity> = {
                            let query = Query::<euca_ecs::Entity>::new(&self.world);
                            query.iter().collect()
                        };
                        for entity in entities {
                            self.world.despawn(entity);
                        }
                        // Rebuild from scene file
                        let cube_mesh = self.cube_mesh;
                        let sphere_mesh = self.sphere_mesh;
                        euca_editor::load_scene_into_world(
                            &mut self.world,
                            &scene,
                            &|name| match name {
                                n if n.contains("0") => cube_mesh, // mesh_0 = cube (first uploaded)
                                n if n.contains("1") => sphere_mesh, // mesh_1 = sphere
                                _ => cube_mesh,
                            },
                            6, // number of uploaded materials
                        );
                        // Re-add light
                        self.world.spawn(DirectionalLight {
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

        // Handle entity spawn requests
        if let Some(req) = spawn_request {
            let pos = Vec3::new(0.0, 2.0, 0.0);
            let e = self
                .world
                .spawn(LocalTransform(Transform::from_translation(pos)));
            self.world.insert(e, GlobalTransform::default());
            match req {
                SpawnRequest::Cube => {
                    if let Some(mesh) = self.cube_mesh {
                        self.world.insert(e, MeshRenderer { mesh });
                    }
                    if let Some(mat) = self.default_material {
                        self.world.insert(e, MaterialRef { handle: mat });
                    }
                    self.world.insert(e, Collider::aabb(0.5, 0.5, 0.5));
                }
                SpawnRequest::Sphere => {
                    if let Some(mesh) = self.sphere_mesh {
                        self.world.insert(e, MeshRenderer { mesh });
                    }
                    if let Some(mat) = self.default_material {
                        self.world.insert(e, MaterialRef { handle: mat });
                    }
                    self.world.insert(e, Collider::sphere(0.5));
                }
                SpawnRequest::Empty => {}
            }
            self.editor_state.selected_entity = Some(e.index());
            // Track spawn for undo
            self.editor_state
                .undo
                .push(euca_editor::undo::UndoAction::SpawnEntity {
                    entity_index: e.index(),
                });
        }

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

        // egui render pass: LoadOp::Load (don't clear — render ON TOP of 3D scene)
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

        // Submit all commands
        let mut cmds: Vec<wgpu::CommandBuffer> = vec![encoder.finish()];
        cmds.extend(user_bufs);
        gpu.queue.submit(cmds);

        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
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
                // Delete selected entity (with undo support)
                if let Some(idx) = self.editor_state.selected_entity {
                    if let Some(e) = find_alive_entity(&self.world, idx) {
                        // Capture state before despawn
                        let transform = self
                            .world
                            .get::<LocalTransform>(e)
                            .map(|lt| lt.0)
                            .unwrap_or_default();
                        let mesh = self.world.get::<MeshRenderer>(e).map(|mr| mr.mesh);
                        let material = self.world.get::<MaterialRef>(e).map(|mr| mr.handle);
                        let collider = self.world.get::<Collider>(e).cloned();
                        self.world.despawn(e);
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
                // Focus camera on selected entity
                if let Some(idx) = self.editor_state.selected_entity {
                    if let Some(e) = find_alive_entity(&self.world, idx) {
                        if let Some(gt) = self.world.get::<GlobalTransform>(e) {
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
                self.editor_state.undo.undo(&mut self.world);
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
                self.editor_state.undo.redo(&mut self.world);
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

                // Update gizmo drag if active
                if self.editor_state.gizmo.active_drag.is_some() {
                    self.update_gizmo_drag();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    winit::event::MouseButton::Left => {
                        if pressed {
                            // Try gizmo pick first; fall through to entity pick
                            if !self.try_begin_gizmo_drag() {
                                self.pick_entity_at_cursor();
                            }
                        } else {
                            self.end_gizmo_drag();
                        }
                    }
                    winit::event::MouseButton::Right => {
                        self.right_mouse_down = pressed;
                    }
                    winit::event::MouseButton::Middle => {
                        self.middle_mouse_down = pressed;
                    }
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
        let camera = match self.world.resource::<Camera>() {
            Some(c) => c.clone(),
            None => return,
        };

        let screen_w = gpu.surface_config.width as f32;
        let screen_h = gpu.surface_config.height as f32;
        let (ray_origin, ray_dir) =
            camera.screen_to_ray(self.mouse_pos[0], self.mouse_pos[1], screen_w, screen_h);
        let ray = Ray::new(ray_origin, ray_dir);

        // Test against all entities with colliders
        let mut closest: Option<(euca_ecs::Entity, f32)> = None;

        let candidates: Vec<(euca_ecs::Entity, Vec3, Collider)> = {
            let query = Query::<(euca_ecs::Entity, &GlobalTransform, &Collider)>::new(&self.world);
            query
                .iter()
                .map(|(e, gt, col)| (e, gt.0.translation, col.clone()))
                .collect()
        };

        for (entity, pos, collider) in &candidates {
            if let Some(hit) = raycast_collider(&ray, *pos, collider) {
                if hit.t >= 0.0 {
                    if closest.is_none() || hit.t < closest.unwrap().1 {
                        closest = Some((*entity, hit.t));
                    }
                }
            }
        }

        self.editor_state.selected_entity = closest.map(|(e, _)| e.index());
    }

    /// Try to begin a gizmo drag. Returns true if a gizmo axis was hit.
    fn try_begin_gizmo_drag(&mut self) -> bool {
        let sel_idx = match self.editor_state.selected_entity {
            Some(idx) => idx,
            None => return false,
        };
        let entity = match find_alive_entity(&self.world, sel_idx) {
            Some(e) => e,
            None => return false,
        };
        let entity_pos = match self.world.get::<GlobalTransform>(entity) {
            Some(gt) => gt.0.translation,
            None => return false,
        };
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return false,
        };
        let camera = match self.world.resource::<Camera>() {
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
            // Compute grab point on the axis line
            let axis_dir = axis.direction();
            let grab_t =
                Vec3::closest_line_param(entity_pos, axis_dir, ray_origin, ray_dir.normalize());
            let grab_point = entity_pos + axis_dir * grab_t;

            let current_transform = self
                .world
                .get::<LocalTransform>(entity)
                .map(|lt| lt.0)
                .unwrap_or_default();

            self.editor_state.gizmo.active_drag = Some(euca_editor::gizmo::GizmoDrag {
                axis,
                entity_index: sel_idx,
                start_position: entity_pos,
                grab_point,
            });

            // Begin undo tracking for this drag
            self.editor_state
                .undo
                .begin_drag(sel_idx, current_transform);

            return true;
        }

        false
    }

    /// End an active gizmo drag and commit the undo action.
    fn end_gizmo_drag(&mut self) {
        if let Some(drag) = self.editor_state.gizmo.active_drag.take() {
            if let Some(entity) = find_alive_entity(&self.world, drag.entity_index) {
                let current = self
                    .world
                    .get::<LocalTransform>(entity)
                    .map(|lt| lt.0)
                    .unwrap_or_default();
                self.editor_state.undo.end_drag(current);
            } else {
                self.editor_state.undo.cancel_drag();
            }
        }
    }

    /// Update entity position during an active gizmo drag.
    fn update_gizmo_drag(&mut self) {
        let drag = match &self.editor_state.gizmo.active_drag {
            Some(d) => d.clone(),
            None => return,
        };
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };
        let camera = match self.world.resource::<Camera>() {
            Some(c) => c.clone(),
            None => return,
        };

        let screen_w = gpu.surface_config.width as f32;
        let screen_h = gpu.surface_config.height as f32;
        let (ray_origin, ray_dir) =
            camera.screen_to_ray(self.mouse_pos[0], self.mouse_pos[1], screen_w, screen_h);

        let new_pos = euca_editor::gizmo::update_gizmo_drag(&drag, ray_origin, ray_dir.normalize());

        if let Some(entity) = find_alive_entity(&self.world, drag.entity_index) {
            if let Some(lt) = self.world.get_mut::<LocalTransform>(entity) {
                lt.0.translation = new_pos;
            }
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = EditorApp::new();
    event_loop.run_app(&mut app).unwrap();
}
