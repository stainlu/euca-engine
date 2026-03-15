use euca_core::Time;
use euca_ecs::{Query, World};
use euca_editor::{EditorState, hierarchy_panel, inspector_panel, toolbar_panel};
use euca_math::{Transform, Vec3};
use euca_physics::{Collider, PhysicsBody, PhysicsConfig, Velocity, physics_step_system};
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
    editor_state: EditorState,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    egui_ctx: egui::Context,
    egui_winit: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    window_attrs: WindowAttributes,
}

impl EditorApp {
    fn new() -> Self {
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
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        let cube = renderer.upload_mesh(gpu, &Mesh::cube());
        let sphere = renderer.upload_mesh(gpu, &Mesh::sphere(0.5, 16, 32));
        let plane = renderer.upload_mesh(gpu, &Mesh::plane(20.0));

        let gray = renderer.upload_material(gpu, &Material::gray());
        let red = renderer.upload_material(gpu, &Material::red_plastic());
        let blue = renderer.upload_material(gpu, &Material::blue_plastic());
        let gold = renderer.upload_material(gpu, &Material::gold());
        let green = renderer.upload_material(gpu, &Material::green());

        // Ground
        let g = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        self.world.insert(g, GlobalTransform::default());
        self.world.insert(g, MeshRenderer { mesh: plane });
        self.world.insert(g, MaterialRef { handle: gray });
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

        spawn(&mut self.world, Vec3::new(0.0, 4.0, 0.0), cube, red, 0.5);
        spawn(&mut self.world, Vec3::new(1.5, 6.0, 0.5), cube, blue, 0.5);
        spawn(&mut self.world, Vec3::new(-1.0, 8.0, -0.5), cube, gold, 0.5);
        spawn(&mut self.world, Vec3::new(0.5, 10.0, 1.0), cube, green, 0.5);

        // Sphere
        let s = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                -2.0, 7.0, 1.0,
            ))));
        self.world.insert(s, GlobalTransform::default());
        self.world.insert(s, MeshRenderer { mesh: sphere });
        self.world.insert(s, MaterialRef { handle: gold });
        self.world.insert(s, PhysicsBody::dynamic());
        self.world.insert(s, Velocity::default());
        self.world
            .insert(s, Collider::sphere(0.5).with_restitution(0.6));

        // Light
        self.world.spawn(DirectionalLight {
            direction: [0.5, -1.0, 0.3],
            color: [1.0, 0.98, 0.95],
            intensity: 2.0,
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
        let elapsed = self.world.resource::<Time>().unwrap().elapsed as f32;

        // Tick simulation when playing
        if self.editor_state.should_tick() {
            physics_step_system(&mut self.world);
        }
        euca_scene::transform_propagation_system(&mut self.world);

        // Orbit camera
        let cam = self.world.resource_mut::<Camera>().unwrap();
        let angle = elapsed * 0.2;
        let radius = 12.0;
        cam.eye = Vec3::new(angle.cos() * radius, 6.0, angle.sin() * radius);

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
            let draw_commands: Vec<DrawCommand> = {
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

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            toolbar_panel(ctx, &mut self.editor_state, &self.world);
            hierarchy_panel(ctx, &mut self.editor_state, &self.world);
            inspector_panel(ctx, &mut self.editor_state, &mut self.world);
            // Central panel is transparent — 3D scene shows through
        });

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
            let gpu = GpuContext::new(window);
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
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = EditorApp::new();
    event_loop.run_app(&mut app).unwrap();
}
