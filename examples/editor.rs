use euca_core::Time;
use euca_ecs::World;
use euca_editor::{EditorState, hierarchy_panel, inspector_panel, toolbar_panel};
use euca_math::{Transform, Vec3};
use euca_physics::{PhysicsBody, PhysicsCollider, PhysicsWorld, physics_step_system};
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
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    egui_ctx: egui::Context,
    egui_winit: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    window_attrs: WindowAttributes,
}

impl EditorApp {
    fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(PhysicsWorld::new());

        // Spawn some entities for the editor to inspect
        let spawn_obj = |w: &mut World, pos: Vec3, _name: &str| {
            let e = w.spawn(LocalTransform(Transform::from_translation(pos)));
            w.insert(e, GlobalTransform::default());
            w.insert(e, PhysicsBody::dynamic());
            w.insert(e, PhysicsCollider::cuboid(0.5, 0.5, 0.5));
        };

        spawn_obj(&mut world, Vec3::new(0.0, 5.0, 0.0), "Red Cube");
        spawn_obj(&mut world, Vec3::new(2.0, 7.0, 1.0), "Blue Cube");
        spawn_obj(&mut world, Vec3::new(-1.0, 9.0, -0.5), "Gold Cube");

        // Ground
        let g = world.spawn(LocalTransform(Transform::IDENTITY));
        world.insert(g, GlobalTransform::default());
        world.insert(g, PhysicsBody::fixed());
        world.insert(g, PhysicsCollider::cuboid(10.0, 0.01, 10.0));

        Self {
            world,
            editor_state: EditorState::new(),
            window: None,
            device: None,
            queue: None,
            surface: None,
            surface_config: None,
            egui_ctx: egui::Context::default(),
            egui_winit: None,
            egui_renderer: None,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — Editor")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 800)),
        }
    }

    fn render_frame(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();

        if self.editor_state.should_tick() {
            physics_step_system(&mut self.world);
        }
        euca_scene::transform_propagation_system(&mut self.world);

        let window = self.window.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let config = self.surface_config.as_ref().unwrap();

        let output = match surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Run egui
        let egui_winit = self.egui_winit.as_mut().unwrap();
        let raw_input = egui_winit.take_egui_input(window);

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            toolbar_panel(ctx, &mut self.editor_state, &self.world);
            hierarchy_panel(ctx, &mut self.editor_state, &self.world);
            inspector_panel(ctx, &mut self.editor_state, &mut self.world);

            egui::CentralPanel::default().show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.heading("3D Viewport (coming soon)");
                });
            });
        });

        egui_winit.handle_platform_output(window, full_output.platform_output);

        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [config.width, config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        let egui_renderer = self.egui_renderer.as_mut().unwrap();
        for (id, delta) in &full_output.textures_delta.set {
            egui_renderer.update_texture(device, queue, *id, delta);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("editor"),
        });

        let user_bufs =
            egui_renderer.update_buffers(device, queue, &mut encoder, &paint_jobs, &screen_desc);

        // Clear + egui render in one pass
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("editor pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.12,
                            g: 0.12,
                            b: 0.15,
                            a: 1.0,
                        }),
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
        queue.submit(cmds);

        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }

        output.present();
    }
}

impl ApplicationHandler for EditorApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(event_loop.create_window(self.window_attrs.clone()).unwrap());

            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let surface = instance.create_surface(window.clone()).unwrap();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    ..Default::default()
                }))
                .unwrap();
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                    .unwrap();

            let size = window.inner_size();
            let caps = surface.get_capabilities(&adapter);
            let format = caps
                .formats
                .iter()
                .find(|f| f.is_srgb())
                .copied()
                .unwrap_or(caps.formats[0]);
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&device, &config);

            let egui_winit = egui_winit::State::new(
                self.egui_ctx.clone(),
                egui::ViewportId::ROOT,
                &*window,
                Some(window.scale_factor() as f32),
                None,
                None,
            );
            let egui_renderer =
                egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());

            self.window = Some(window);
            self.device = Some(device);
            self.queue = Some(queue);
            self.surface = Some(surface);
            self.surface_config = Some(config);
            self.egui_winit = Some(egui_winit);
            self.egui_renderer = Some(egui_renderer);
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
                if let (Some(surface), Some(config), Some(device)) =
                    (&self.surface, &mut self.surface_config, &self.device)
                {
                    config.width = size.width.max(1);
                    config.height = size.height.max(1);
                    surface.configure(device, config);
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
