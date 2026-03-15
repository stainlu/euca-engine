use euca_ecs::{Entity, Query, World};
use euca_math::{Vec3, Quat, Transform};
use euca_scene::{LocalTransform, GlobalTransform};
use euca_render::{GpuContext, Renderer, Mesh, MeshHandle, MeshRenderer, Camera, DrawCommand};
use euca_core::Time;

use winit::application::ApplicationHandler;
use winit::event::{WindowEvent, ElementState, KeyEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId, WindowAttributes};

/// Component marking an entity as spinning.
#[derive(Clone, Copy, Debug)]
struct Spin {
    speed: f32,
}

struct HelloCubesApp {
    world: World,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    cube_mesh: Option<MeshHandle>,
    window_attrs: WindowAttributes,
}

impl HelloCubesApp {
    fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(Camera::new(
            Vec3::new(3.0, 3.0, 3.0),
            Vec3::ZERO,
        ));

        Self {
            world,
            gpu: None,
            renderer: None,
            cube_mesh: None,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — Hello Cubes")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        // Upload cube meshes with different colors
        let red_cube = Mesh::cube([0.9, 0.2, 0.2]);
        let green_cube = Mesh::cube([0.2, 0.9, 0.2]);
        let blue_cube = Mesh::cube([0.2, 0.2, 0.9]);

        let red_handle = renderer.upload_mesh(gpu, &red_cube);
        let green_handle = renderer.upload_mesh(gpu, &green_cube);
        let blue_handle = renderer.upload_mesh(gpu, &blue_cube);

        self.cube_mesh = Some(red_handle);

        // Spawn entities
        let spawn_cube = |world: &mut World, pos: Vec3, mesh: MeshHandle, speed: f32| -> Entity {
            let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
            world.insert(e, GlobalTransform::default());
            world.insert(e, MeshRenderer { mesh });
            world.insert(e, Spin { speed });
            e
        };

        spawn_cube(&mut self.world, Vec3::new(-2.0, 0.0, 0.0), red_handle, 1.0);
        spawn_cube(&mut self.world, Vec3::new(0.0, 0.0, 0.0), green_handle, 1.5);
        spawn_cube(&mut self.world, Vec3::new(2.0, 0.0, 0.0), blue_handle, 2.0);
    }

    fn update_and_render(&mut self) {
        // Update time
        self.world.resource_mut::<Time>().unwrap().update();

        let delta = self.world.resource::<Time>().unwrap().delta;
        let elapsed = self.world.resource::<Time>().unwrap().elapsed as f32;

        // Spin system: rotate entities with Spin component
        let updates: Vec<(Entity, f32)> = {
            let query = Query::<(Entity, &Spin)>::new(&self.world);
            query.iter().map(|(e, s)| (e, s.speed)).collect()
        };
        for (entity, speed) in updates {
            if let Some(lt) = self.world.get_mut::<LocalTransform>(entity) {
                lt.0.rotation = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), elapsed * speed);
            }
        }

        // Transform propagation
        euca_scene::transform_propagation_system(&mut self.world);

        // Orbit camera
        let camera = self.world.resource_mut::<Camera>().unwrap();
        let angle = elapsed * 0.3;
        let radius = 6.0;
        camera.eye = Vec3::new(angle.cos() * radius, 3.0, angle.sin() * radius);

        // Collect draw commands
        let draw_commands: Vec<DrawCommand> = {
            let query = Query::<(&GlobalTransform, &MeshRenderer)>::new(&self.world);
            query
                .iter()
                .map(|(gt, mr)| DrawCommand {
                    mesh: mr.mesh,
                    model_matrix: gt.0.to_matrix(),
                })
                .collect()
        };

        // Render
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_ref().unwrap();
        let camera = self.world.resource::<Camera>().unwrap();
        renderer.draw(gpu, camera, &draw_commands);
    }
}

impl ApplicationHandler for HelloCubesApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let window = event_loop
                .create_window(self.window_attrs.clone())
                .expect("Failed to create window");

            let gpu = GpuContext::new(window);
            let renderer = Renderer::new(&gpu);
            self.gpu = Some(gpu);
            self.renderer = Some(renderer);

            self.setup_scene();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key: Key::Named(NamedKey::Escape),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                    if let Some(renderer) = &mut self.renderer {
                        renderer.resize(gpu);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_and_render();
                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = HelloCubesApp::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}
