use euca_core::Time;
use euca_ecs::{Entity, Query, World};
use euca_math::{Quat, Transform, Vec3};
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{WindowAttributes, WindowId};

#[derive(Clone, Copy, Debug)]
struct Spin {
    speed: f32,
}

struct HelloCubesApp {
    world: World,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    window_attrs: WindowAttributes,
}

impl HelloCubesApp {
    fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(Camera::new(Vec3::new(3.0, 3.0, 3.0), Vec3::ZERO));
        world.insert_resource(AmbientLight::default());

        Self {
            world,
            gpu: None,
            renderer: None,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — Hello Cubes (PBR)")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        let cube_mesh = renderer.upload_mesh(gpu, &Mesh::cube());
        let red_mat = renderer.upload_material(gpu, &Material::red_plastic());
        let green_mat = renderer.upload_material(gpu, &Material::green());
        let blue_mat = renderer.upload_material(gpu, &Material::blue_plastic());

        let spawn = |world: &mut World, pos: Vec3, mat: MaterialHandle, speed: f32| {
            let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
            world.insert(e, GlobalTransform::default());
            world.insert(e, MeshRenderer { mesh: cube_mesh });
            world.insert(e, MaterialRef { handle: mat });
            world.insert(e, Spin { speed });
        };

        spawn(&mut self.world, Vec3::new(-2.0, 0.0, 0.0), red_mat, 1.0);
        spawn(&mut self.world, Vec3::new(0.0, 0.0, 0.0), green_mat, 1.5);
        spawn(&mut self.world, Vec3::new(2.0, 0.0, 0.0), blue_mat, 2.0);

        // Light
        self.world.spawn(DirectionalLight::default());
    }

    fn update_and_render(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();
        let elapsed = self.world.resource::<Time>().unwrap().elapsed as f32;

        let updates: Vec<(Entity, f32)> = {
            let query = Query::<(Entity, &Spin)>::new(&self.world);
            query.iter().map(|(e, s)| (e, s.speed)).collect()
        };
        for (entity, speed) in updates {
            if let Some(lt) = self.world.get_mut::<LocalTransform>(entity) {
                lt.0.rotation = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), elapsed * speed);
            }
        }

        euca_scene::transform_propagation_system(&mut self.world);

        let cam = self.world.resource_mut::<Camera>().unwrap();
        let angle = elapsed * 0.3;
        cam.eye = Vec3::new(angle.cos() * 6.0, 3.0, angle.sin() * 6.0);

        let draw_commands: Vec<DrawCommand> = {
            let query = Query::<(&GlobalTransform, &MeshRenderer, &MaterialRef)>::new(&self.world);
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

        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_ref().unwrap();
        renderer.draw(gpu, &camera, &light, &ambient, &draw_commands);
    }
}

impl ApplicationHandler for HelloCubesApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let window = event_loop.create_window(self.window_attrs.clone()).unwrap();
            let gpu = GpuContext::new(window);
            let renderer = Renderer::new(&gpu);
            self.gpu = Some(gpu);
            self.renderer = Some(renderer);
            self.setup_scene();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
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
    let event_loop = EventLoop::new().unwrap();
    let mut app = HelloCubesApp::new();
    event_loop.run_app(&mut app).unwrap();
}
