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

struct TextureDemoApp {
    world: World,
    survey: HardwareSurvey,
    wgpu_instance: wgpu::Instance,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    window_attrs: WindowAttributes,
}

impl TextureDemoApp {
    fn new() -> Self {
        let (survey, wgpu_instance) = HardwareSurvey::detect();

        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(Camera::new(Vec3::new(4.0, 4.0, 4.0), Vec3::ZERO));
        world.insert_resource(AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.15,
        });

        Self {
            world,
            survey,
            wgpu_instance,
            gpu: None,
            renderer: None,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — Texture Demo (PBR)")
                .with_inner_size(winit::dpi::LogicalSize::new(900, 700)),
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        // Upload meshes
        let cube_mesh = renderer.upload_mesh(gpu, &Mesh::cube());
        let sphere_mesh = renderer.upload_mesh(gpu, &Mesh::sphere(0.6, 24, 32));
        let plane_mesh = renderer.upload_mesh(gpu, &Mesh::plane(12.0));

        // Create a checkerboard texture
        let checker = renderer.checkerboard_texture(gpu, 256, 32);

        // Textured ground plane: checkerboard with white tint
        let ground_mat = renderer.upload_material(gpu, &Material::textured(checker));

        // Textured cube: checkerboard tinted red
        let red_checker_mat = renderer.upload_material(
            gpu,
            &Material::new([1.0, 0.3, 0.3, 1.0], 0.0, 0.6).with_texture(checker),
        );

        // Color-only materials (backward compatibility)
        let gold_mat = renderer.upload_material(gpu, &Material::gold());
        let blue_mat = renderer.upload_material(gpu, &Material::blue_plastic());

        // Ground plane
        let ground = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                0.0, -1.0, 0.0,
            ))));
        self.world.insert(ground, GlobalTransform::default());
        self.world.insert(ground, MeshRenderer { mesh: plane_mesh });
        self.world
            .insert(ground, MaterialRef { handle: ground_mat });

        // Textured spinning cube (red checkerboard)
        let cube = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                -2.0, 0.0, 0.0,
            ))));
        self.world.insert(cube, GlobalTransform::default());
        self.world.insert(cube, MeshRenderer { mesh: cube_mesh });
        self.world.insert(
            cube,
            MaterialRef {
                handle: red_checker_mat,
            },
        );
        self.world.insert(cube, Spin { speed: 1.0 });

        // Gold sphere (color only — no texture)
        let sphere = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                0.0, 0.0, 0.0,
            ))));
        self.world.insert(sphere, GlobalTransform::default());
        self.world
            .insert(sphere, MeshRenderer { mesh: sphere_mesh });
        self.world.insert(sphere, MaterialRef { handle: gold_mat });
        self.world.insert(sphere, Spin { speed: 0.5 });

        // Blue cube (color only)
        let blue_cube = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                2.0, 0.0, 0.0,
            ))));
        self.world.insert(blue_cube, GlobalTransform::default());
        self.world
            .insert(blue_cube, MeshRenderer { mesh: cube_mesh });
        self.world
            .insert(blue_cube, MaterialRef { handle: blue_mat });
        self.world.insert(blue_cube, Spin { speed: 1.5 });

        // Light
        self.world.spawn(DirectionalLight {
            direction: [-0.5, -1.0, -0.3],
            color: [1.0, 0.98, 0.95],
            intensity: 2.5,
        });
    }

    fn update_and_render(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();
        let elapsed = self.world.resource::<Time>().unwrap().elapsed as f32;

        // Spin entities
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

        // Orbit camera
        let cam = self.world.resource_mut::<Camera>().unwrap();
        let angle = elapsed * 0.25;
        cam.eye = Vec3::new(angle.cos() * 7.0, 3.5, angle.sin() * 7.0);

        // Collect draw commands
        let draw_commands: Vec<DrawCommand> = {
            let query = Query::<(&GlobalTransform, &MeshRenderer, &MaterialRef)>::new(&self.world);
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

impl ApplicationHandler for TextureDemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let window = event_loop.create_window(self.window_attrs.clone()).unwrap();
            let gpu = GpuContext::new(window, &self.survey, &self.wgpu_instance);
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
    let mut app = TextureDemoApp::new();
    event_loop.run_app(&mut app).unwrap();
}
