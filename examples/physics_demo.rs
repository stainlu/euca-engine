use euca_core::Time;
use euca_ecs::{Query, World};
use euca_math::{Transform, Vec3};
use euca_physics::*;
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{WindowAttributes, WindowId};

struct PhysicsDemoApp {
    world: World,
    survey: HardwareSurvey,
    wgpu_instance: wgpu::Instance,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    cube_mesh: Option<MeshHandle>,
    sphere_mesh: Option<MeshHandle>,
    plane_mesh: Option<MeshHandle>,
    window_attrs: WindowAttributes,
}

impl PhysicsDemoApp {
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
            gpu: None,
            renderer: None,
            cube_mesh: None,
            sphere_mesh: None,
            plane_mesh: None,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — Physics Demo (PBR)")
                .with_inner_size(winit::dpi::LogicalSize::new(1024, 768)),
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        // Upload meshes
        let cube_mesh = renderer.upload_mesh(gpu, &Mesh::cube());
        let sphere_mesh = renderer.upload_mesh(gpu, &Mesh::sphere(0.5, 16, 32));
        let plane_mesh = renderer.upload_mesh(gpu, &Mesh::plane(20.0));
        self.cube_mesh = Some(cube_mesh);
        self.sphere_mesh = Some(sphere_mesh);
        self.plane_mesh = Some(plane_mesh);

        // Upload materials
        let gray_mat = renderer.upload_material(gpu, &Material::gray());
        let red_mat = renderer.upload_material(gpu, &Material::red_plastic());
        let blue_mat = renderer.upload_material(gpu, &Material::blue_plastic());
        let gold_mat = renderer.upload_material(gpu, &Material::gold());
        let green_mat = renderer.upload_material(gpu, &Material::green());

        // Ground plane (static)
        let ground = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                0.0, 0.0, 0.0,
            ))));
        self.world.insert(ground, GlobalTransform::default());
        self.world.insert(ground, MeshRenderer { mesh: plane_mesh });
        self.world.insert(ground, MaterialRef { handle: gray_mat });
        self.world.insert(ground, PhysicsBody::fixed());
        self.world.insert(ground, Collider::aabb(10.0, 0.01, 10.0));

        // Spawn cubes at different heights
        let spawn = |world: &mut World,
                     pos: Vec3,
                     mesh: MeshHandle,
                     mat: MaterialHandle,
                     half_size: f32| {
            let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
            world.insert(e, GlobalTransform::default());
            world.insert(e, MeshRenderer { mesh });
            world.insert(e, MaterialRef { handle: mat });
            world.insert(e, PhysicsBody::dynamic());
            world.insert(e, euca_physics::Velocity::default());
            world.insert(
                e,
                Collider::aabb(half_size, half_size, half_size).with_restitution(0.5),
            );
        };

        spawn(
            &mut self.world,
            Vec3::new(0.0, 5.0, 0.0),
            cube_mesh,
            red_mat,
            0.5,
        );
        spawn(
            &mut self.world,
            Vec3::new(1.5, 7.0, 0.5),
            cube_mesh,
            blue_mat,
            0.5,
        );
        spawn(
            &mut self.world,
            Vec3::new(-1.0, 9.0, -0.5),
            cube_mesh,
            gold_mat,
            0.5,
        );
        spawn(
            &mut self.world,
            Vec3::new(0.5, 11.0, 1.0),
            cube_mesh,
            green_mat,
            0.5,
        );

        // Spawn a sphere
        let s = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                -2.0, 8.0, 1.0,
            ))));
        self.world.insert(s, GlobalTransform::default());
        self.world.insert(s, MeshRenderer { mesh: sphere_mesh });
        self.world.insert(s, MaterialRef { handle: gold_mat });
        self.world.insert(s, PhysicsBody::dynamic());
        self.world.insert(s, euca_physics::Velocity::default());
        self.world
            .insert(s, Collider::sphere(0.5).with_restitution(0.7));

        // Directional light
        let light_entity = self.world.spawn(DirectionalLight {
            direction: [0.5, -1.0, 0.3],
            color: [1.0, 0.98, 0.95],
            intensity: 2.0,
        });
        let _ = light_entity;
    }

    fn update_and_render(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();
        let elapsed = self.world.resource::<Time>().unwrap().elapsed as f32;

        // Step physics
        physics_step_system(&mut self.world);

        // Transform propagation
        euca_scene::transform_propagation_system(&mut self.world);

        // Orbit camera
        let cam = self.world.resource_mut::<Camera>().unwrap();
        let angle = elapsed * 0.3;
        let radius = 12.0;
        cam.eye = Vec3::new(angle.cos() * radius, 6.0, angle.sin() * radius);

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

        // Get light
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

        // Render
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();
        renderer.draw(gpu, &camera, &light, &ambient, &draw_commands);
    }
}

impl ApplicationHandler for PhysicsDemoApp {
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
    let mut app = PhysicsDemoApp::new();
    event_loop.run_app(&mut app).unwrap();
}
