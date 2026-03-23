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

const ENTITY_COUNT: u32 = 1000;
const AREA_SIZE: f32 = 50.0;
const FPS_PRINT_INTERVAL: u64 = 60;

/// Simple PCG-based pseudo-random number generator (no external dependency).
struct PcgRng {
    state: u64,
}

impl PcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Returns a float in [0.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Returns a float in [lo, hi).
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

struct StressTestApp {
    world: World,
    survey: HardwareSurvey,
    wgpu_instance: wgpu::Instance,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    window_attrs: WindowAttributes,
    fps_frame_count: u64,
    fps_last_printed: f64,
    current_fps: f32,
}

impl StressTestApp {
    fn new() -> Self {
        let (survey, wgpu_instance) = HardwareSurvey::detect();

        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(Camera::new(
            Vec3::new(25.0, 30.0, 25.0),
            Vec3::new(25.0, 0.0, 25.0),
        ));
        world.insert_resource(PhysicsConfig::new());
        world.insert_resource(AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.3,
        });

        Self {
            world,
            survey,
            wgpu_instance,
            gpu: None,
            renderer: None,
            window_attrs: WindowAttributes::default()
                .with_title(format!(
                    "Euca Engine -- Stress Test ({ENTITY_COUNT} entities) | FPS: --"
                ))
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
            fps_frame_count: 0,
            fps_last_printed: 0.0,
            current_fps: 0.0,
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        let cube_mesh = renderer.upload_mesh(gpu, &Mesh::cube());
        let mat = renderer.upload_material(gpu, &Material::gray());

        let mut rng = PcgRng::new(42);
        let half = AREA_SIZE / 2.0;

        for _ in 0..ENTITY_COUNT {
            let x = rng.range(-half, half);
            let y = rng.range(0.0, 10.0);
            let z = rng.range(-half, half);

            let vx = rng.range(-0.5, 0.5);
            let vy = rng.range(-0.2, 0.2);
            let vz = rng.range(-0.5, 0.5);

            let e = self
                .world
                .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                    x, y, z,
                ))));
            self.world.insert(e, GlobalTransform::default());
            self.world.insert(e, MeshRenderer { mesh: cube_mesh });
            self.world.insert(e, MaterialRef { handle: mat });
            self.world.insert(e, PhysicsBody::dynamic());
            self.world.insert(
                e,
                Velocity {
                    linear: Vec3::new(vx, vy, vz),
                    angular: Vec3::ZERO,
                },
            );
            self.world.insert(e, Collider::sphere(0.3));
        }

        // Ground plane (static)
        let plane_mesh = renderer.upload_mesh(gpu, &Mesh::plane(AREA_SIZE * 2.0));
        let ground_mat = renderer.upload_material(gpu, &Material::green());
        let ground = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                0.0, -0.5, 0.0,
            ))));
        self.world.insert(ground, GlobalTransform::default());
        self.world.insert(ground, MeshRenderer { mesh: plane_mesh });
        self.world
            .insert(ground, MaterialRef { handle: ground_mat });
        self.world.insert(ground, PhysicsBody::fixed());
        self.world
            .insert(ground, Collider::aabb(AREA_SIZE, 0.01, AREA_SIZE));

        // Directional light
        self.world.spawn(DirectionalLight {
            direction: [0.4, -1.0, 0.3],
            color: [1.0, 0.98, 0.95],
            intensity: 2.0,
        });
    }

    fn update_and_render(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();
        let time = self.world.resource::<Time>().unwrap();
        let elapsed = time.elapsed;
        let frame_count = time.frame_count;

        // FPS tracking: print every FPS_PRINT_INTERVAL frames
        self.fps_frame_count += 1;
        if self.fps_frame_count >= FPS_PRINT_INTERVAL {
            let dt = elapsed - self.fps_last_printed;
            if dt > 0.0 {
                self.current_fps = self.fps_frame_count as f32 / dt as f32;
                println!(
                    "[frame {}] {ENTITY_COUNT} entities | FPS: {:.1}",
                    frame_count, self.current_fps
                );
            }
            self.fps_frame_count = 0;
            self.fps_last_printed = elapsed;

            // Update window title
            if let Some(gpu) = &self.gpu {
                gpu.window.set_title(&format!(
                    "Euca Engine -- Stress Test ({ENTITY_COUNT} entities) | FPS: {:.0}",
                    self.current_fps
                ));
            }
        }

        // Step physics
        physics_step_system(&mut self.world);

        // Transform propagation
        euca_scene::transform_propagation_system(&mut self.world);

        // Orbit camera
        let elapsed_f32 = elapsed as f32;
        let cam = self.world.resource_mut::<Camera>().unwrap();
        let angle = elapsed_f32 * 0.15;
        let radius = 60.0;
        cam.eye = Vec3::new(angle.cos() * radius, 30.0, angle.sin() * radius);
        cam.target = Vec3::new(0.0, 0.0, 0.0);

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

impl ApplicationHandler for StressTestApp {
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
    let mut app = StressTestApp::new();
    event_loop.run_app(&mut app).unwrap();
}
