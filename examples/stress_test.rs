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

/// Read entity count from EUCA_ENTITIES env var. Default: 1000.
fn entity_count() -> u32 {
    std::env::var("EUCA_ENTITIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000)
}

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
    #[allow(dead_code)]
    extractor: RenderExtractor,
    window_attrs: WindowAttributes,
    fps_frame_count: u64,
    fps_last_printed: f64,
    current_fps: f32,
    num_entities: u32,
}

impl StressTestApp {
    fn new() -> Self {
        let (survey, wgpu_instance) = HardwareSurvey::detect();
        let num_entities = entity_count();
        let area_size = (num_entities as f32).sqrt() * 2.0;

        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(Camera::new(
            Vec3::new(area_size / 2.0, area_size * 0.6, area_size / 2.0),
            Vec3::new(0.0, 0.0, 0.0),
        ));
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO, // Zero gravity: entities drift freely, no pile-up.
            ..PhysicsConfig::new()
        });
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
            extractor: RenderExtractor::new(),
            window_attrs: WindowAttributes::default()
                .with_title(format!(
                    "Euca Engine -- Stress Test ({num_entities} entities) | FPS: --"
                ))
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
            fps_frame_count: 0,
            fps_last_printed: 0.0,
            current_fps: 0.0,
            num_entities,
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        // Enable bindless materials if GPU supports it.
        renderer.enable_bindless(gpu);

        let cube_mesh = renderer.upload_mesh(gpu, &Mesh::cube());

        // Multiple materials to exercise bindless path.
        let materials = [
            renderer.upload_material(gpu, &Material::gray()),
            renderer.upload_material(
                gpu,
                &Material {
                    albedo: [0.8, 0.2, 0.2, 1.0],
                    ..Material::default()
                },
            ),
            renderer.upload_material(
                gpu,
                &Material {
                    albedo: [0.2, 0.6, 0.8, 1.0],
                    ..Material::default()
                },
            ),
            renderer.upload_material(
                gpu,
                &Material {
                    albedo: [0.3, 0.8, 0.3, 1.0],
                    ..Material::default()
                },
            ),
            renderer.upload_material(
                gpu,
                &Material {
                    albedo: [0.9, 0.7, 0.2, 1.0],
                    ..Material::default()
                },
            ),
        ];

        let area_size = (self.num_entities as f32).sqrt() * 2.0;
        let side = (self.num_entities as f32).sqrt().ceil() as u32;
        let spacing = 2.5_f32;
        let mut rng = PcgRng::new(42);

        for i in 0..self.num_entities {
            // Grid placement: entities rest on ground, spaced to avoid pile-up.
            let row = i / side;
            let col = i % side;
            let x = (col as f32 - side as f32 / 2.0) * spacing;
            let y = 0.5; // Cube center at 0.5 → bottom face at Y=0.0, above ground (Y=-0.5)
            let z = (row as f32 - side as f32 / 2.0) * spacing;

            // Horizontal-only velocity — no vertical drop that causes pile-up.
            let vx = rng.range(-0.2, 0.2);
            let vy = 0.0;
            let vz = rng.range(-0.2, 0.2);

            let mat = materials[(i as usize) % materials.len()];

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
        let plane_mesh = renderer.upload_mesh(gpu, &Mesh::plane(area_size * 2.0));
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
            .insert(ground, Collider::aabb(area_size, 0.01, area_size));

        // Directional light
        self.world.spawn(DirectionalLight {
            direction: [0.4, -1.0, 0.3],
            color: [1.0, 0.98, 0.95],
            intensity: 2.0,
            ..Default::default()
        });

        if renderer.is_bindless() {
            println!("[stress_test] Bindless materials ENABLED");
        } else {
            println!("[stress_test] Bindless materials not available — using traditional path");
        }
        println!(
            "[stress_test] {} entities, area {:.0}×{:.0}",
            self.num_entities, area_size, area_size
        );
    }

    fn update_and_render(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();
        let time = self.world.resource::<Time>().unwrap();
        let elapsed = time.elapsed;
        let frame_count = time.frame_count;

        // FPS tracking
        self.fps_frame_count += 1;
        if self.fps_frame_count >= FPS_PRINT_INTERVAL {
            let dt = elapsed - self.fps_last_printed;
            if dt > 0.0 {
                self.current_fps = self.fps_frame_count as f32 / dt as f32;
                println!(
                    "[frame {}] {} entities | FPS: {:.1}",
                    frame_count, self.num_entities, self.current_fps
                );
            }
            self.fps_frame_count = 0;
            self.fps_last_printed = elapsed;

            if let Some(gpu) = &self.gpu {
                gpu.window.set_title(&format!(
                    "Euca Engine -- Stress Test ({} entities) | FPS: {:.0}",
                    self.num_entities, self.current_fps
                ));
            }
        }

        // Step physics (disable for render-only profiling with EUCA_NO_PHYSICS=1)
        if std::env::var("EUCA_NO_PHYSICS").is_err() {
            physics_step_system(&mut self.world);
        }

        // Transform propagation
        euca_scene::transform_propagation_system(&mut self.world);

        // Orbit camera
        let area_size = (self.num_entities as f32).sqrt() * 2.0;
        let elapsed_f32 = elapsed as f32;
        let cam = self.world.resource_mut::<Camera>().unwrap();
        let angle = elapsed_f32 * 0.15;
        let radius = area_size * 1.2;
        cam.eye = Vec3::new(angle.cos() * radius, area_size * 0.6, angle.sin() * radius);
        cam.target = Vec3::new(0.0, 0.0, 0.0);

        // Direct extraction (bypassing RenderExtractor for debugging).
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

        // Advance ECS tick for change detection.
        self.world.tick();
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
