use euca_core::Time;
use euca_ecs::{Query, World};
use euca_math::{Mat4, Quat, Vec3};
use euca_particle::{
    EmitterConfig, EmitterShape, ParticleEmitter, emit_particles_system, particle_update_system,
    render::ParticleBlendMode,
};
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{WindowAttributes, WindowId};

struct ParticleDemoApp {
    world: World,
    survey: HardwareSurvey,
    wgpu_instance: wgpu::Instance,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    /// Small sphere mesh used to visualize each particle.
    particle_mesh: Option<MeshHandle>,
    /// Material per emitter (indexed by spawn order).
    emitter_materials: Vec<MaterialHandle>,
    window_attrs: WindowAttributes,
}

impl ParticleDemoApp {
    fn new() -> Self {
        let (survey, wgpu_instance) = HardwareSurvey::detect();

        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(Camera::new(
            Vec3::new(6.0, 4.0, 6.0),
            Vec3::new(0.0, 1.0, 0.0),
        ));
        world.insert_resource(AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.4,
        });

        Self {
            world,
            survey,
            wgpu_instance,
            gpu: None,
            renderer: None,
            particle_mesh: None,
            emitter_materials: Vec::new(),
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — CPU Particle Demo")
                .with_inner_size(winit::dpi::LogicalSize::new(1024, 768)),
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        let particle_mesh = renderer.upload_mesh(gpu, &Mesh::sphere(0.5, 6, 12));
        self.particle_mesh = Some(particle_mesh);

        // Ground plane for spatial reference.
        let plane_mesh = renderer.upload_mesh(gpu, &Mesh::plane(20.0));
        let ground_mat = renderer.upload_material(gpu, &Material::gray());
        let ground = self
            .world
            .spawn(LocalTransform(euca_math::Transform::from_translation(
                Vec3::ZERO,
            )));
        self.world.insert(ground, GlobalTransform::default());
        self.world.insert(ground, MeshRenderer { mesh: plane_mesh });
        self.world
            .insert(ground, MaterialRef { handle: ground_mat });

        // Materials for each emitter type.
        let fire_mat =
            renderer.upload_material(gpu, &Material::new([1.0, 0.4, 0.05, 1.0], 0.0, 0.9));
        let smoke_mat =
            renderer.upload_material(gpu, &Material::new([0.45, 0.45, 0.45, 1.0], 0.0, 1.0));
        let sparkle_mat =
            renderer.upload_material(gpu, &Material::new([1.0, 0.95, 0.6, 1.0], 0.8, 0.2));

        // ── Fire emitter (Point shape, upward, orange-red, short lifetime) ──
        let fire_emitter = ParticleEmitter::new(EmitterConfig {
            rate: 80.0,
            particle_lifetime: 1.2,
            speed_range: [2.0, 4.0],
            size_range: [0.08, 0.18],
            color_start: [1.0, 0.6, 0.1, 1.0],
            color_end: [1.0, 0.1, 0.0, 0.0],
            shape: EmitterShape::Point,
            max_particles: 500,
            gravity: [0.0, 1.5, 0.0], // Upward drift (fire rises)
        });
        self.spawn_emitter(Vec3::new(-3.0, 0.5, 0.0), fire_emitter, fire_mat);

        // ── Smoke emitter (Sphere shape, slow rise, gray, longer lifetime) ──
        let smoke_emitter = ParticleEmitter::new(EmitterConfig {
            rate: 30.0,
            particle_lifetime: 4.0,
            speed_range: [0.3, 1.0],
            size_range: [0.15, 0.35],
            color_start: [0.5, 0.5, 0.5, 0.8],
            color_end: [0.3, 0.3, 0.3, 0.0],
            shape: EmitterShape::Sphere { radius: 0.5 },
            max_particles: 400,
            gravity: [0.0, 0.4, 0.0], // Gentle upward drift
        });
        self.spawn_emitter(Vec3::new(0.0, 0.5, 0.0), smoke_emitter, smoke_mat);

        // ── Sparkle emitter (Cone shape, burst emission, yellow-white) ──
        let mut sparkle_emitter = ParticleEmitter::new(EmitterConfig {
            rate: 120.0,
            particle_lifetime: 1.5,
            speed_range: [3.0, 7.0],
            size_range: [0.04, 0.10],
            color_start: [1.0, 1.0, 0.8, 1.0],
            color_end: [1.0, 0.85, 0.3, 0.0],
            shape: EmitterShape::Cone { angle: 45.0 },
            max_particles: 600,
            gravity: [0.0, -4.0, 0.0], // Sparkles fall after launch
        });
        sparkle_emitter.blend_mode = ParticleBlendMode::Additive;
        self.spawn_emitter(Vec3::new(3.0, 0.5, 0.0), sparkle_emitter, sparkle_mat);

        // Directional light for the scene.
        self.world.spawn(DirectionalLight {
            direction: [0.4, -1.0, 0.3],
            color: [1.0, 0.98, 0.95],
            intensity: 2.0,
            ..Default::default()
        });
    }

    /// Spawn a particle emitter entity and record its visual material.
    fn spawn_emitter(
        &mut self,
        position: Vec3,
        emitter: ParticleEmitter,
        material: MaterialHandle,
    ) {
        let entity = self.world.spawn(emitter);
        self.world.insert(
            entity,
            LocalTransform(euca_math::Transform::from_translation(position)),
        );
        self.world.insert(entity, GlobalTransform::default());
        self.emitter_materials.push(material);
    }

    fn update_and_render(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();
        let time = self.world.resource::<Time>().unwrap();
        let dt = time.delta;
        let elapsed = time.elapsed as f32;

        // ── Run CPU particle systems ──
        emit_particles_system(&mut self.world, dt);
        particle_update_system(&mut self.world, dt);

        // ── Transform propagation ──
        euca_scene::transform_propagation_system(&mut self.world);

        // ── Orbiting camera ──
        let cam = self.world.resource_mut::<Camera>().unwrap();
        let angle = elapsed * 0.25;
        let radius = 10.0;
        cam.eye = Vec3::new(angle.cos() * radius, 5.0, angle.sin() * radius);

        // ── Collect particle data for visualization ──
        let particle_mesh = match self.particle_mesh {
            Some(h) => h,
            None => return,
        };

        // Build per-emitter draw commands from live particles.
        let mut draw_commands = Vec::new();

        // Ground plane and any other mesh-rendered entities.
        {
            let query = Query::<(&GlobalTransform, &MeshRenderer, &MaterialRef)>::new(&self.world);
            for (gt, mr, mat) in query.iter() {
                draw_commands.push(DrawCommand {
                    mesh: mr.mesh,
                    material: mat.handle,
                    model_matrix: gt.0.to_matrix(),
                    aabb: None,
                });
            }
        }

        // Particle visualization: one draw per particle using its emitter's material.
        {
            let query = Query::<&ParticleEmitter>::new(&self.world);
            for (idx, emitter) in query.iter().enumerate() {
                let material = match self.emitter_materials.get(idx) {
                    Some(&m) => m,
                    None => continue,
                };
                for particle in &emitter.particles {
                    let scale = Vec3::new(particle.size, particle.size, particle.size);
                    let model = Mat4::from_scale_rotation_translation(
                        scale,
                        Quat::IDENTITY,
                        particle.position,
                    );
                    draw_commands.push(DrawCommand {
                        mesh: particle_mesh,
                        material,
                        model_matrix: model,
                        aabb: None,
                    });
                }
            }
        }

        // ── Render ──
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
        let renderer = self.renderer.as_mut().unwrap();
        renderer.draw(gpu, &camera, &light, &ambient, &draw_commands);
    }
}

impl ApplicationHandler for ParticleDemoApp {
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
    let mut app = ParticleDemoApp::new();
    event_loop.run_app(&mut app).unwrap();
}
