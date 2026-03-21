use euca_asset::load_gltf;
use euca_core::Time;
use euca_ecs::{Query, World};
use euca_math::{Transform, Vec3};
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{WindowAttributes, WindowId};

struct GltfViewerApp {
    world: World,
    survey: HardwareSurvey,
    wgpu_instance: wgpu::Instance,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    gltf_path: String,
    window_attrs: WindowAttributes,
}

impl GltfViewerApp {
    fn new(gltf_path: String) -> Self {
        let (survey, wgpu_instance) = HardwareSurvey::detect();

        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(Camera::new(Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO));
        world.insert_resource(AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.25,
        });

        Self {
            world,
            survey,
            wgpu_instance,
            gpu: None,
            renderer: None,
            gltf_path,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — glTF Viewer")
                .with_inner_size(winit::dpi::LogicalSize::new(1024, 768)),
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        // Load glTF
        let scene = match load_gltf(&self.gltf_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        };

        println!(
            "Loaded {} mesh(es) from {}",
            scene.meshes.len(),
            self.gltf_path
        );

        // Upload and spawn each mesh
        for gltf_mesh in &scene.meshes {
            let mesh_handle = renderer.upload_mesh(gpu, &gltf_mesh.mesh);
            let mat_handle = renderer.upload_material(gpu, &gltf_mesh.material);

            let entity = self.world.spawn(LocalTransform(Transform::IDENTITY));
            self.world.insert(entity, GlobalTransform::default());
            self.world
                .insert(entity, MeshRenderer { mesh: mesh_handle });
            self.world
                .insert(entity, MaterialRef { handle: mat_handle });
        }

        // Add a ground plane
        let plane_mesh = renderer.upload_mesh(gpu, &Mesh::plane(10.0));
        let gray = renderer.upload_material(gpu, &Material::new([0.3, 0.3, 0.3, 1.0], 0.0, 0.9));
        let ground = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                0.0, -0.01, 0.0,
            ))));
        self.world.insert(ground, GlobalTransform::default());
        self.world.insert(ground, MeshRenderer { mesh: plane_mesh });
        self.world.insert(ground, MaterialRef { handle: gray });

        // Light
        self.world.spawn(DirectionalLight {
            direction: [0.4, -1.0, 0.3],
            color: [1.0, 0.98, 0.95],
            intensity: 2.5,
        });
    }

    fn update_and_render(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();
        let elapsed = self.world.resource::<Time>().unwrap().elapsed as f32;

        euca_scene::transform_propagation_system(&mut self.world);

        // Orbit camera
        let cam = self.world.resource_mut::<Camera>().unwrap();
        let angle = elapsed * 0.4;
        let radius = 3.0;
        cam.eye = Vec3::new(angle.cos() * radius, 1.5, angle.sin() * radius);

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

impl ApplicationHandler for GltfViewerApp {
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

    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: gltf_viewer <path-to-model.glb>");
        eprintln!("Example: cargo run -p euca-asset --example gltf_viewer -- assets/helmet.glb");
        std::process::exit(1);
    });

    let event_loop = EventLoop::new().unwrap();
    let mut app = GltfViewerApp::new(path);
    event_loop.run_app(&mut app).unwrap();
}
