//! GPU terrain generation demo.
//!
//! Creates a procedural heightmap (sine-wave hills), subdivides it into chunks
//! with LOD selection, generates each chunk's mesh on the GPU via compute
//! shader, and renders the result.
//!
//! Run:
//!   cargo run -p euca-game --example gpu_terrain_demo --features gpu-terrain --release

use euca_core::Time;
use euca_ecs::{Query, World};
use euca_math::Vec3;
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};
use euca_terrain::{
    GpuTerrainGenerator, Heightmap, LodConfig, TerrainGenParams, build_chunks, select_chunk_lod,
};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{WindowAttributes, WindowId};

#[cfg(all(target_os = "macos", feature = "metal-native"))]
type Dev = euca_render::euca_rhi::metal_backend::MetalDevice;
#[cfg(not(all(target_os = "macos", feature = "metal-native")))]
type Dev = euca_render::euca_rhi::wgpu_backend::WgpuDevice;

/// Grid dimensions for the heightmap.
const HEIGHTMAP_SIZE: u32 = 128;
/// Number of grid cells per chunk side.
const CHUNK_SIZE: u32 = 32;
/// Height scale applied to the normalised [0,1] heightmap values.
const HEIGHT_SCALE: f32 = 30.0;
/// World-space distance between adjacent grid vertices.
const CELL_SIZE: f32 = 1.0;

/// Generate a procedural heightmap with overlapping sine waves.
fn generate_sine_heightmap(width: u32, height: u32) -> Heightmap {
    let mut data = vec![0.0f32; (width as usize) * (height as usize)];
    for row in 0..height {
        for col in 0..width {
            let x = col as f32 / width as f32;
            let z = row as f32 / height as f32;
            // Overlapping sine waves at different frequencies for natural-looking hills.
            let h = 0.5
                + 0.25 * (x * std::f32::consts::TAU * 2.0).sin()
                    * (z * std::f32::consts::TAU * 2.0).cos()
                + 0.15 * (x * std::f32::consts::TAU * 5.0 + 1.0).sin()
                + 0.10 * (z * std::f32::consts::TAU * 3.0 + 2.0).cos();
            data[(row * width + col) as usize] = h.clamp(0.0, 1.0);
        }
    }
    Heightmap::from_raw(width, height, data)
        .with_cell_size(CELL_SIZE)
        .with_max_height(HEIGHT_SCALE)
}

struct GpuTerrainApp {
    world: World,
    heightmap: Heightmap,
    gpu: Option<GpuContext<Dev>>,
    renderer: Option<Renderer<Dev>>,
    window_attrs: WindowAttributes,
}

impl GpuTerrainApp {
    fn new() -> Self {
        let heightmap = generate_sine_heightmap(HEIGHTMAP_SIZE, HEIGHTMAP_SIZE);

        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.3,
        });

        // Camera overlooking the terrain.
        let center = HEIGHTMAP_SIZE as f32 * CELL_SIZE * 0.5;
        world.insert_resource(Camera::new(
            Vec3::new(center + 60.0, 80.0, center + 60.0),
            Vec3::new(center, 0.0, center),
        ));

        Self {
            world,
            heightmap,
            gpu: None,
            renderer: None,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — GPU Terrain Demo")
                .with_inner_size(winit::dpi::LogicalSize::new(1024, 768)),
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();
        let rhi: &Dev = gpu;

        // Create the GPU terrain generator from the heightmap data.
        let terrain_gen = GpuTerrainGenerator::<Dev>::new(
            rhi,
            &self.heightmap.data,
            self.heightmap.width,
            self.heightmap.height,
        );

        // Subdivide the heightmap into chunks and select LOD for each.
        let chunks = build_chunks(&self.heightmap, CHUNK_SIZE);
        let camera = self.world.resource::<Camera>().unwrap().clone();
        let lod_config = LodConfig::default();

        // Create a shared material for all terrain chunks.
        let terrain_mat = renderer.upload_material(
            gpu,
            &Material {
                albedo: [0.45, 0.55, 0.30, 1.0],
                metallic: 0.0,
                roughness: 0.9,
                ..Material::default()
            },
        );

        // Generate each chunk on the GPU and register the resulting mesh.
        let mut encoder = rhi.create_command_encoder(Some("Terrain Gen"));

        for chunk in &chunks {
            let lod = select_chunk_lod(chunk, camera.eye, &lod_config);
            let grid_cols = (chunk.col_end - chunk.col_start).div_ceil(lod.step);
            let grid_rows = (chunk.row_end - chunk.row_start).div_ceil(lod.step);

            let params = TerrainGenParams {
                grid_cols,
                grid_rows,
                cell_size: self.heightmap.cell_size * lod.step as f32,
                step: lod.step,
                origin_x: chunk.col_start as f32 * self.heightmap.cell_size,
                origin_z: chunk.row_start as f32 * self.heightmap.cell_size,
                heightmap_width: self.heightmap.width,
                heightmap_height: self.heightmap.height,
                height_scale: self.heightmap.max_height,
                _pad: [0.0; 3],
            };

            let output = terrain_gen.generate_chunk(rhi, &mut encoder, &params);

            let mesh_handle = renderer.register_gpu_mesh(
                output.vertex_buffer,
                output.vertex_buffer_size,
                output.index_buffer,
                output.index_buffer_size,
                output.index_count,
            );

            let e = self
                .world
                .spawn(LocalTransform(euca_math::Transform::IDENTITY));
            self.world.insert(e, GlobalTransform::default());
            self.world.insert(e, MeshRenderer { mesh: mesh_handle });
            self.world
                .insert(e, MaterialRef { handle: terrain_mat });
        }

        rhi.submit(encoder);
        log::info!(
            "Generated {} terrain chunks on GPU ({} vertices per chunk at LOD 0)",
            chunks.len(),
            CHUNK_SIZE * CHUNK_SIZE,
        );

        // Directional light.
        let dir = Vec3::new(-0.5, -1.0, -0.3).normalize();
        self.world.spawn(DirectionalLight {
            direction: [dir.x, dir.y, dir.z],
            color: [1.0, 0.98, 0.95],
            intensity: 1.2,
            light_size: 1.0,
        });
    }

    fn update_and_render(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();
        let elapsed = self.world.resource::<Time>().unwrap().elapsed as f32;

        // Slow orbit camera around the terrain center.
        let center = HEIGHTMAP_SIZE as f32 * CELL_SIZE * 0.5;
        let radius = HEIGHTMAP_SIZE as f32 * CELL_SIZE * 0.6;
        let angle = elapsed * 0.15;
        let cam = self.world.resource_mut::<Camera>().unwrap();
        cam.eye = Vec3::new(
            center + angle.cos() * radius,
            radius * 0.7,
            center + angle.sin() * radius,
        );

        euca_scene::transform_propagation_system(&mut self.world);

        let draw_commands: Vec<DrawCommand> = {
            let query = Query::<(&GlobalTransform, &MeshRenderer, &MaterialRef)>::new(&self.world);
            query
                .iter()
                .map(|(gt, mr, mat)| DrawCommand {
                    mesh: mr.mesh,
                    material: mat.handle,
                    model_matrix: gt.0.to_matrix(),
                    aabb: None,
                    is_water: false,
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
        let renderer = self.renderer.as_mut().unwrap();
        renderer.draw(gpu, &camera, &light, &ambient, &draw_commands);
    }
}

impl ApplicationHandler for GpuTerrainApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let window = event_loop.create_window(self.window_attrs.clone()).unwrap();

            #[cfg(all(target_os = "macos", feature = "metal-native"))]
            let gpu = GpuContext::new_metal(std::sync::Arc::new(window));
            #[cfg(not(all(target_os = "macos", feature = "metal-native")))]
            let gpu = {
                let (survey, wgpu_instance) = HardwareSurvey::detect();
                GpuContext::new(window, &survey, &wgpu_instance)
            };

            let renderer = Renderer::new(&gpu);
            self.gpu = Some(gpu);
            self.renderer = Some(renderer);
            self.setup_scene();
            self.gpu.as_ref().unwrap().window.request_redraw();
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
    let mut app = GpuTerrainApp::new();
    event_loop.run_app(&mut app).unwrap();
}
