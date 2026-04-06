//! Tiled / LDtk level importer demo.
//!
//! Loads a Tiled (.tmj) or LDtk (.ldtk) map file, converts it to engine
//! LevelData, generates terrain mesh chunks, and renders the result.
//!
//! Run:
//!   cargo run -p euca-game --example tiled_level --release
//!   cargo run -p euca-game --example tiled_level --release -- assets/maps/demo.ldtk

use std::path::Path;

use euca_core::Time;
use euca_ecs::{Query, World};
use euca_math::{Quat, Vec3};
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};

#[cfg(all(target_os = "macos", feature = "metal-native"))]
type Dev = euca_render::euca_rhi::metal_backend::MetalDevice;
#[cfg(not(all(target_os = "macos", feature = "metal-native")))]
type Dev = euca_render::euca_rhi::wgpu_backend::WgpuDevice;
use euca_terrain::level_data::{LevelData, SurfaceType};
use euca_terrain::level_render::{generate_mesh_from_level, surface_color};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{WindowAttributes, WindowId};

const DEFAULT_MAP: &str = "assets/maps/demo.tmj";

fn load_level_from_path(path: &str) -> LevelData {
    let p = Path::new(path);
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "ldtk" => {
            log::info!("Loading LDtk map: {path}");
            euca_terrain::ldtk_import::load_ldtk_json(p, 1.0)
                .unwrap_or_else(|e| panic!("Failed to load LDtk map: {e}"))
        }
        _ => {
            log::info!("Loading Tiled map: {path}");
            euca_terrain::tiled_import::load_tiled_json(p, 1.0)
                .unwrap_or_else(|e| panic!("Failed to load Tiled map: {e}"))
        }
    }
}

struct TiledLevelApp {
    world: World,
    level: LevelData,
    gpu: Option<GpuContext<Dev>>,
    renderer: Option<Renderer<Dev>>,
    window_attrs: WindowAttributes,
}

impl TiledLevelApp {
    fn new(map_path: &str) -> Self {
        let level = load_level_from_path(map_path);
        log::info!(
            "Level loaded: {}x{}, {} entities, {} triggers",
            level.width,
            level.height,
            level.entities.len(),
            level.triggers.len(),
        );

        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.3,
        });

        // Camera: top-down isometric looking at center of the map
        let center_x = level.width as f32 * level.cell_size * 0.5;
        let center_z = level.height as f32 * level.cell_size * 0.5;
        let cam_dist = (level.width.max(level.height) as f32) * level.cell_size * 0.8;
        world.insert_resource(Camera::new(
            Vec3::new(
                center_x + cam_dist * 0.5,
                cam_dist * 0.7,
                center_z + cam_dist * 0.5,
            ),
            Vec3::new(center_x, 0.0, center_z),
        ));

        Self {
            world,
            level,
            gpu: None,
            renderer: None,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — Tiled/LDtk Level Viewer")
                .with_inner_size(winit::dpi::LogicalSize::new(1024, 768)),
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        // Generate terrain mesh chunks from the level data
        let chunks = generate_mesh_from_level(&self.level);
        log::info!("Generated {} terrain chunks", chunks.len());

        for chunk in &chunks {
            let color = surface_color(chunk.surface);
            let mat = Material {
                albedo: color,
                metallic: 0.0,
                roughness: 0.8,
                ..Material::default()
            };

            let mesh_handle = renderer.upload_mesh(gpu, &chunk.mesh);
            let mat_handle = renderer.upload_material(gpu, &mat);

            let e = self
                .world
                .spawn(LocalTransform(euca_math::Transform::IDENTITY));
            self.world.insert(e, GlobalTransform::default());
            self.world.insert(e, MeshRenderer { mesh: mesh_handle });
            self.world.insert(e, MaterialRef { handle: mat_handle });
        }

        // Entity markers: place small colored cubes at each entity position
        let marker_mesh = renderer.upload_mesh(gpu, &Mesh::cube());
        let marker_mat = renderer.upload_material(
            gpu,
            &Material {
                albedo: [1.0, 0.3, 0.1, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                ..Material::default()
            },
        );

        for entity in &self.level.entities {
            let pos = Vec3::new(entity.position.x, 0.5, entity.position.z);
            let transform = euca_math::Transform {
                translation: pos,
                rotation: Quat::IDENTITY,
                scale: Vec3::new(0.3, 0.3, 0.3),
            };
            let e = self.world.spawn(LocalTransform(transform));
            self.world.insert(e, GlobalTransform::default());
            self.world.insert(e, MeshRenderer { mesh: marker_mesh });
            self.world.insert(e, MaterialRef { handle: marker_mat });
        }

        log::info!("Placed {} entity markers", self.level.entities.len());

        // Foliage: scatter grass blades on cells with SurfaceType::Grass
        scatter_grass(&self.level, &mut self.world, gpu, renderer);

        // Directional light
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

        // Slow orbit camera
        let center_x = self.level.width as f32 * self.level.cell_size * 0.5;
        let center_z = self.level.height as f32 * self.level.cell_size * 0.5;
        let radius = (self.level.width.max(self.level.height) as f32) * self.level.cell_size * 0.8;
        let angle = elapsed * 0.1;
        let cam = self.world.resource_mut::<Camera>().unwrap();
        cam.eye = Vec3::new(
            center_x + angle.cos() * radius * 0.5,
            radius * 0.7,
            center_z + angle.sin() * radius * 0.5,
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

impl ApplicationHandler for TiledLevelApp {
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

/// Scatter procedural grass blade entities on all cells with [`SurfaceType::Grass`].
///
/// Uses [`scatter_foliage`] (Poisson-disk sampling) over the full map area,
/// then retains only those instances whose XZ position falls on a grass cell.
/// Each surviving instance becomes an ECS entity with a thin, tall cube mesh
/// and a green material.
fn scatter_grass(
    level: &LevelData,
    world: &mut World,
    gpu: &GpuContext<Dev>,
    renderer: &mut Renderer<Dev>,
) {
    let cell = level.cell_size;
    let area_min = Vec3::ZERO;
    let area_max = Vec3::new(level.width as f32 * cell, 0.0, level.height as f32 * cell);

    let grass_mesh = renderer.upload_mesh(gpu, &Mesh::cube());
    let grass_mat = renderer.upload_material(
        gpu,
        &Material {
            albedo: [0.18, 0.52, 0.10, 1.0],
            metallic: 0.0,
            roughness: 0.85,
            ..Material::default()
        },
    );

    // Fixed seed ensures deterministic placement across runs.
    let mut layer = FoliageLayer {
        mesh: grass_mesh,
        material: grass_mat,
        density: 2.0, // instances per square unit
        min_scale: 0.08,
        max_scale: 0.18,
        max_distance: 200.0,
        instances: Vec::new(),
    };
    scatter_foliage(&mut layer, area_min, area_max, 42);

    layer.instances.retain(|inst| {
        let col = (inst.position.x / cell) as u32;
        let row = (inst.position.z / cell) as u32;
        level.surface_at(col, row) == SurfaceType::Grass
    });

    for inst in &layer.instances {
        let transform = euca_math::Transform {
            translation: inst.position,
            rotation: Quat::from_axis_angle(Vec3::Y, inst.rotation),
            // Thin in X/Z, taller in Y to look like a grass blade.
            scale: Vec3::new(inst.scale * 0.3, inst.scale * 2.5, inst.scale * 0.3),
        };
        let e = world.spawn(LocalTransform(transform));
        world.insert(e, GlobalTransform::default());
        world.insert(e, MeshRenderer { mesh: grass_mesh });
        world.insert(e, MaterialRef { handle: grass_mat });
    }

    log::info!(
        "Scattered {} grass blade instances on grass cells",
        layer.instances.len(),
    );
}

fn main() {
    env_logger::init();
    let map_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MAP.to_string());
    let event_loop = EventLoop::new().unwrap();
    let mut app = TiledLevelApp::new(&map_path);
    event_loop.run_app(&mut app).unwrap();
}
