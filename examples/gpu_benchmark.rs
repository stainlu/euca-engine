//! Industrial-level GPU benchmark: Metal vs wgpu on Apple Silicon.
//!
//! Measures frame time with statistical rigor: percentiles, stdev, min/max.
//! Deterministic camera path, configurable entity count, warmup period.
//!
//! Environment variables:
//!   BENCH_ENTITIES   — number of cubes to spawn (default: 1000)
//!   BENCH_WARMUP     — warmup frames to discard (default: 60)
//!   BENCH_FRAMES     — frames to measure (default: 600)
//!   BENCH_CSV        — output CSV file (default: stdout)
//!
//! Run:
//!   cargo run -p euca-game --example gpu_benchmark --release
//!   cargo run -p euca-game --example gpu_benchmark --features metal-native --release

use std::time::Instant;

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

// Backend type alias
#[cfg(all(target_os = "macos", feature = "metal-native"))]
type Dev = euca_render::euca_rhi::metal_backend::MetalDevice;
#[cfg(not(all(target_os = "macos", feature = "metal-native")))]
type Dev = euca_render::euca_rhi::wgpu_backend::WgpuDevice;

fn backend_name() -> &'static str {
    #[cfg(all(target_os = "macos", feature = "metal-native"))]
    {
        "metal"
    }
    #[cfg(not(all(target_os = "macos", feature = "metal-native")))]
    {
        "wgpu"
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

struct Stats {
    min: f64,
    max: f64,
    avg: f64,
    median: f64,
    stdev: f64,
    p1: f64,
    p01: f64,
    p99: f64,
    count: usize,
}

fn compute_stats(times: &[f64]) -> Stats {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let sum: f64 = sorted.iter().sum();
    let avg = sum / n as f64;
    let variance: f64 = sorted.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / n as f64;

    Stats {
        min: sorted[0],
        max: sorted[n - 1],
        avg,
        median: sorted[n / 2],
        stdev: variance.sqrt(),
        p1: sorted[(n as f64 * 0.99) as usize],
        p01: sorted[(n as f64 * 0.999).min((n - 1) as f64) as usize],
        p99: sorted[(n as f64 * 0.01) as usize],
        count: n,
    }
}

// ---------------------------------------------------------------------------
// Benchmark App
// ---------------------------------------------------------------------------

struct BenchApp {
    world: World,
    gpu: Option<GpuContext<Dev>>,
    renderer: Option<Renderer<Dev>>,
    window_attrs: WindowAttributes,

    entity_count: usize,
    warmup_frames: usize,
    measure_frames: usize,

    frame_index: usize,
    frame_times_ms: Vec<f64>,
    last_frame: Instant,
    done: bool,
}

impl BenchApp {
    fn new() -> Self {
        let entity_count = env_usize("BENCH_ENTITIES", 1000);
        let warmup_frames = env_usize("BENCH_WARMUP", 60);
        let measure_frames = env_usize("BENCH_FRAMES", 600);

        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(Camera::new(Vec3::new(10.0, 10.0, 10.0), Vec3::ZERO));
        world.insert_resource(AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.3,
        });

        // Disable post-process for raw GPU throughput measurement
        let mut pps = PostProcessSettings::default();
        pps.ssao_enabled = false;
        pps.fxaa_enabled = false;
        world.insert_resource(pps);

        eprintln!("=== Euca Engine GPU Benchmark ===");
        eprintln!("Backend:  {}", backend_name());
        eprintln!("Entities: {entity_count}");
        eprintln!("Warmup:   {warmup_frames} frames");
        eprintln!("Measure:  {measure_frames} frames");
        eprintln!("=================================");

        Self {
            world,
            gpu: None,
            renderer: None,
            window_attrs: WindowAttributes::default()
                .with_title(format!(
                    "Euca Benchmark — {} — {} entities",
                    backend_name(),
                    entity_count
                ))
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
            entity_count,
            warmup_frames,
            measure_frames,
            frame_index: 0,
            frame_times_ms: Vec::with_capacity(measure_frames),
            last_frame: Instant::now(),
            done: false,
        }
    }

    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        let cube = renderer.upload_mesh(gpu, &Mesh::cube());
        let materials = [
            renderer.upload_material(gpu, &Material::red_plastic()),
            renderer.upload_material(gpu, &Material::green()),
            renderer.upload_material(gpu, &Material::blue_plastic()),
            renderer.upload_material(
                gpu,
                &Material {
                    albedo: [1.0, 0.8, 0.2, 1.0],
                    metallic: 0.8,
                    roughness: 0.3,
                    ..Material::default()
                },
            ),
        ];

        // Spawn entities in a grid
        let side = (self.entity_count as f64).cbrt().ceil() as usize;
        let spacing = 2.5;
        let offset = side as f32 * spacing / 2.0;
        let mut spawned = 0;

        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    if spawned >= self.entity_count {
                        break;
                    }
                    let pos = Vec3::new(
                        x as f32 * spacing - offset,
                        y as f32 * spacing - offset,
                        z as f32 * spacing - offset,
                    );
                    let e = self.world.spawn(LocalTransform(Transform {
                        translation: pos,
                        rotation: Quat::IDENTITY,
                        scale: Vec3::new(0.8, 0.8, 0.8),
                    }));
                    self.world.insert(e, GlobalTransform::default());
                    self.world.insert(e, MeshRenderer { mesh: cube });
                    self.world.insert(
                        e,
                        MaterialRef {
                            handle: materials[spawned % materials.len()],
                        },
                    );
                    spawned += 1;
                }
            }
        }

        // Directional light
        self.world.spawn(DirectionalLight::default());

        eprintln!("Scene: {spawned} cubes spawned in {side}x{side}x{side} grid");
    }

    fn update_and_render(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame);
        self.last_frame = now;

        self.frame_index += 1;

        // Record frame time (skip warmup)
        if self.frame_index > self.warmup_frames {
            self.frame_times_ms.push(dt.as_secs_f64() * 1000.0);

            if self.frame_times_ms.len() >= self.measure_frames {
                self.report_results();
                self.done = true;
                return;
            }
        }

        // Progress indicator
        if self.frame_index % 100 == 0 {
            if self.frame_index <= self.warmup_frames {
                eprintln!("  Warmup: {}/{}", self.frame_index, self.warmup_frames);
            } else {
                eprintln!(
                    "  Measuring: {}/{}",
                    self.frame_times_ms.len(),
                    self.measure_frames
                );
            }
        }

        // Deterministic camera orbit
        self.world.resource_mut::<Time>().unwrap().update();
        let elapsed = self.world.resource::<Time>().unwrap().elapsed as f32;
        let radius = (self.entity_count as f32).cbrt() * 3.0;
        let angle = elapsed * 0.5;
        let cam = self.world.resource_mut::<Camera>().unwrap();
        cam.eye = Vec3::new(angle.cos() * radius, radius * 0.6, angle.sin() * radius);
        cam.target = Vec3::ZERO;

        // Rotate cubes
        let updates: Vec<(Entity, f32)> = {
            let q = Query::<(Entity, &MaterialRef)>::new(&self.world);
            q.iter().map(|(e, _)| (e, elapsed)).collect()
        };
        for (i, (entity, t)) in updates.iter().enumerate() {
            if let Some(lt) = self.world.get_mut::<LocalTransform>(*entity) {
                let speed = (i % 5) as f32 * 0.3 + 0.5;
                lt.0.rotation = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), *t * speed);
            }
        }

        euca_scene::transform_propagation_system(&mut self.world);

        // Collect and render
        let draw_commands: Vec<DrawCommand> = {
            let q = Query::<(&GlobalTransform, &MeshRenderer, &MaterialRef)>::new(&self.world);
            q.iter()
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
            let q = Query::<&DirectionalLight>::new(&self.world);
            q.iter().next().cloned().unwrap_or_default()
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

    fn report_results(&self) {
        let s = compute_stats(&self.frame_times_ms);

        eprintln!();
        eprintln!("=== BENCHMARK RESULTS ===");
        eprintln!("Backend:    {}", backend_name());
        eprintln!("Entities:   {}", self.entity_count);
        eprintln!("Frames:     {}", s.count);
        eprintln!("Resolution: 1280x720");
        eprintln!("─────────────────────────");
        eprintln!("Avg:        {:.2} ms ({:.0} FPS)", s.avg, 1000.0 / s.avg);
        eprintln!(
            "Median:     {:.2} ms ({:.0} FPS)",
            s.median,
            1000.0 / s.median
        );
        eprintln!("Min:        {:.2} ms", s.min);
        eprintln!("Max:        {:.2} ms", s.max);
        eprintln!("Stdev:      {:.2} ms", s.stdev);
        eprintln!("1% low:     {:.2} ms ({:.0} FPS)", s.p1, 1000.0 / s.p1);
        eprintln!("0.1% low:   {:.2} ms ({:.0} FPS)", s.p01, 1000.0 / s.p01);
        eprintln!("=========================");

        // CSV output
        if let Ok(path) = std::env::var("BENCH_CSV") {
            let header = "backend,entities,frames,resolution,avg_ms,median_ms,min_ms,max_ms,stdev_ms,p1_ms,p01_ms,p99_ms\n";
            let row = format!(
                "{},{},{},1280x720,{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}\n",
                backend_name(),
                self.entity_count,
                s.count,
                s.avg,
                s.median,
                s.min,
                s.max,
                s.stdev,
                s.p1,
                s.p01,
                s.p99
            );
            let write_header = !std::path::Path::new(&path).exists();
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .expect("Failed to open CSV file");
            use std::io::Write;
            if write_header {
                f.write_all(header.as_bytes()).ok();
            }
            f.write_all(row.as_bytes()).ok();
            eprintln!("Results appended to {path}");
        }
    }
}

impl ApplicationHandler for BenchApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let window = event_loop.create_window(self.window_attrs.clone()).unwrap();

        #[cfg(all(target_os = "macos", feature = "metal-native"))]
        let gpu = {
            eprintln!("Initializing native Metal backend...");
            GpuContext::new_metal(std::sync::Arc::new(window))
        };
        #[cfg(not(all(target_os = "macos", feature = "metal-native")))]
        let gpu = {
            eprintln!("Initializing wgpu backend...");
            let (survey, inst) = HardwareSurvey::detect();
            GpuContext::new(window, &survey, &inst)
        };

        let renderer = Renderer::new(&gpu);
        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.setup_scene();
        self.last_frame = Instant::now();
        self.gpu.as_ref().unwrap().window.request_redraw();
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
                if self.done {
                    event_loop.exit();
                    return;
                }
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
    let mut app = BenchApp::new();
    event_loop.run_app(&mut app).unwrap();
}
