//! WASM entry point for running Euca Engine in the browser.
//!
//! This crate provides [`WebApp`], a trait that games implement to run inside
//! an HTML canvas. The WASM bootstrap handles:
//!
//! * Canvas element acquisition from the DOM
//! * Async GPU initialization (no `pollster::block_on` on WASM)
//! * winit event loop via `EventLoop::spawn()` on web
//! * `requestAnimationFrame`-driven render loop
//!
//! # Usage (from a game crate)
//!
//! ```ignore
//! use euca_web::run_web_app;
//!
//! struct MyGame { /* ... */ }
//! impl euca_web::WebApp for MyGame { /* ... */ }
//!
//! #[cfg(target_arch = "wasm32")]
//! #[wasm_bindgen::prelude::wasm_bindgen(start)]
//! pub fn main() {
//!     run_web_app::<MyGame>();
//! }
//! ```

use euca_core::Time;
use euca_ecs::World;
use euca_math::Vec3;
use euca_render::{
    AmbientLight, Camera, DirectionalLight, DrawCommand, GpuContext, HardwareSurvey, MaterialRef,
    MeshRenderer, Renderer,
};
use euca_scene::GlobalTransform;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::EventLoop;
use winit::window::{WindowAttributes, WindowId};

// ---------------------------------------------------------------------------
// WebApp trait
// ---------------------------------------------------------------------------

/// Trait that web games implement to plug into the WASM bootstrap.
pub trait WebApp: 'static {
    /// Called once after GPU is ready. Set up your scene (spawn entities,
    /// upload meshes/materials, configure camera).
    fn init(&mut self, world: &mut World, renderer: &mut Renderer, gpu: &GpuContext);

    /// Called every frame before rendering. Update game state, handle input.
    fn update(&mut self, world: &mut World, dt: f32);
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

/// Run a [`WebApp`] in the browser.
///
/// On WASM this sets up the canvas, initializes the GPU asynchronously, and
/// enters the winit event loop. On native it falls back to a normal windowed
/// application (useful for development).
pub fn run_web_app<T: WebApp + Default>() {
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).ok();
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let mut app = WebAppRunner::<T> {
        game: T::default(),
        world: World::new(),
        gpu: None,
        renderer: None,
        initialized: false,
    };

    #[cfg(target_arch = "wasm32")]
    {
        // On WASM, winit uses spawn() instead of run_app().
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        event_loop.run_app(&mut app).ok();
    }
}

// ---------------------------------------------------------------------------
// Internal runner
// ---------------------------------------------------------------------------

struct WebAppRunner<T: WebApp> {
    game: T,
    world: World,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    initialized: bool,
}

impl<T: WebApp> ApplicationHandler for WebAppRunner<T> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.gpu.is_some() {
            return; // Already initialized.
        }

        let window_attrs = WindowAttributes::default().with_title("Euca Engine");

        #[cfg(target_arch = "wasm32")]
        let window_attrs = {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            // Attach to existing canvas with id="euca-canvas", or let winit
            // create one and append it to the document body.
            let canvas = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("euca-canvas"))
                .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok());

            if let Some(canvas) = canvas {
                window_attrs.with_canvas(Some(canvas))
            } else {
                window_attrs
            }
        };

        let window = event_loop
            .create_window(window_attrs)
            .expect("Failed to create window");

        let (survey, instance) = HardwareSurvey::detect();

        // On WASM, pollster delegates to the browser's microtask queue.
        #[cfg(target_arch = "wasm32")]
        let gpu = pollster::block_on(GpuContext::new_async(window, &survey, &instance));

        #[cfg(not(target_arch = "wasm32"))]
        let gpu = GpuContext::new(window, &survey, &instance);

        let mut renderer = Renderer::new(&gpu);

        self.world.insert_resource(Time::new());
        self.world
            .insert_resource(Camera::new(Vec3::new(0.0, 5.0, -8.0), Vec3::ZERO));
        self.world.insert_resource(DirectionalLight::default());
        self.world.insert_resource(AmbientLight::default());

        self.game.init(&mut self.world, &mut renderer, &gpu);

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.initialized = true;
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                    if let Some(r) = &mut self.renderer {
                        r.resize(gpu);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if !self.initialized {
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

impl<T: WebApp> WebAppRunner<T> {
    fn update_and_render(&mut self) {
        // Update time.
        let dt = {
            let time = self.world.resource_mut::<Time>().unwrap();
            time.update();
            time.delta
        };

        // Let the game update.
        self.game.update(&mut self.world, dt);

        // Propagate transforms.
        euca_scene::transform_propagation_system(&mut self.world);

        // Collect draw commands.
        let draw_commands = {
            let query = euca_ecs::Query::<(&GlobalTransform, &MeshRenderer, &MaterialRef)>::new(
                &self.world,
            );
            query
                .iter()
                .map(|(gt, mr, mat)| DrawCommand {
                    mesh: mr.mesh,
                    material: mat.handle,
                    model_matrix: gt.0.to_matrix(),
                    aabb: None,
                })
                .collect::<Vec<_>>()
        };

        // Render.
        let gpu = self.gpu.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();
        let camera = self.world.resource::<Camera>().unwrap().clone();
        let light = self
            .world
            .resource::<DirectionalLight>()
            .cloned()
            .unwrap_or_default();
        let ambient = self
            .world
            .resource::<AmbientLight>()
            .cloned()
            .unwrap_or_default();

        renderer.draw(gpu, &camera, &light, &ambient, &draw_commands);
    }
}

// ---------------------------------------------------------------------------
// Re-exports for convenience
// ---------------------------------------------------------------------------

pub use euca_core;
pub use euca_ecs;
pub use euca_math;
pub use euca_render;
pub use euca_scene;
