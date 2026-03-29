//! Browser game runner using winit web support + WebGPU.
//!
//! winit's `EventLoopExtWebSys::spawn_app` drives the event loop via the
//! browser's `requestAnimationFrame`, so no manual RAF scheduling is needed.

use wasm_bindgen::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};
use winit::window::{WindowAttributes, WindowId};

// ---------------------------------------------------------------------------
// WebConfig — configurable from JavaScript via wasm-bindgen
// ---------------------------------------------------------------------------

/// Configuration for the web game runner.
#[wasm_bindgen]
pub struct WebConfig {
    /// CSS selector for the canvas element (e.g., `"#game-canvas"`).
    /// If empty, a new canvas is created and appended to `<body>`.
    canvas_selector: String,
    /// Desired canvas width in CSS pixels.
    width: u32,
    /// Desired canvas height in CSS pixels.
    height: u32,
}

#[wasm_bindgen]
impl WebConfig {
    /// Create a new `WebConfig`.
    ///
    /// `canvas_selector` — CSS selector for an existing `<canvas>` element, or
    /// an empty string to create one automatically.
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_selector: &str, width: u32, height: u32) -> Self {
        Self {
            canvas_selector: canvas_selector.to_string(),
            width,
            height,
        }
    }
}

// ---------------------------------------------------------------------------
// WASM initialization
// ---------------------------------------------------------------------------

/// Initialize the WASM environment — call this before [`run_web`].
///
/// Sets up the panic hook (for readable browser console stack traces) and
/// routes Rust `log` output to the browser console.
#[wasm_bindgen]
pub fn init_wasm() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
    log::info!("Euca Engine WASM initialized");
}

// ---------------------------------------------------------------------------
// Canvas helpers
// ---------------------------------------------------------------------------

/// Find an existing canvas by CSS selector, or create a new one and append it
/// to `<body>`.
fn get_or_create_canvas(selector: &str) -> web_sys::HtmlCanvasElement {
    let window = web_sys::window().expect("no global `window`");
    let document = window.document().expect("no `document` on `window`");

    if !selector.is_empty() {
        if let Some(element) = document.query_selector(selector).ok().flatten() {
            return element
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("selected element is not a <canvas>");
        }
    }

    // No existing canvas found — create one.
    let canvas = document
        .create_element("canvas")
        .expect("failed to create <canvas>")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("created element is not a <canvas>");

    canvas.set_id("euca-canvas");
    document
        .body()
        .expect("no <body> element")
        .append_child(&canvas)
        .expect("failed to append <canvas> to <body>");

    canvas
}

/// Convert an `HtmlCanvasElement` into a [`wgpu::SurfaceTarget`] that can be
/// passed to `wgpu::Instance::create_surface()`.
pub fn canvas_surface_target(canvas: &web_sys::HtmlCanvasElement) -> wgpu::SurfaceTarget<'static> {
    wgpu::SurfaceTarget::Canvas(canvas.clone())
}

// ---------------------------------------------------------------------------
// run_web — main entry point
// ---------------------------------------------------------------------------

/// Run the game loop in the browser.
///
/// This is the main entry point for web games. It:
///
/// 1. Creates or finds the `<canvas>` element.
/// 2. Builds a winit event loop (backed by `requestAnimationFrame`).
/// 3. Spawns the loop via [`EventLoopExtWebSys::spawn_app`] (never returns in
///    WASM).
///
/// Call from your game's WASM entry point:
///
/// ```ignore
/// #[wasm_bindgen(start)]
/// pub fn main() {
///     euca_web::init_wasm();
///     euca_web::run_web(euca_web::WebConfig::new("#game-canvas", 1280, 720));
/// }
/// ```
#[wasm_bindgen]
pub fn run_web(config: WebConfig) {
    let canvas = get_or_create_canvas(&config.canvas_selector);
    canvas.set_width(config.width);
    canvas.set_height(config.height);

    let event_loop = EventLoop::new().expect("failed to create event loop");

    let app = WebApp {
        canvas,
        width: config.width,
        height: config.height,
        initialized: false,
    };

    // `spawn_app` hands control to the browser's event loop and never returns.
    event_loop.spawn_app(app);
}

// ---------------------------------------------------------------------------
// WebApp — winit ApplicationHandler
// ---------------------------------------------------------------------------

struct WebApp {
    canvas: web_sys::HtmlCanvasElement,
    width: u32,
    height: u32,
    initialized: bool,
}

impl ApplicationHandler for WebApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.initialized {
            return;
        }

        let window_attrs = WindowAttributes::default()
            .with_title("Euca Engine")
            .with_canvas(Some(self.canvas.clone()));

        let _window = event_loop
            .create_window(window_attrs)
            .expect("failed to create window");

        log::info!("WebGPU canvas created: {}x{}", self.width, self.height);

        self.initialized = true;

        // TODO: Initialize wgpu instance, adapter, device, queue
        // TODO: Create surface from canvas via canvas_surface_target()
        // TODO: Initialize Renderer<WgpuDevice>
        // TODO: Start game systems
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::RedrawRequested => {
                // TODO: Run game tick + render frame
                log::trace!("Redraw requested");
            }
            WindowEvent::Resized(size) => {
                self.width = size.width;
                self.height = size.height;
                log::info!("Canvas resized: {}x{}", size.width, size.height);
                // TODO: Resize surface and renderer
            }
            _ => {
                // TODO: Forward input events to InputState
            }
        }
    }
}
