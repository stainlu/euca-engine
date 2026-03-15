use euca_ecs::{World, Schedule, IntoSystem};
use crate::plugin::Plugin;
use crate::time::Time;

/// The application — owns the ECS world, schedule, and runs the main loop.
pub struct App {
    pub world: World,
    pub schedule: Schedule,
    plugins: Vec<Box<dyn Plugin>>,
    /// Callback invoked each frame after schedule runs (for rendering, etc.)
    render_fn: Option<Box<dyn FnMut(&mut World)>>,
}

impl App {
    pub fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(Time::new());

        Self {
            world,
            schedule: Schedule::new(),
            plugins: Vec::new(),
            render_fn: None,
        }
    }

    /// Add a plugin to the app.
    pub fn add_plugin(&mut self, plugin: impl Plugin) -> &mut Self {
        plugin.build(self);
        self.plugins.push(Box::new(plugin));
        self
    }

    /// Add a system to the schedule.
    pub fn add_system<S: IntoSystem + 'static>(&mut self, system: S) -> &mut Self
    where
        S::System: 'static,
    {
        self.schedule.add_system(system);
        self
    }

    /// Insert a resource into the world.
    pub fn insert_resource<T: Send + Sync + 'static>(&mut self, value: T) -> &mut Self {
        self.world.insert_resource(value);
        self
    }

    /// Set the render callback (called after each schedule tick).
    pub fn set_render_fn(&mut self, f: impl FnMut(&mut World) + 'static) -> &mut Self {
        self.render_fn = Some(Box::new(f));
        self
    }

    /// Run the app in headless mode (no window, no event loop).
    /// Useful for AI agent simulations and testing.
    pub fn run_headless(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.world.resource_mut::<Time>().unwrap().update();
            self.schedule.run(&mut self.world);
        }
    }

    /// Run the app with a winit event loop (opens a window).
    ///
    /// This takes ownership because winit's event loop requires it on most platforms.
    pub fn run_windowed(self, window_title: &str, width: u32, height: u32) -> ! {
        use winit::application::ApplicationHandler;
        use winit::event::WindowEvent;
        use winit::event_loop::{ActiveEventLoop, EventLoop};
        use winit::window::{Window, WindowId, WindowAttributes};

        struct NovaApp {
            world: World,
            schedule: Schedule,
            render_fn: Option<Box<dyn FnMut(&mut World)>>,
            window: Option<Window>,
            window_attrs: WindowAttributes,
        }

        impl ApplicationHandler for NovaApp {
            fn resumed(&mut self, event_loop: &ActiveEventLoop) {
                if self.window.is_none() {
                    let window = event_loop
                        .create_window(self.window_attrs.clone())
                        .expect("Failed to create window");
                    self.window = Some(window);
                }
            }

            fn window_event(
                &mut self,
                event_loop: &ActiveEventLoop,
                _window_id: WindowId,
                event: WindowEvent,
            ) {
                match event {
                    WindowEvent::CloseRequested => {
                        event_loop.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        // Update time
                        self.world.resource_mut::<Time>().unwrap().update();

                        // Run ECS schedule
                        self.schedule.run(&mut self.world);

                        // Run render callback
                        if let Some(render_fn) = &mut self.render_fn {
                            (render_fn)(&mut self.world);
                        }

                        // Request next frame
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                    }
                    _ => {}
                }
            }
        }

        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let window_attrs = WindowAttributes::default()
            .with_title(window_title)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height));

        let mut app = NovaApp {
            world: self.world,
            schedule: self.schedule,
            render_fn: self.render_fn,
            window: None,
            window_attrs,
        };

        event_loop.run_app(&mut app).expect("Event loop error");
        std::process::exit(0);
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Counter(u32);

    #[test]
    fn headless_run() {
        let mut app = App::new();
        app.insert_resource(Counter(0));
        app.add_system(|w: &mut World| {
            w.resource_mut::<Counter>().unwrap().0 += 1;
        });

        app.run_headless(10);

        assert_eq!(app.world.resource::<Counter>().unwrap().0, 10);
        assert_eq!(app.world.current_tick(), 10);
    }
}
