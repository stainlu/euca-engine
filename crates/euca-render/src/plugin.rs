use euca_core::Plugin;

/// Plugin that adds rendering capabilities to the app.
///
/// Must be added after the window is created (the GpuContext needs a Window).
pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, _app: &mut euca_core::App) {
        // GPU init happens after window creation (in the event loop).
        // This plugin currently serves as a marker that rendering is enabled.
        // The actual GPU initialization is done in the example's event handler
        // because wgpu needs a live window handle.
        log::info!("RenderPlugin registered");
    }

    fn name(&self) -> &str {
        "RenderPlugin"
    }
}
