use crate::app::App;

/// Trait for modular engine extensions.
///
/// Plugins add systems, resources, and other plugins to the app during build time.
pub trait Plugin: Send + Sync + 'static {
    /// Configure the app (add systems, resources, etc.).
    fn build(&self, app: &mut App);

    /// Plugin name for debugging.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}
