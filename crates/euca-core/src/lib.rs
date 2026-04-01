//! Core application framework: time, plugins, profiling, and the main loop.
//!
//! Start with [`App`], register [`Plugin`]s, then call [`App::run_headless`]
//! or [`App::run_windowed`].

mod app;
pub mod platform;
mod plugin;
mod profiler;
mod time;

pub use app::App;
pub use platform::performance_core_count;
pub use plugin::Plugin;
pub use profiler::{ProfileSection, Profiler, profiler_begin, profiler_end};
pub use time::Time;

/// Re-export `winit` for downstream crates that need window types.
pub use winit;
