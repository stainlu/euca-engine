// Euca Game — standalone game runtime and project management.
pub use euca_ecs;
pub use euca_math;
pub use euca_net;
pub use euca_physics;
pub use euca_render;
pub use euca_scene;

pub mod arena;
pub mod project;

pub use project::{PROJECT_FILE_NAME, ProjectConfig, WindowConfig};
