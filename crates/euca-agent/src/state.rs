pub use euca_ecs::SharedWorld;

/// Unique identifier for an agent/player.
pub type AgentId = u32;

/// ECS component marking entity ownership by a specific agent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Owner(pub AgentId);
