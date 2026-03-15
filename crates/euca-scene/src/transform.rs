use euca_math::Transform;
use serde::{Deserialize, Serialize};

/// Local transform relative to the entity's parent (or world origin if no parent).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalTransform(pub Transform);

impl Default for LocalTransform {
    fn default() -> Self {
        Self(Transform::IDENTITY)
    }
}

/// Computed world-space transform, updated by the transform propagation system.
///
/// Do not set this manually — it is overwritten each frame by `transform_propagation_system`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlobalTransform(pub Transform);

impl Default for GlobalTransform {
    fn default() -> Self {
        Self(Transform::IDENTITY)
    }
}
