use euca_math::Transform;
use euca_reflect::Reflect;
use serde::{Deserialize, Serialize};

/// Local transform relative to the entity's parent (or world origin if no parent).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalTransform(pub Transform);

impl Default for LocalTransform {
    fn default() -> Self {
        Self(Transform::IDENTITY)
    }
}

impl Reflect for LocalTransform {
    fn type_name(&self) -> &'static str {
        "LocalTransform"
    }
    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            (
                "translation",
                format!(
                    "({:.3}, {:.3}, {:.3})",
                    self.0.translation.x, self.0.translation.y, self.0.translation.z
                ),
            ),
            (
                "scale",
                format!(
                    "({:.3}, {:.3}, {:.3})",
                    self.0.scale.x, self.0.scale.y, self.0.scale.z
                ),
            ),
        ]
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

impl Reflect for GlobalTransform {
    fn type_name(&self) -> &'static str {
        "GlobalTransform"
    }
    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![(
            "world_pos",
            format!(
                "({:.3}, {:.3}, {:.3})",
                self.0.translation.x, self.0.translation.y, self.0.translation.z
            ),
        )]
    }
}
