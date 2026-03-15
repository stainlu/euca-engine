use euca_ecs::Entity;
use serde::{Deserialize, Serialize};

/// Marks an entity as a child of another entity.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parent(pub Entity);

/// Lists the children of an entity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Children(pub Vec<Entity>);
