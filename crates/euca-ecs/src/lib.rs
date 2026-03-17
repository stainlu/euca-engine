//! A minimal, archetype-based Entity Component System (ECS) for the Euca game engine.
//!
//! Provides entities, components, queries, resources, events, and a schedule
//! for deterministic system execution with change detection and parallel iteration.

mod archetype;
mod command;
mod component;
mod entity;
mod event;
mod query;
mod resource;
mod schedule;
mod snapshot;
mod system;
mod system_param;
mod world;

pub use archetype::{Archetype, ArchetypeId};
pub use command::Commands;
pub use component::{Component, ComponentId, ComponentInfo, ComponentStorage};
pub use entity::Entity;
pub use event::Events;
pub use query::{ComponentAccess, Query, QueryFilter, With, Without, WorldQuery};
pub use resource::Resources;
pub use schedule::Schedule;
pub use snapshot::{EntitySnapshot, WorldSnapshot};
pub use system::{AccessSystem, IntoSystem, LabeledSystem, System};
pub use system_param::{Res, ResMut, SystemAccess};
pub use world::World;
