//! Lua scripting with hot reload, sandboxing, and ECS bridge for the Euca engine.
//!
//! Provides:
//! - **ScriptEngine** resource: manages the Lua VM, loads scripts, enforces sandboxing
//! - **ScriptComponent**: per-entity component binding an entity to a Lua update function
//! - **ECS bridge**: `euca.spawn()`, `euca.set_position()`, `euca.get_health()`, etc.
//! - **Hot reload**: `ScriptWatcher` monitors a directory and reloads changed scripts
//! - **Script events**: `euca.on("event", fn)` / `euca.emit("event", data)` for Lua-side handlers
//!
//! # Forking semantics
//!
//! The Lua VM inside `ScriptEngine` is not deep-cloneable — a Lua VM owns
//! unique runtime state (loaded modules, event handlers, globals) that
//! cannot be duplicated by value. For [`World::clone`] forks, the engine
//! is stored as a `Shared<ScriptEngine>` (an `Arc<Mutex<ScriptEngine>>`
//! alias) so that the fork shares the same VM instance. Forked worlds
//! can still execute scripts, but any VM-level side effects (script
//! globals, loaded scripts) are visible across both parent and fork.

mod ecs_api;
mod engine;
mod events;
mod watcher;

use std::sync::{Arc, Mutex};

pub use engine::ScriptEngine;
pub use events::ScriptEventBus;
pub use watcher::ScriptWatcher;

/// Shared handle to a [`ScriptEngine`]. Registered as a world resource
/// so that forks (see [`World::clone`](euca_ecs::World::clone)) share a
/// single Lua VM.
pub type Shared<T> = Arc<Mutex<T>>;

/// Construct a [`Shared<ScriptEngine>`] suitable for insertion as a
/// world resource.
pub fn shared_engine(engine: ScriptEngine) -> Shared<ScriptEngine> {
    Arc::new(Mutex::new(engine))
}

/// Per-entity component that binds an entity to a Lua script.
///
/// The `script_name` identifies which loaded script file contains this entity's logic.
/// The `update_fn` names the Lua function called each tick (e.g. `"on_update"`).
#[derive(Clone, Debug)]
pub struct ScriptComponent {
    pub script_name: String,
    pub update_fn: String,
    pub enabled: bool,
}

impl ScriptComponent {
    /// Create a new script component with the default update function name `"on_update"`.
    pub fn new(script_name: impl Into<String>) -> Self {
        Self {
            script_name: script_name.into(),
            update_fn: "on_update".into(),
            enabled: true,
        }
    }

    /// Override the update function name.
    pub fn with_update_fn(mut self, name: impl Into<String>) -> Self {
        self.update_fn = name.into();
        self
    }
}

/// Maximum number of Lua instructions before a script is terminated.
pub const DEFAULT_INSTRUCTION_LIMIT: u32 = 100_000;

/// The system entry point: call each entity's Lua update function.
///
/// Follows the engine convention: `fn(world: &mut World, dt: f32)`.
pub fn script_tick_system(world: &mut euca_ecs::World, dt: f32) {
    // Collect entities + their script info to avoid borrow conflicts.
    let scripts: Vec<(euca_ecs::Entity, String, String)> = {
        let query = euca_ecs::Query::<(euca_ecs::Entity, &ScriptComponent)>::new(world);
        query
            .iter()
            .filter(|(_, sc)| sc.enabled)
            .map(|(e, sc)| (e, sc.script_name.clone(), sc.update_fn.clone()))
            .collect()
    };

    if scripts.is_empty() {
        return;
    }

    // Clone the shared engine handle. `Arc::clone` only bumps the refcount,
    // leaving no borrow on `world`, so we can pass `&mut world` below.
    let engine_arc = match world.resource::<Shared<ScriptEngine>>() {
        Some(arc) => arc.clone(),
        None => {
            log::warn!("script_tick_system: no ScriptEngine resource");
            return;
        }
    };
    let mut engine = engine_arc.lock().expect("ScriptEngine mutex poisoned");

    // Process pending hot-reload events before running scripts.
    engine.process_reload_queue();

    // Fire any pending ECS events into Lua event handlers.
    engine.dispatch_ecs_events(world);

    for (entity, script_name, update_fn) in &scripts {
        if let Err(err) = engine.call_entity_update(world, *entity, script_name, update_fn, dt) {
            log::error!("Lua error in {script_name}:{update_fn} for entity {entity}: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_ecs::World;

    #[test]
    fn script_component_defaults() {
        let sc = ScriptComponent::new("player.lua");
        assert_eq!(sc.script_name, "player.lua");
        assert_eq!(sc.update_fn, "on_update");
        assert!(sc.enabled);
    }

    #[test]
    fn script_component_custom_update_fn() {
        let sc = ScriptComponent::new("boss.lua").with_update_fn("on_boss_tick");
        assert_eq!(sc.update_fn, "on_boss_tick");
    }

    #[test]
    fn script_tick_system_no_engine_no_panic() {
        let mut world = World::new();
        // No ScriptEngine resource — should log a warning and return gracefully.
        script_tick_system(&mut world, 0.016);
    }

    #[test]
    fn script_tick_system_no_scripts() {
        let mut world = World::new();
        let engine = ScriptEngine::new(DEFAULT_INSTRUCTION_LIMIT).unwrap();
        world.insert_resource(shared_engine(engine));
        // No entities with ScriptComponent — should be a no-op.
        script_tick_system(&mut world, 0.016);
        assert!(world.resource::<Shared<ScriptEngine>>().is_some());
    }

    #[test]
    fn roundtrip_spawn_and_lua_update() {
        let mut world = World::new();
        let mut engine = ScriptEngine::new(DEFAULT_INSTRUCTION_LIMIT).unwrap();

        let script = r#"
            function on_update(entity_id, dt)
                local e = euca.spawn()
                euca.despawn(e)
            end
        "#;
        engine.load_string("test.lua", script).unwrap();

        world.insert_resource(shared_engine(engine));
        let e = world.spawn(ScriptComponent::new("test.lua"));
        assert!(world.is_alive(e));

        script_tick_system(&mut world, 0.016);

        // Engine should still be in resources.
        assert!(world.resource::<Shared<ScriptEngine>>().is_some());
    }
}
