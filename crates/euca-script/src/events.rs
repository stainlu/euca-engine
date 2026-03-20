//! Script event bus: Lua scripts can register handlers via `euca.on("event", fn)`
//! and the engine fires them when ECS events occur.

use std::collections::HashMap;

use mlua::{Function, Lua, RegistryKey, Result as LuaResult};

/// Stores Lua callback registry keys grouped by event name.
///
/// When an ECS event fires (e.g. "death"), the engine looks up all registered
/// handlers and calls them with relevant data.
pub struct ScriptEventBus {
    /// event_name -> list of Lua registry keys pointing to handler functions.
    handlers: HashMap<String, Vec<RegistryKey>>,
}

impl ScriptEventBus {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a Lua function as a handler for the given event name.
    /// The function is stored in the Lua registry to prevent garbage collection.
    pub fn register(&mut self, lua: &Lua, event_name: &str, func: Function) -> LuaResult<()> {
        let key = lua.create_registry_value(func)?;
        self.handlers
            .entry(event_name.to_owned())
            .or_default()
            .push(key);
        Ok(())
    }

    /// Fire all handlers registered for the given event name.
    /// Each handler receives the provided arguments.
    pub fn fire<A: mlua::IntoLuaMulti + Clone>(
        &self,
        lua: &Lua,
        event_name: &str,
        args: A,
    ) -> LuaResult<()> {
        let Some(keys) = self.handlers.get(event_name) else {
            return Ok(());
        };
        for key in keys {
            let func: Function = lua.registry_value(key)?;
            func.call::<()>(args.clone())?;
        }
        Ok(())
    }

    /// Remove all handlers (e.g. on full script reload).
    pub fn clear(&mut self, lua: &Lua) {
        for (_, keys) in self.handlers.drain() {
            for key in keys {
                let _ = lua.remove_registry_value(key);
            }
        }
    }

    /// Remove handlers for a specific event.
    pub fn clear_event(&mut self, lua: &Lua, event_name: &str) {
        if let Some(keys) = self.handlers.remove(event_name) {
            for key in keys {
                let _ = lua.remove_registry_value(key);
            }
        }
    }
}

impl Default for ScriptEventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_fire() {
        let lua = Lua::new();
        let mut bus = ScriptEventBus::new();

        lua.load(
            r#"
            _test_called = false
            function _test_handler(entity_id)
                _test_called = true
            end
        "#,
        )
        .exec()
        .unwrap();

        let handler: Function = lua.globals().get("_test_handler").unwrap();
        bus.register(&lua, "death", handler).unwrap();
        bus.fire(&lua, "death", 42u32).unwrap();

        let called: bool = lua.globals().get("_test_called").unwrap();
        assert!(called);
    }

    #[test]
    fn fire_unknown_event_is_noop() {
        let lua = Lua::new();
        let bus = ScriptEventBus::new();
        // Should not error.
        bus.fire(&lua, "nonexistent", ()).unwrap();
    }

    #[test]
    fn clear_removes_all() {
        let lua = Lua::new();
        let mut bus = ScriptEventBus::new();

        let func = lua
            .create_function(|_, ()| Ok(()))
            .unwrap();
        bus.register(&lua, "test", func).unwrap();
        bus.clear(&lua);

        assert!(bus.handlers.is_empty());
    }
}
