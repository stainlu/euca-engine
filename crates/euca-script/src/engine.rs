//! The `ScriptEngine` resource: manages Lua VM, script loading, sandboxing,
//! hot reload, and ECS event dispatch.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use euca_ecs::{Entity, World};
use euca_gameplay::DeathEvent;
use mlua::{Function, Lua, Result as LuaResult, Table};

use crate::ecs_api::{self, register_euca_api, with_world_context};
use crate::watcher::ScriptWatcher;

/// The central Lua scripting resource. Stored as a World resource.
///
/// Owns the Lua VM, tracks loaded script names, manages the instruction-count
/// sandbox, and optionally watches a directory for hot reload.
pub struct ScriptEngine {
    lua: Lua,
    loaded_scripts: HashSet<String>,
    instruction_limit: u32,
    watcher: Option<ScriptWatcher>,
    /// Paths queued for reload by the watcher (drained each tick).
    reload_queue: Vec<PathBuf>,
}

// SAFETY: Lua VM is only accessed from the script_tick_system, which runs
// single-threaded on the main thread. The ScriptEngine is removed from World
// resources during Lua calls to prevent concurrent access.
unsafe impl Send for ScriptEngine {}
unsafe impl Sync for ScriptEngine {}

impl ScriptEngine {
    /// Create a new script engine with the given instruction limit per call.
    pub fn new(instruction_limit: u32) -> LuaResult<Self> {
        let lua = Lua::new();

        // ── Sandboxing: remove dangerous globals ──
        Self::apply_sandbox(&lua)?;

        // ── Register ECS bridge ──
        register_euca_api(&lua)?;

        // ── Initialize event handler registry ──
        lua.load("_euca_event_handlers = {}").exec()?;

        Ok(Self {
            lua,
            loaded_scripts: HashSet::new(),
            instruction_limit,
            watcher: None,
            reload_queue: Vec::new(),
        })
    }

    /// Start watching a directory for `.lua` file changes.
    /// Changed files are reloaded automatically each tick.
    pub fn watch_directory(&mut self, dir: &Path) -> Result<(), notify::Error> {
        let watcher = ScriptWatcher::new(dir)?;
        self.watcher = Some(watcher);
        Ok(())
    }

    /// Load a script from a string. The `name` is used to identify the script
    /// (e.g. `"player.lua"`). Replaces any previously loaded script with the same name.
    pub fn load_string(&mut self, name: &str, source: &str) -> LuaResult<()> {
        self.lua.load(source).set_name(name).exec()?;
        self.loaded_scripts.insert(name.to_owned());
        log::info!("Loaded script: {name}");
        Ok(())
    }

    /// Load a script from a file path. The script name is derived from the file name.
    pub fn load_file(&mut self, path: &Path) -> LuaResult<()> {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.lua");
        let source = std::fs::read_to_string(path).map_err(mlua::Error::external)?;
        self.load_string(name, &source)
    }

    /// Load all `.lua` files from a directory (non-recursive).
    pub fn load_directory(&mut self, dir: &Path) -> LuaResult<()> {
        let entries = std::fs::read_dir(dir).map_err(mlua::Error::external)?;
        for entry in entries {
            let entry = entry.map_err(mlua::Error::external)?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "lua") {
                self.load_file(&path)?;
            }
        }
        Ok(())
    }

    /// Call an entity's Lua update function with instruction-count sandboxing.
    pub(crate) fn call_entity_update(
        &self,
        world: &mut World,
        entity: Entity,
        script_name: &str,
        update_fn: &str,
        dt: f32,
    ) -> LuaResult<()> {
        if !self.loaded_scripts.contains(script_name) {
            return Err(mlua::Error::external(format!(
                "Script not loaded: {script_name}"
            )));
        }

        let func: Function = match self.lua.globals().get(update_fn) {
            Ok(f) => f,
            Err(_) => {
                return Err(mlua::Error::external(format!(
                    "Function '{update_fn}' not found in Lua globals (expected from {script_name})"
                )));
            }
        };

        // Set up instruction count hook for sandboxing.
        let limit = self.instruction_limit;
        self.lua.set_hook(
            mlua::HookTriggers::new().every_nth_instruction(limit),
            move |_lua, _debug| {
                Err(mlua::Error::external(format!(
                    "Script exceeded instruction limit ({limit})"
                )))
            },
        );

        let entity_id = ecs_api::entity_to_lua_id(entity);

        let result = with_world_context(world, dt, || func.call::<()>((entity_id, dt)));

        // Remove the hook after execution.
        self.lua.remove_hook();

        result
    }

    /// Process filesystem watcher events and reload changed scripts.
    pub(crate) fn process_reload_queue(&mut self) {
        // Drain watcher notifications.
        if let Some(ref watcher) = self.watcher {
            let changed = watcher.drain();
            self.reload_queue.extend(changed);
        }

        // Reload queued files.
        let queue = std::mem::take(&mut self.reload_queue);
        for path in queue {
            match self.load_file(&path) {
                Ok(()) => log::info!("Hot-reloaded: {}", path.display()),
                Err(e) => log::error!("Failed to hot-reload {}: {e}", path.display()),
            }
        }
    }

    /// Dispatch ECS events (e.g. DeathEvent) to Lua event handlers.
    pub(crate) fn dispatch_ecs_events(&self, world: &mut World) {
        // Dispatch death events.
        let death_events: Vec<(u64, Option<u64>)> = world
            .read_events::<DeathEvent>()
            .map(|de| {
                (
                    ecs_api::entity_to_lua_id(de.entity),
                    de.killer.map(ecs_api::entity_to_lua_id),
                )
            })
            .collect();

        if death_events.is_empty() {
            return;
        }

        let registry: Option<Table> = self.lua.globals().get("_euca_event_handlers").ok();
        let Some(registry) = registry else { return };
        let handlers: Option<Table> = registry.get("death").ok().flatten();
        let Some(handlers) = handlers else { return };

        for (entity_id, killer_id) in death_events {
            for pair in handlers.pairs::<mlua::Integer, Function>() {
                let Ok((_, func)) = pair else { continue };
                let killer_val = match killer_id {
                    Some(kid) => mlua::Value::Integer(kid as i64),
                    None => mlua::Value::Nil,
                };
                if let Err(e) = func.call::<()>((entity_id, killer_val)) {
                    log::error!("Error in death event handler: {e}");
                }
            }
        }
    }

    /// Apply sandboxing: remove dangerous Lua standard library globals.
    fn apply_sandbox(lua: &Lua) -> LuaResult<()> {
        let globals = lua.globals();

        // Remove dangerous modules.
        globals.set("io", mlua::Value::Nil)?;
        globals.set("os", mlua::Value::Nil)?;
        globals.set("debug", mlua::Value::Nil)?;
        globals.set("loadfile", mlua::Value::Nil)?;
        globals.set("dofile", mlua::Value::Nil)?;
        globals.set("package", mlua::Value::Nil)?;

        // Remove raw load to prevent loading arbitrary bytecode.
        globals.set("load", mlua::Value::Nil)?;

        Ok(())
    }

    /// Access the underlying Lua VM (for advanced usage / testing).
    pub fn lua(&self) -> &Lua {
        &self.lua
    }

    /// Check if a script has been loaded by name.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.loaded_scripts.contains(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_INSTRUCTION_LIMIT;

    #[test]
    fn create_engine() {
        let engine = ScriptEngine::new(DEFAULT_INSTRUCTION_LIMIT).unwrap();
        assert!(!engine.is_loaded("anything.lua"));
    }

    #[test]
    fn sandbox_removes_dangerous_globals() {
        let engine = ScriptEngine::new(DEFAULT_INSTRUCTION_LIMIT).unwrap();
        let lua = engine.lua();

        let io: mlua::Value = lua.globals().get("io").unwrap();
        assert!(io.is_nil());

        let os: mlua::Value = lua.globals().get("os").unwrap();
        assert!(os.is_nil());

        let debug: mlua::Value = lua.globals().get("debug").unwrap();
        assert!(debug.is_nil());

        let loadfile: mlua::Value = lua.globals().get("loadfile").unwrap();
        assert!(loadfile.is_nil());

        let dofile: mlua::Value = lua.globals().get("dofile").unwrap();
        assert!(dofile.is_nil());

        let package: mlua::Value = lua.globals().get("package").unwrap();
        assert!(package.is_nil());

        let load: mlua::Value = lua.globals().get("load").unwrap();
        assert!(load.is_nil());
    }

    #[test]
    fn sandbox_allows_safe_functions() {
        let engine = ScriptEngine::new(DEFAULT_INSTRUCTION_LIMIT).unwrap();
        let lua = engine.lua();

        // Math, string, table, print should still work.
        lua.load("local x = math.abs(-5)").exec().unwrap();
        lua.load("local s = string.upper('hello')").exec().unwrap();
        lua.load("local t = {}; table.insert(t, 1)").exec().unwrap();
    }

    #[test]
    fn load_and_call_script() {
        let mut engine = ScriptEngine::new(DEFAULT_INSTRUCTION_LIMIT).unwrap();
        let mut world = World::new();

        engine
            .load_string(
                "test.lua",
                r#"
                function on_update(entity_id, dt)
                    _test_result = dt
                end
            "#,
            )
            .unwrap();

        assert!(engine.is_loaded("test.lua"));

        let entity = world.spawn_empty();
        engine
            .call_entity_update(&mut world, entity, "test.lua", "on_update", 0.05)
            .unwrap();

        let result: f32 = engine.lua().globals().get("_test_result").unwrap();
        assert!((result - 0.05).abs() < 1e-6);
    }

    #[test]
    fn instruction_limit_terminates_infinite_loop() {
        let mut engine = ScriptEngine::new(1_000).unwrap(); // Very low limit.
        let mut world = World::new();

        engine
            .load_string(
                "infinite.lua",
                r#"
                function on_update(entity_id, dt)
                    while true do end
                end
            "#,
            )
            .unwrap();

        let entity = world.spawn_empty();
        let result =
            engine.call_entity_update(&mut world, entity, "infinite.lua", "on_update", 0.016);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("instruction limit"));
    }

    #[test]
    fn missing_script_returns_error() {
        let engine = ScriptEngine::new(DEFAULT_INSTRUCTION_LIMIT).unwrap();
        let mut world = World::new();
        let entity = world.spawn_empty();

        let result = engine.call_entity_update(&mut world, entity, "nope.lua", "on_update", 0.016);
        assert!(result.is_err());
    }

    #[test]
    fn load_file_from_disk() {
        let dir = std::env::temp_dir().join("euca_script_test_load");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let script_path = dir.join("hello.lua");
        std::fs::write(&script_path, "function hello() return 42 end").unwrap();

        let mut engine = ScriptEngine::new(DEFAULT_INSTRUCTION_LIMIT).unwrap();
        engine.load_file(&script_path).unwrap();
        assert!(engine.is_loaded("hello.lua"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_directory() {
        let dir = std::env::temp_dir().join("euca_script_test_dir");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        std::fs::write(dir.join("a.lua"), "function a() end").unwrap();
        std::fs::write(dir.join("b.lua"), "function b() end").unwrap();
        std::fs::write(dir.join("c.txt"), "not lua").unwrap();

        let mut engine = ScriptEngine::new(DEFAULT_INSTRUCTION_LIMIT).unwrap();
        engine.load_directory(&dir).unwrap();

        assert!(engine.is_loaded("a.lua"));
        assert!(engine.is_loaded("b.lua"));
        assert!(!engine.is_loaded("c.txt"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ecs_spawn_from_lua() {
        let mut engine = ScriptEngine::new(DEFAULT_INSTRUCTION_LIMIT).unwrap();
        let mut world = World::new();

        engine
            .load_string(
                "spawn_test.lua",
                r#"
                function on_update(entity_id, dt)
                    _spawned = euca.spawn()
                end
            "#,
            )
            .unwrap();

        let entity = world.spawn_empty();
        let initial_count = world.entity_count();

        engine
            .call_entity_update(&mut world, entity, "spawn_test.lua", "on_update", 0.016)
            .unwrap();

        // Should have spawned one more entity.
        assert_eq!(world.entity_count(), initial_count + 1);
    }
}
