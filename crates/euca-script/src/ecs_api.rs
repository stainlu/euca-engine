//! ECS bridge: exposes `euca.*` functions to Lua scripts.
//!
//! The bridge works by passing a raw `*mut World` pointer through Lua's app data
//! during each script call. This pointer is only valid for the duration of that call.
//!
//! Exposed API:
//! - `euca.spawn()` -> entity_id (u64)
//! - `euca.despawn(entity_id)`
//! - `euca.set_position(entity_id, x, y, z)`
//! - `euca.get_position(entity_id)` -> x, y, z
//! - `euca.get_health(entity_id)` -> current, max
//! - `euca.set_health(entity_id, current)`
//! - `euca.delta_time()` -> f32
//! - `euca.on("event", handler_fn)` — register a script event handler
//! - `euca.emit("event", ...)` — fire a script event

use std::cell::Cell;

use euca_ecs::{Entity, World};
use euca_gameplay::Health;
use euca_math::Vec3;
use euca_scene::LocalTransform;
use mlua::{Function, Lua, Result as LuaResult, Table};

// Thread-local raw pointer to the current World and delta time.
// Set before each Lua call, cleared after. This avoids storing references
// inside the Lua VM that would create lifetime issues.
thread_local! {
    static WORLD_PTR: Cell<*mut World> = const { Cell::new(std::ptr::null_mut()) };
    static DELTA_TIME: Cell<f32> = const { Cell::new(0.0) };
}

/// Encode an Entity as a u64 for Lua (index in high 32 bits, generation in low 32 bits).
pub(crate) fn entity_to_lua_id(entity: Entity) -> u64 {
    ((entity.index() as u64) << 32) | (entity.generation() as u64)
}

/// Decode a u64 back to an Entity.
pub(crate) fn lua_id_to_entity(id: u64) -> Entity {
    let index = (id >> 32) as u32;
    let generation = id as u32;
    Entity::from_raw(index, generation)
}

/// Set the world pointer and delta time for the duration of a Lua call.
/// Returns a guard that clears the pointer on drop.
pub(crate) fn with_world_context<R>(
    world: &mut World,
    dt: f32,
    f: impl FnOnce() -> R,
) -> R {
    WORLD_PTR.set(world as *mut World);
    DELTA_TIME.set(dt);
    let result = f();
    WORLD_PTR.set(std::ptr::null_mut());
    result
}

/// Access the current World pointer. Panics if called outside a world context.
fn with_world<R>(f: impl FnOnce(&mut World) -> R) -> R {
    let ptr = WORLD_PTR.get();
    assert!(!ptr.is_null(), "euca API called outside of script context");
    // SAFETY: The pointer is valid for the duration of the script call,
    // and we are single-threaded within script execution.
    let world = unsafe { &mut *ptr };
    f(world)
}

/// Register the `euca` global table with all bridge functions.
pub(crate) fn register_euca_api(lua: &Lua) -> LuaResult<()> {
    let euca = lua.create_table()?;

    euca.set("spawn", lua.create_function(api_spawn)?)?;
    euca.set("despawn", lua.create_function(api_despawn)?)?;
    euca.set("set_position", lua.create_function(api_set_position)?)?;
    euca.set("get_position", lua.create_function(api_get_position)?)?;
    euca.set("get_health", lua.create_function(api_get_health)?)?;
    euca.set("set_health", lua.create_function(api_set_health)?)?;
    euca.set("delta_time", lua.create_function(api_delta_time)?)?;
    euca.set("on", lua.create_function(api_on)?)?;
    euca.set("emit", lua.create_function(api_emit)?)?;

    lua.globals().set("euca", euca)?;
    Ok(())
}

// ── API Implementations ──

fn api_spawn(_lua: &Lua, _: ()) -> LuaResult<u64> {
    Ok(with_world(|world| {
        let entity = world.spawn_empty();
        entity_to_lua_id(entity)
    }))
}

fn api_despawn(_lua: &Lua, id: u64) -> LuaResult<bool> {
    Ok(with_world(|world| {
        let entity = lua_id_to_entity(id);
        world.despawn(entity)
    }))
}

fn api_set_position(_lua: &Lua, (id, x, y, z): (u64, f32, f32, f32)) -> LuaResult<()> {
    with_world(|world| {
        let entity = lua_id_to_entity(id);
        if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
            lt.0.translation = Vec3::new(x, y, z);
        } else {
            // Entity doesn't have a LocalTransform yet — add one.
            let mut transform = LocalTransform::default();
            transform.0.translation = Vec3::new(x, y, z);
            world.insert(entity, transform);
        }
    });
    Ok(())
}

fn api_get_position(_lua: &Lua, id: u64) -> LuaResult<(f32, f32, f32)> {
    Ok(with_world(|world| {
        let entity = lua_id_to_entity(id);
        match world.get::<LocalTransform>(entity) {
            Some(lt) => {
                let t = lt.0.translation;
                (t.x, t.y, t.z)
            }
            None => (0.0, 0.0, 0.0),
        }
    }))
}

fn api_get_health(_lua: &Lua, id: u64) -> LuaResult<(f32, f32)> {
    Ok(with_world(|world| {
        let entity = lua_id_to_entity(id);
        match world.get::<Health>(entity) {
            Some(h) => (h.current, h.max),
            None => (0.0, 0.0),
        }
    }))
}

fn api_set_health(_lua: &Lua, (id, current): (u64, f32)) -> LuaResult<()> {
    with_world(|world| {
        let entity = lua_id_to_entity(id);
        if let Some(h) = world.get_mut::<Health>(entity) {
            h.current = current.max(0.0).min(h.max);
        }
    });
    Ok(())
}

fn api_delta_time(_lua: &Lua, _: ()) -> LuaResult<f32> {
    Ok(DELTA_TIME.get())
}

/// `euca.on("event_name", handler_fn)` — register a Lua event handler.
///
/// The handler is stored in a registry table `_euca_event_handlers[event_name]`.
fn api_on(lua: &Lua, (event_name, handler): (String, Function)) -> LuaResult<()> {
    let registry: Table = lua.globals().get("_euca_event_handlers")?;
    let list: Table = match registry.get::<Option<Table>>(event_name.as_str())? {
        Some(t) => t,
        None => {
            let t = lua.create_table()?;
            registry.set(event_name.as_str(), t.clone())?;
            t
        }
    };
    let len = list.len()?;
    list.set(len + 1, handler)?;
    Ok(())
}

/// `euca.emit("event_name", ...)` — fire all handlers for the given event.
fn api_emit(lua: &Lua, args: mlua::MultiValue) -> LuaResult<()> {
    let mut iter = args.into_iter();
    let event_name: String = iter
        .next()
        .and_then(|v| lua.unpack(v).ok())
        .ok_or_else(|| mlua::Error::external("euca.emit requires an event name"))?;
    let handler_args: Vec<mlua::Value> = iter.collect();

    let registry: Table = lua.globals().get("_euca_event_handlers")?;
    let list: Option<Table> = registry.get(event_name.as_str())?;
    if let Some(list) = list {
        for pair in list.pairs::<mlua::Integer, Function>() {
            let (_, func) = pair?;
            let multi = mlua::MultiValue::from_iter(handler_args.clone());
            func.call::<()>(multi)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_id_roundtrip() {
        let entity = Entity::from_raw(42, 7);
        let id = entity_to_lua_id(entity);
        let back = lua_id_to_entity(id);
        assert_eq!(back.index(), 42);
        assert_eq!(back.generation(), 7);
    }

    #[test]
    fn entity_id_zero() {
        let entity = Entity::from_raw(0, 0);
        let id = entity_to_lua_id(entity);
        assert_eq!(id, 0);
        let back = lua_id_to_entity(id);
        assert_eq!(back.index(), 0);
        assert_eq!(back.generation(), 0);
    }

    #[test]
    fn entity_id_max_values() {
        let entity = Entity::from_raw(u32::MAX, u32::MAX);
        let id = entity_to_lua_id(entity);
        let back = lua_id_to_entity(id);
        assert_eq!(back.index(), u32::MAX);
        assert_eq!(back.generation(), u32::MAX);
    }

    #[test]
    fn spawn_and_despawn_via_lua() {
        let lua = Lua::new();
        let mut world = World::new();

        register_euca_api(&lua).unwrap();

        with_world_context(&mut world, 0.016, || {
            lua.load("local e = euca.spawn(); euca.despawn(e)")
                .exec()
                .unwrap();
        });

        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn delta_time_exposed() {
        let lua = Lua::new();
        let mut world = World::new();

        register_euca_api(&lua).unwrap();

        with_world_context(&mut world, 0.033, || {
            lua.load("_dt = euca.delta_time()").exec().unwrap();
        });

        let dt: f32 = lua.globals().get("_dt").unwrap();
        assert!((dt - 0.033).abs() < 1e-6);
    }

    #[test]
    fn on_and_emit() {
        let lua = Lua::new();
        let mut world = World::new();

        register_euca_api(&lua).unwrap();
        // Initialize the handler registry.
        lua.load("_euca_event_handlers = {}").exec().unwrap();

        with_world_context(&mut world, 0.0, || {
            lua.load(
                r#"
                _received = nil
                euca.on("test_event", function(val) _received = val end)
                euca.emit("test_event", 42)
            "#,
            )
            .exec()
            .unwrap();
        });

        let received: i32 = lua.globals().get("_received").unwrap();
        assert_eq!(received, 42);
    }
}
