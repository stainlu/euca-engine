//! Per-entity typed key-value store for behavior tree data.

use std::collections::HashMap;

use euca_ecs::Entity;
use euca_math::Vec3;

/// A dynamically-typed value stored in a [`Blackboard`].
#[derive(Clone, Debug, PartialEq)]
pub enum BlackboardValue {
    Bool(bool),
    Float(f32),
    Int(i64),
    Vec3(Vec3),
    Entity(Entity),
    Str(String),
}

/// Per-entity typed key-value store.
///
/// Behavior tree nodes read and write data here instead of coupling directly
/// to component queries, keeping the tree logic decoupled from the ECS world.
#[derive(Clone, Debug, Default)]
pub struct Blackboard {
    data: HashMap<String, BlackboardValue>,
}

impl Blackboard {
    /// Creates an empty blackboard.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a key to the given value, returning the previous value if any.
    pub fn set(
        &mut self,
        key: impl Into<String>,
        value: BlackboardValue,
    ) -> Option<BlackboardValue> {
        self.data.insert(key.into(), value)
    }

    /// Gets a reference to the value for `key`, if present.
    pub fn get(&self, key: &str) -> Option<&BlackboardValue> {
        self.data.get(key)
    }

    /// Returns `true` if `key` exists in the blackboard.
    pub fn has(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Removes a key from the blackboard, returning its value if it existed.
    pub fn remove(&mut self, key: &str) -> Option<BlackboardValue> {
        self.data.remove(key)
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the blackboard is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    // ── Convenience getters ──

    /// Gets a `bool` value, returning `None` if the key is missing or a different type.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            BlackboardValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Gets an `f32` value.
    pub fn get_float(&self, key: &str) -> Option<f32> {
        match self.get(key)? {
            BlackboardValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Gets an `i64` value.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        match self.get(key)? {
            BlackboardValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Gets a [`Vec3`] value.
    pub fn get_vec3(&self, key: &str) -> Option<Vec3> {
        match self.get(key)? {
            BlackboardValue::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    /// Gets an [`Entity`] value.
    pub fn get_entity(&self, key: &str) -> Option<Entity> {
        match self.get(key)? {
            BlackboardValue::Entity(v) => Some(*v),
            _ => None,
        }
    }

    /// Gets a `&str` value.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            BlackboardValue::Str(v) => Some(v),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_get() {
        let mut bb = Blackboard::new();
        bb.set("health", BlackboardValue::Float(100.0));
        bb.set("alive", BlackboardValue::Bool(true));
        bb.set("name", BlackboardValue::Str("Guard".into()));

        assert_eq!(bb.get_float("health"), Some(100.0));
        assert_eq!(bb.get_bool("alive"), Some(true));
        assert_eq!(bb.get_str("name"), Some("Guard"));
        assert_eq!(bb.len(), 3);
    }

    #[test]
    fn has_and_remove() {
        let mut bb = Blackboard::new();
        bb.set("key", BlackboardValue::Int(42));

        assert!(bb.has("key"));
        assert!(!bb.has("missing"));

        let removed = bb.remove("key");
        assert_eq!(removed, Some(BlackboardValue::Int(42)));
        assert!(!bb.has("key"));
    }

    #[test]
    fn type_mismatch_returns_none() {
        let mut bb = Blackboard::new();
        bb.set("val", BlackboardValue::Float(1.0));

        assert_eq!(bb.get_bool("val"), None);
        assert_eq!(bb.get_int("val"), None);
        assert_eq!(bb.get_float("val"), Some(1.0));
    }

    #[test]
    fn overwrite_returns_previous() {
        let mut bb = Blackboard::new();
        assert!(bb.set("k", BlackboardValue::Int(1)).is_none());
        let prev = bb.set("k", BlackboardValue::Int(2));
        assert_eq!(prev, Some(BlackboardValue::Int(1)));
    }

    #[test]
    fn vec3_and_entity_values() {
        let mut bb = Blackboard::new();
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let ent = Entity::from_raw(7, 1);

        bb.set("pos", BlackboardValue::Vec3(pos));
        bb.set("target", BlackboardValue::Entity(ent));

        assert_eq!(bb.get_vec3("pos"), Some(pos));
        assert_eq!(bb.get_entity("target"), Some(ent));
    }
}
