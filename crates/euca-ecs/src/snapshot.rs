use serde::{Deserialize, Serialize};

use crate::entity::Entity;
use crate::query::Query;
use crate::world::World;

/// A serializable snapshot of an entity's state.
/// Contains only position/rotation/scale — the universal networked state.
/// Additional component data can be added via extension traits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntitySnapshot {
    pub id: u32,
    pub generation: u32,
    pub position: Option<[f32; 3]>,
    pub rotation: Option<[f32; 4]>,
    pub scale: Option<[f32; 3]>,
}

/// A serializable snapshot of the entire world state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub tick: u64,
    pub entities: Vec<EntitySnapshot>,
}

impl WorldSnapshot {
    /// Capture a snapshot of all entities in the world that have transforms.
    /// Uses euca_scene::LocalTransform if available via the provided extractor.
    pub fn capture_with<F>(world: &World, extractor: F) -> Self
    where
        F: Fn(&World, Entity) -> EntitySnapshot,
    {
        let entities: Vec<EntitySnapshot> = {
            let query = Query::<Entity>::new(world);
            query.iter().map(|e| extractor(world, e)).collect()
        };

        WorldSnapshot {
            tick: world.current_tick(),
            entities,
        }
    }

    /// Capture a minimal snapshot (entity IDs only, no component data).
    pub fn capture_ids(world: &World) -> Self {
        let entities: Vec<EntitySnapshot> = {
            let query = Query::<Entity>::new(world);
            query
                .iter()
                .map(|e| EntitySnapshot {
                    id: e.index(),
                    generation: e.generation(),
                    position: None,
                    rotation: None,
                    scale: None,
                })
                .collect()
        };

        WorldSnapshot {
            tick: world.current_tick(),
            entities,
        }
    }

    /// Serialize to bytes (bincode).
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("WorldSnapshot serialization failed")
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        bincode::deserialize(data).map_err(|e| format!("WorldSnapshot deserialization failed: {e}"))
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("WorldSnapshot JSON serialization failed")
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json)
            .map_err(|e| format!("WorldSnapshot JSON deserialization failed: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_ids_roundtrip_bincode() {
        let mut world = World::new();
        world.spawn(42u32);
        world.spawn(99u32);
        world.tick();

        let snapshot = WorldSnapshot::capture_ids(&world);
        assert_eq!(snapshot.tick, 1);
        assert_eq!(snapshot.entities.len(), 2);

        let bytes = snapshot.to_bytes();
        let restored = WorldSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(restored.tick, 1);
        assert_eq!(restored.entities.len(), 2);
    }

    #[test]
    fn snapshot_json_roundtrip() {
        let mut world = World::new();
        world.spawn(42u32);
        world.tick();
        world.tick();

        let snapshot = WorldSnapshot::capture_ids(&world);
        let json = snapshot.to_json();
        assert!(json.contains("\"tick\": 2"));

        let restored = WorldSnapshot::from_json(&json).unwrap();
        assert_eq!(restored.tick, 2);
        assert_eq!(restored.entities.len(), 1);
    }

    #[test]
    fn snapshot_with_custom_extractor() {
        let mut world = World::new();
        let _e = world.spawn(42u32);

        let snapshot = WorldSnapshot::capture_with(&world, |_w, entity| EntitySnapshot {
            id: entity.index(),
            generation: entity.generation(),
            position: Some([1.0, 2.0, 3.0]),
            rotation: Some([0.0, 0.0, 0.0, 1.0]),
            scale: Some([1.0, 1.0, 1.0]),
        });

        assert_eq!(snapshot.entities.len(), 1);
        assert_eq!(snapshot.entities[0].position, Some([1.0, 2.0, 3.0]));
    }
}
