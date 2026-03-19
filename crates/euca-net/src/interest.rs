//! Interest management: only replicate entities near each client.

use euca_ecs::{Entity, Query, World};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for interest-based culling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterestConfig {
    /// Max distance at which entities are replicated to a client.
    pub max_relevance_distance: f32,
    /// How often to re-evaluate interest (in ticks). 0 = every tick.
    pub update_interval: u32,
}

impl Default for InterestConfig {
    fn default() -> Self {
        Self {
            max_relevance_distance: 100.0,
            update_interval: 5,
        }
    }
}

/// Per-client interest region: tracks which entities are relevant.
#[derive(Clone, Debug, Default)]
pub struct ClientInterest {
    /// Entities currently in this client's interest set.
    pub relevant_entities: HashSet<u32>,
    /// Client's viewpoint position (updated each tick).
    pub position: [f32; 3],
    /// Ticks since last interest re-evaluation.
    pub ticks_since_update: u32,
}

/// World resource: manages interest for all connected clients.
#[derive(Clone, Debug, Default)]
pub struct InterestManager {
    pub config: InterestConfig,
    /// Per-client interest data, keyed by client ID.
    pub clients: HashMap<u32, ClientInterest>,
}

impl InterestManager {
    pub fn new(config: InterestConfig) -> Self {
        Self {
            config,
            clients: HashMap::new(),
        }
    }

    /// Register a new client with an initial position.
    pub fn add_client(&mut self, client_id: u32, position: [f32; 3]) {
        self.clients.insert(
            client_id,
            ClientInterest {
                relevant_entities: HashSet::new(),
                position,
                ticks_since_update: 0,
            },
        );
    }

    /// Remove a client.
    pub fn remove_client(&mut self, client_id: u32) {
        self.clients.remove(&client_id);
    }

    /// Update a client's viewpoint position.
    pub fn update_position(&mut self, client_id: u32, position: [f32; 3]) {
        if let Some(interest) = self.clients.get_mut(&client_id) {
            interest.position = position;
        }
    }

    /// Compute which entities are relevant for a client based on distance.
    /// `entity_positions` is a list of (entity_index, x, y, z).
    pub fn compute_interest(&mut self, client_id: u32, entity_positions: &[(u32, f32, f32, f32)]) {
        let max_dist_sq = self.config.max_relevance_distance * self.config.max_relevance_distance;

        if let Some(interest) = self.clients.get_mut(&client_id) {
            interest.ticks_since_update += 1;
            if interest.ticks_since_update < self.config.update_interval {
                return; // Skip re-evaluation
            }
            interest.ticks_since_update = 0;

            let cx = interest.position[0];
            let cy = interest.position[1];
            let cz = interest.position[2];

            interest.relevant_entities.clear();
            for &(entity_idx, ex, ey, ez) in entity_positions {
                let dx = ex - cx;
                let dy = ey - cy;
                let dz = ez - cz;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq <= max_dist_sq {
                    interest.relevant_entities.insert(entity_idx);
                }
            }
        }
    }

    /// Check if an entity is relevant for a client.
    pub fn is_relevant(&self, client_id: u32, entity_index: u32) -> bool {
        self.clients
            .get(&client_id)
            .is_some_and(|i| i.relevant_entities.contains(&entity_index))
    }

    /// Get the set of relevant entities for a client.
    pub fn relevant_entities(&self, client_id: u32) -> Option<&HashSet<u32>> {
        self.clients.get(&client_id).map(|i| &i.relevant_entities)
    }
}

/// System: update interest sets for all clients using world entity positions.
pub fn interest_culling_system(world: &mut World) {
    // Collect entity positions
    let positions: Vec<(u32, f32, f32, f32)> = {
        let query = Query::<(Entity, &crate::protocol::Replicated)>::new(world);
        query
            .iter()
            .map(|(e, _)| {
                // Placeholder: real implementation would read GlobalTransform
                (e.index(), 0.0, 0.0, 0.0)
            })
            .collect()
    };

    if let Some(manager) = world.resource_mut::<InterestManager>() {
        let client_ids: Vec<u32> = manager.clients.keys().copied().collect();
        for client_id in client_ids {
            manager.compute_interest(client_id, &positions);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interest_culling_by_distance() {
        let mut manager = InterestManager::new(InterestConfig {
            max_relevance_distance: 10.0,
            update_interval: 0,
        });

        manager.add_client(1, [0.0, 0.0, 0.0]);

        let entities = vec![
            (0, 5.0, 0.0, 0.0),  // within range
            (1, 15.0, 0.0, 0.0), // out of range
            (2, 0.0, 8.0, 0.0),  // within range
        ];

        manager.compute_interest(1, &entities);

        assert!(manager.is_relevant(1, 0));
        assert!(!manager.is_relevant(1, 1));
        assert!(manager.is_relevant(1, 2));
    }

    #[test]
    fn update_interval_skips() {
        let mut manager = InterestManager::new(InterestConfig {
            max_relevance_distance: 10.0,
            update_interval: 3,
        });

        manager.add_client(1, [0.0, 0.0, 0.0]);

        let entities = vec![(0, 5.0, 0.0, 0.0)];

        // First compute should skip (ticks_since_update = 0 < 3)
        manager.compute_interest(1, &entities);
        assert!(manager.relevant_entities(1).unwrap().is_empty());

        // Tick 2, 3 still skip
        manager.compute_interest(1, &entities);
        manager.compute_interest(1, &entities);

        // Tick 3 should compute (ticks_since_update reaches 3)
        assert!(manager.is_relevant(1, 0));
    }

    #[test]
    fn client_lifecycle() {
        let mut manager = InterestManager::new(InterestConfig::default());

        manager.add_client(42, [0.0, 0.0, 0.0]);
        assert!(manager.clients.contains_key(&42));

        manager.update_position(42, [10.0, 0.0, 0.0]);
        assert_eq!(manager.clients[&42].position, [10.0, 0.0, 0.0]);

        manager.remove_client(42);
        assert!(!manager.clients.contains_key(&42));
    }
}
