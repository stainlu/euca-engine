//! Interest management: only replicate entities near each client.
//!
//! Uses a [`SpatialGrid`] to accelerate proximity queries from O(n) to O(k),
//! where k is the number of nearby entities rather than the total entity count.

use euca_ecs::{Entity, Query, World};
use euca_scene::GlobalTransform;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// An entity index paired with its 3D position.
type GridEntry = (u32, [f32; 3]);

/// A 3D uniform-grid spatial index for fast proximity queries over entity
/// indices.
///
/// Designed for the networking interest system where entities are identified
/// by `u32` indices (not full [`Entity`] handles). Positions are stored
/// alongside indices so that radius queries can perform exact distance checks
/// after the coarse cell scan.
///
/// The grid is rebuilt every evaluation tick via [`SpatialGrid::rebuild`].
#[derive(Clone, Debug)]
pub struct SpatialGrid {
    cell_size: f32,
    /// Reciprocal of `cell_size`, cached to avoid division in hot paths.
    inv_cell_size: f32,
    /// Maps 3D cell coordinates to the entities within that cell.
    cells: HashMap<(i32, i32, i32), Vec<GridEntry>>,
    /// Cached total entity count for O(1) access.
    len: usize,
}

impl SpatialGrid {
    /// Create a new spatial grid with the given cell size.
    ///
    /// # Panics
    /// Panics if `cell_size` is not positive.
    pub fn new(cell_size: f32) -> Self {
        assert!(cell_size > 0.0, "cell_size must be positive");
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
            len: 0,
        }
    }

    /// The cell size this grid was configured with.
    #[inline]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Compute the cell key for a world-space position.
    ///
    /// Components are clamped to `(i32::MIN / 2)..=(i32::MAX / 2)` after
    /// flooring to prevent overflow at extreme coordinates.
    #[inline]
    fn cell_key(&self, pos: [f32; 3]) -> (i32, i32, i32) {
        const LO: i32 = i32::MIN / 2;
        const HI: i32 = i32::MAX / 2;
        (
            ((pos[0] * self.inv_cell_size).floor() as i32).clamp(LO, HI),
            ((pos[1] * self.inv_cell_size).floor() as i32).clamp(LO, HI),
            ((pos[2] * self.inv_cell_size).floor() as i32).clamp(LO, HI),
        )
    }

    /// Clear all cells and re-insert entities from `positions`.
    ///
    /// Each entry is `(entity_index, [x, y, z])`.
    pub fn rebuild(&mut self, positions: &[(u32, [f32; 3])]) {
        self.cells.clear();
        self.len = positions.len();
        for &(entity_idx, pos) in positions {
            let key = self.cell_key(pos);
            self.cells.entry(key).or_default().push((entity_idx, pos));
        }
    }

    /// Clear all cells and re-insert entities from flat-tuple positions.
    ///
    /// Each entry is `(entity_index, x, y, z)`. This avoids an intermediate
    /// allocation when the caller already has positions in this layout.
    pub fn rebuild_from_tuples(&mut self, positions: &[(u32, f32, f32, f32)]) {
        self.cells.clear();
        self.len = positions.len();
        for &(entity_idx, x, y, z) in positions {
            let pos = [x, y, z];
            let key = self.cell_key(pos);
            self.cells.entry(key).or_default().push((entity_idx, pos));
        }
    }

    /// Return all entity indices within `radius` of `center`.
    ///
    /// Scans only the grid cells that overlap the bounding box of the query
    /// sphere, then filters by squared Euclidean distance for exactness.
    pub fn query_radius(&self, center: [f32; 3], radius: f32) -> Vec<u32> {
        let radius_sq = radius * radius;

        let min = [center[0] - radius, center[1] - radius, center[2] - radius];
        let max = [center[0] + radius, center[1] + radius, center[2] + radius];
        let min_key = self.cell_key(min);
        let max_key = self.cell_key(max);

        let mut result = Vec::new();

        for cx in min_key.0..=max_key.0 {
            for cy in min_key.1..=max_key.1 {
                for cz in min_key.2..=max_key.2 {
                    if let Some(entries) = self.cells.get(&(cx, cy, cz)) {
                        for &(entity_idx, pos) in entries {
                            let dx = pos[0] - center[0];
                            let dy = pos[1] - center[1];
                            let dz = pos[2] - center[2];
                            if dx * dx + dy * dy + dz * dz <= radius_sq {
                                result.push(entity_idx);
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// The total number of entities currently indexed.
    #[inline]
    pub fn entity_count(&self) -> usize {
        self.len
    }

    /// The number of non-empty grid cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }
}

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
///
/// Holds a [`SpatialGrid`] that is rebuilt each evaluation tick to accelerate
/// proximity queries from O(n) to O(k).
#[derive(Clone, Debug)]
pub struct InterestManager {
    pub config: InterestConfig,
    /// Per-client interest data, keyed by client ID.
    pub clients: HashMap<u32, ClientInterest>,
    /// Spatial acceleration structure, rebuilt each evaluation tick.
    grid: SpatialGrid,
}

impl Default for InterestManager {
    fn default() -> Self {
        Self::new(InterestConfig::default())
    }
}

impl InterestManager {
    pub fn new(config: InterestConfig) -> Self {
        let cell_size = config.max_relevance_distance / 2.0;
        Self {
            grid: SpatialGrid::new(cell_size),
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

    /// Rebuild the spatial grid from a list of entity positions.
    ///
    /// Call this once before computing interest for all clients in a tick.
    /// Each entry is `(entity_index, x, y, z)`.
    pub fn rebuild_grid(&mut self, entity_positions: &[(u32, f32, f32, f32)]) {
        self.grid.rebuild_from_tuples(entity_positions);
    }

    /// Compute which entities are relevant for a client based on distance.
    ///
    /// Uses the spatial grid for O(k) lookup. The grid must be populated
    /// via [`rebuild_grid`] before calling this.
    pub fn compute_interest(&mut self, client_id: u32) {
        let max_dist = self.config.max_relevance_distance;

        if let Some(interest) = self.clients.get_mut(&client_id) {
            interest.ticks_since_update += 1;
            if interest.ticks_since_update < self.config.update_interval {
                return;
            }
            interest.ticks_since_update = 0;

            let nearby = self.grid.query_radius(interest.position, max_dist);
            interest.relevant_entities.clear();
            interest.relevant_entities.extend(nearby);
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

    /// Access the spatial grid.
    pub fn grid(&self) -> &SpatialGrid {
        &self.grid
    }
}

/// System: update interest sets for all clients using world entity positions.
pub fn interest_culling_system(world: &mut World) {
    let positions: Vec<(u32, f32, f32, f32)> = {
        let query = Query::<(Entity, &crate::protocol::Replicated)>::new(world);
        query
            .iter()
            .map(|(e, _)| {
                let (x, y, z) = match world.get::<GlobalTransform>(e) {
                    Some(gt) => {
                        let t = &gt.0.translation;
                        (t.x, t.y, t.z)
                    }
                    None => (0.0, 0.0, 0.0),
                };
                (e.index(), x, y, z)
            })
            .collect()
    };

    if let Some(manager) = world.resource_mut::<InterestManager>() {
        manager.rebuild_grid(&positions);

        let client_ids: Vec<u32> = manager.clients.keys().copied().collect();
        for client_id in client_ids {
            manager.compute_interest(client_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_insertion_and_entity_count() {
        let mut grid = SpatialGrid::new(10.0);
        let positions = vec![
            (0, [0.0, 0.0, 0.0]),
            (1, [5.0, 5.0, 5.0]),
            (2, [100.0, 0.0, 0.0]),
        ];
        grid.rebuild(&positions);
        assert_eq!(grid.entity_count(), 3);
        assert!(grid.cell_count() >= 1);
    }

    #[test]
    fn grid_radius_query_finds_nearby() {
        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&[
            (0, [0.0, 0.0, 0.0]),
            (1, [5.0, 0.0, 0.0]),
            (2, [100.0, 0.0, 0.0]),
        ]);

        let result = grid.query_radius([0.0, 0.0, 0.0], 6.0);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
        assert!(!result.contains(&2));
    }

    #[test]
    fn grid_radius_query_excludes_outside_sphere() {
        let mut grid = SpatialGrid::new(5.0);
        // Distance = sqrt(6^2 + 6^2 + 6^2) = sqrt(108) ~= 10.39
        grid.rebuild(&[(0, [6.0, 6.0, 6.0])]);

        let result = grid.query_radius([0.0, 0.0, 0.0], 10.0);
        assert!(
            !result.contains(&0),
            "entity at distance ~10.39 should be outside radius 10.0"
        );
    }

    #[test]
    fn grid_entity_on_cell_boundary() {
        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&[(0, [10.0, 0.0, 0.0])]);

        let result = grid.query_radius([0.0, 0.0, 0.0], 10.0);
        assert!(
            result.contains(&0),
            "entity exactly at radius boundary should be included"
        );
    }

    #[test]
    fn grid_empty_returns_nothing() {
        let grid = SpatialGrid::new(10.0);
        let result = grid.query_radius([0.0, 0.0, 0.0], 100.0);
        assert!(result.is_empty());
        assert_eq!(grid.entity_count(), 0);
        assert_eq!(grid.cell_count(), 0);
    }

    #[test]
    fn grid_large_radius_covers_many_cells() {
        let mut grid = SpatialGrid::new(5.0);
        let positions: Vec<(u32, [f32; 3])> =
            (0..100).map(|i| (i, [i as f32 * 3.0, 0.0, 0.0])).collect();
        grid.rebuild(&positions);

        let result = grid.query_radius([150.0, 0.0, 0.0], 200.0);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn grid_rebuild_clears_previous() {
        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&[(0, [0.0, 0.0, 0.0])]);
        assert_eq!(grid.entity_count(), 1);

        grid.rebuild(&[]);
        assert_eq!(grid.entity_count(), 0);
        assert_eq!(grid.cell_count(), 0);
    }

    #[test]
    fn grid_negative_coordinates() {
        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&[(0, [-50.0, -50.0, -50.0]), (1, [50.0, 50.0, 50.0])]);

        let result = grid.query_radius([-50.0, -50.0, -50.0], 5.0);
        assert!(result.contains(&0));
        assert!(!result.contains(&1));
    }

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

        manager.rebuild_grid(&entities);
        manager.compute_interest(1);

        assert!(manager.is_relevant(1, 0));
        assert!(!manager.is_relevant(1, 1));
        assert!(manager.is_relevant(1, 2));
    }

    #[test]
    fn interest_empty_world_yields_no_relevance() {
        let mut manager = InterestManager::new(InterestConfig {
            max_relevance_distance: 10.0,
            update_interval: 0,
        });

        manager.add_client(1, [0.0, 0.0, 0.0]);

        manager.rebuild_grid(&[]);
        manager.compute_interest(1);

        assert!(manager.relevant_entities(1).unwrap().is_empty());
    }

    #[test]
    fn update_interval_skips() {
        let mut manager = InterestManager::new(InterestConfig {
            max_relevance_distance: 10.0,
            update_interval: 3,
        });

        manager.add_client(1, [0.0, 0.0, 0.0]);

        let entities = vec![(0, 5.0, 0.0, 0.0)];

        manager.rebuild_grid(&entities);
        manager.compute_interest(1);
        assert!(manager.relevant_entities(1).unwrap().is_empty());

        manager.compute_interest(1);
        manager.compute_interest(1);

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

    #[test]
    fn spatial_grid_cell_size_matches_config() {
        let config = InterestConfig {
            max_relevance_distance: 80.0,
            update_interval: 0,
        };
        let manager = InterestManager::new(config);
        assert!((manager.grid().cell_size() - 40.0).abs() < f32::EPSILON);
    }

    #[test]
    fn multiple_clients_share_same_grid() {
        let mut manager = InterestManager::new(InterestConfig {
            max_relevance_distance: 10.0,
            update_interval: 0,
        });

        manager.add_client(1, [0.0, 0.0, 0.0]);
        manager.add_client(2, [50.0, 0.0, 0.0]);

        let entities = vec![
            (0, 5.0, 0.0, 0.0),   // near client 1
            (1, 48.0, 0.0, 0.0),  // near client 2
            (2, 200.0, 0.0, 0.0), // near nobody
        ];

        manager.rebuild_grid(&entities);
        let client_ids: Vec<u32> = manager.clients.keys().copied().collect();
        for id in client_ids {
            manager.compute_interest(id);
        }

        assert!(manager.is_relevant(1, 0));
        assert!(!manager.is_relevant(1, 1));
        assert!(!manager.is_relevant(1, 2));

        assert!(!manager.is_relevant(2, 0));
        assert!(manager.is_relevant(2, 1));
        assert!(!manager.is_relevant(2, 2));
    }
}
