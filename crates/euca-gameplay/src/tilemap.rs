//! Genre-agnostic tile map — data-driven 2D grids with square or hex topology.
//!
//! Resources: `TileMap`.
//! Components: `ResourcePool`.
//! Systems: `tile_income_system`.

use std::collections::HashMap;

use euca_ecs::{Entity, World};
use euca_math::Vec3;

// ── Topology ──

/// How tiles are arranged and which neighbors they have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    /// Four cardinal neighbors (N, E, S, W).
    Square4,
    /// Eight neighbors including diagonals.
    Square8,
    /// Six hex neighbors (pointy-top layout).
    Hex,
}

// ── TileData ──

/// Per-tile payload. Properties are arbitrary key-value pairs — no hardcoded terrain types.
///
/// Examples: `"movement_cost": 2.0`, `"food": 5.0`, `"terrain_id": 3.0`.
#[derive(Clone, Debug, Default)]
pub struct TileData {
    /// Arbitrary numeric properties (e.g. "food", "movement_cost").
    pub properties: HashMap<String, f64>,
    /// Which player (by index) owns this tile, if any.
    pub owner: Option<u8>,
    /// An entity placed on this tile, if any.
    pub entity: Option<Entity>,
}

// ── TileCoord ──

/// Integer tile coordinate. Origin is top-left (0,0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub x: i32,
    pub y: i32,
}

impl TileCoord {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Convert this tile coordinate to a world-space position (center of tile).
    ///
    /// - **Square**: straightforward `(x * cell_size, 0, y * cell_size)`.
    /// - **Hex** (pointy-top, offset-coordinates with even-row offset):
    ///   odd rows are shifted half a cell to the right.
    pub fn to_world(&self, map: &TileMap) -> Vec3 {
        let s = map.cell_size;
        match map.topology {
            Topology::Square4 | Topology::Square8 => {
                Vec3::new(self.x as f32 * s, 0.0, self.y as f32 * s)
            }
            Topology::Hex => {
                // Pointy-top hex, odd-row offset coordinates.
                // Horizontal spacing = cell_size, vertical spacing = cell_size * sqrt(3)/2.
                let vert = s * SQRT_3_OVER_2;
                let x_offset = if self.y & 1 != 0 { s * 0.5 } else { 0.0 };
                Vec3::new(self.x as f32 * s + x_offset, 0.0, self.y as f32 * vert)
            }
        }
    }

    /// Convert a world-space position to the nearest tile coordinate.
    pub fn from_world(pos: Vec3, map: &TileMap) -> TileCoord {
        let s = map.cell_size;
        match map.topology {
            Topology::Square4 | Topology::Square8 => {
                let x = (pos.x / s).round() as i32;
                let y = (pos.z / s).round() as i32;
                TileCoord { x, y }
            }
            Topology::Hex => {
                // Reverse the offset-coordinate mapping.
                let vert = s * SQRT_3_OVER_2;
                // Estimate row first from z.
                let approx_y = (pos.z / vert).round() as i32;
                let x_offset = if approx_y & 1 != 0 { s * 0.5 } else { 0.0 };
                let approx_x = ((pos.x - x_offset) / s).round() as i32;

                // Because rounding can be off by one row for hex grids,
                // check the approximate coord and its immediate vertical neighbors
                // and pick the closest one.
                let mut best = TileCoord::new(approx_x, approx_y);
                let mut best_dist = Self::world_dist_sq(best, pos, map);

                for candidate in nearby_hex_candidates(approx_x, approx_y) {
                    let d = Self::world_dist_sq(candidate, pos, map);
                    if d < best_dist {
                        best = candidate;
                        best_dist = d;
                    }
                }
                best
            }
        }
    }

    /// Convert to a flat array index into `TileMap::tiles`. Returns `None` if out of bounds.
    pub fn to_index(&self, map: &TileMap) -> Option<usize> {
        if self.x < 0 || self.y < 0 {
            return None;
        }
        let (ux, uy) = (self.x as u32, self.y as u32);
        if ux >= map.width || uy >= map.height {
            return None;
        }
        Some((uy * map.width + ux) as usize)
    }

    /// Squared distance between a tile's world center and a given position.
    fn world_dist_sq(coord: TileCoord, pos: Vec3, map: &TileMap) -> f32 {
        let center = coord.to_world(map);
        let dx = pos.x - center.x;
        let dz = pos.z - center.z;
        dx * dx + dz * dz
    }
}

/// sqrt(3) / 2 — used for hex vertical spacing.
const SQRT_3_OVER_2: f32 = 0.866_025_4;

/// Candidate offsets to check when snapping a world position to a hex tile.
fn nearby_hex_candidates(cx: i32, cy: i32) -> [TileCoord; 6] {
    // Check the six surrounding cells (one row above, same row, one row below).
    [
        TileCoord::new(cx - 1, cy),
        TileCoord::new(cx + 1, cy),
        TileCoord::new(cx, cy - 1),
        TileCoord::new(cx, cy + 1),
        TileCoord::new(cx - 1, cy + 1),
        TileCoord::new(cx + 1, cy - 1),
    ]
}

// ── TileMap ──

/// A 2D tile grid stored as a flat array. Acts as an ECS resource.
#[derive(Clone, Debug)]
pub struct TileMap {
    pub width: u32,
    pub height: u32,
    pub cell_size: f32,
    pub topology: Topology,
    pub tiles: Vec<TileData>,
}

impl TileMap {
    /// Create a new tile map filled with default (empty) tiles.
    pub fn new(width: u32, height: u32, cell_size: f32, topology: Topology) -> Self {
        let count = (width as usize) * (height as usize);
        Self {
            width,
            height,
            cell_size,
            topology,
            tiles: (0..count).map(|_| TileData::default()).collect(),
        }
    }

    /// Get a tile by coordinate. Returns `None` if out of bounds.
    pub fn get(&self, coord: TileCoord) -> Option<&TileData> {
        coord.to_index(self).map(|i| &self.tiles[i])
    }

    /// Get a mutable reference to a tile by coordinate. Returns `None` if out of bounds.
    pub fn get_mut(&mut self, coord: TileCoord) -> Option<&mut TileData> {
        coord.to_index(self).map(|i| &mut self.tiles[i])
    }

    /// Return the adjacent tile coordinates based on the map's topology.
    /// Only returns coordinates that are within bounds.
    pub fn neighbors(&self, coord: TileCoord) -> Vec<TileCoord> {
        let offsets: &[(i32, i32)] = match self.topology {
            Topology::Square4 => &[(0, -1), (1, 0), (0, 1), (-1, 0)],
            Topology::Square8 => &[
                (0, -1),
                (1, -1),
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
            ],
            Topology::Hex => {
                // Pointy-top hex, offset coordinates (odd-row shifted right).
                if coord.y & 1 == 0 {
                    &[(0, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (-1, 1)]
                } else {
                    &[(1, -1), (0, -1), (1, 0), (-1, 0), (1, 1), (0, 1)]
                }
            }
        };

        offsets
            .iter()
            .map(|&(dx, dy)| TileCoord::new(coord.x + dx, coord.y + dy))
            .filter(|c| c.to_index(self).is_some())
            .collect()
    }
}

// ── ResourcePool ──

/// Per-entity resource accumulation. Attach to a "player" entity to track income.
#[derive(Clone, Debug, Default)]
pub struct ResourcePool {
    pub resources: HashMap<String, f64>,
}

impl ResourcePool {
    pub fn new() -> Self {
        Self::default()
    }
}

// ── System ──

/// For each tile map, for each owned tile, add the tile's numeric properties
/// to the owner's `ResourcePool`, scaled by `dt`.
///
/// Owner mapping: `tile.owner` is a `u8` index. The system looks up the entity
/// at `owner_entities[index]` — a list you provide as a resource.
///
/// This avoids hardcoding any specific player-entity mapping.
pub fn tile_income_system(world: &mut World, dt: f32) {
    // Collect (owner_entity, properties) pairs while resources are borrowed,
    // then release the borrow before mutating entity components.
    let owned_tiles = {
        let map = match world.resource::<TileMap>() {
            Some(m) => m,
            None => return,
        };
        let owners = match world.resource::<TileOwnerTable>() {
            Some(t) => t,
            None => return,
        };

        let mut owned: Vec<(Entity, Vec<(String, f64)>)> = Vec::new();
        for tile in &map.tiles {
            if let Some(owner_idx) = tile.owner {
                if let Some(&entity) = owners.entities.get(owner_idx as usize) {
                    let props: Vec<(String, f64)> = tile
                        .properties
                        .iter()
                        .map(|(k, v)| (k.clone(), *v))
                        .collect();
                    owned.push((entity, props));
                }
            }
        }
        owned
    };

    // Apply income to each owner's ResourcePool.
    let dt64 = dt as f64;
    for (entity, props) in owned_tiles {
        if let Some(pool) = world.get_mut::<ResourcePool>(entity) {
            for (key, value) in props {
                *pool.resources.entry(key).or_insert(0.0) += value * dt64;
            }
        }
    }
}

/// Maps owner index (`u8`) to the `Entity` that owns tiles with that index.
///
/// Insert this as a resource alongside `TileMap` for `tile_income_system` to work.
#[derive(Clone, Debug, Default)]
pub struct TileOwnerTable {
    pub entities: Vec<Entity>,
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    // ── TileMap creation ──

    #[test]
    fn new_creates_correct_size() {
        let map = TileMap::new(10, 8, 1.0, Topology::Square4);
        assert_eq!(map.width, 10);
        assert_eq!(map.height, 8);
        assert_eq!(map.tiles.len(), 80);
    }

    #[test]
    fn default_tiles_are_empty() {
        let map = TileMap::new(2, 2, 1.0, Topology::Square4);
        for tile in &map.tiles {
            assert!(tile.properties.is_empty());
            assert!(tile.owner.is_none());
            assert!(tile.entity.is_none());
        }
    }

    // ── Coordinate conversion: square ──

    #[test]
    fn square_to_world_and_back() {
        let map = TileMap::new(10, 10, 2.0, Topology::Square4);
        let coord = TileCoord::new(3, 5);
        let world_pos = coord.to_world(&map);
        assert_eq!(world_pos.x, 6.0);
        assert_eq!(world_pos.y, 0.0);
        assert_eq!(world_pos.z, 10.0);

        let back = TileCoord::from_world(world_pos, &map);
        assert_eq!(back, coord);
    }

    #[test]
    fn square_from_world_rounds_correctly() {
        let map = TileMap::new(10, 10, 1.0, Topology::Square4);
        // Position (2.3, 0, 4.7) should snap to (2, 5).
        let coord = TileCoord::from_world(Vec3::new(2.3, 0.0, 4.7), &map);
        assert_eq!(coord, TileCoord::new(2, 5));
    }

    // ── Coordinate conversion: hex ──

    #[test]
    fn hex_to_world_and_back() {
        let map = TileMap::new(10, 10, 1.0, Topology::Hex);
        // Even row — no offset.
        let c0 = TileCoord::new(3, 0);
        let w0 = c0.to_world(&map);
        assert!((w0.x - 3.0).abs() < 1e-5);
        assert!((w0.z - 0.0).abs() < 1e-5);
        assert_eq!(TileCoord::from_world(w0, &map), c0);

        // Odd row — shifted right by 0.5.
        let c1 = TileCoord::new(3, 1);
        let w1 = c1.to_world(&map);
        assert!((w1.x - 3.5).abs() < 1e-5);
        assert_eq!(TileCoord::from_world(w1, &map), c1);
    }

    #[test]
    fn hex_roundtrip_multiple_coords() {
        let map = TileMap::new(8, 8, 1.5, Topology::Hex);
        for y in 0..8 {
            for x in 0..8 {
                let coord = TileCoord::new(x, y);
                let world_pos = coord.to_world(&map);
                let back = TileCoord::from_world(world_pos, &map);
                assert_eq!(back, coord, "roundtrip failed for ({x}, {y})");
            }
        }
    }

    // ── Bounds checking ──

    #[test]
    fn to_index_in_bounds() {
        let map = TileMap::new(5, 5, 1.0, Topology::Square4);
        assert_eq!(TileCoord::new(0, 0).to_index(&map), Some(0));
        assert_eq!(TileCoord::new(4, 4).to_index(&map), Some(24));
        assert_eq!(TileCoord::new(2, 3).to_index(&map), Some(17));
    }

    #[test]
    fn to_index_out_of_bounds() {
        let map = TileMap::new(5, 5, 1.0, Topology::Square4);
        assert_eq!(TileCoord::new(-1, 0).to_index(&map), None);
        assert_eq!(TileCoord::new(0, -1).to_index(&map), None);
        assert_eq!(TileCoord::new(5, 0).to_index(&map), None);
        assert_eq!(TileCoord::new(0, 5).to_index(&map), None);
    }

    #[test]
    fn get_out_of_bounds_returns_none() {
        let map = TileMap::new(3, 3, 1.0, Topology::Square4);
        assert!(map.get(TileCoord::new(-1, 0)).is_none());
        assert!(map.get(TileCoord::new(3, 0)).is_none());
    }

    // ── Tile properties ──

    #[test]
    fn tile_properties_read_write() {
        let mut map = TileMap::new(3, 3, 1.0, Topology::Square4);
        let coord = TileCoord::new(1, 1);
        let tile = map.get_mut(coord).unwrap();
        tile.properties.insert("food".into(), 5.0);
        tile.properties.insert("movement_cost".into(), 2.0);
        tile.owner = Some(0);

        let tile = map.get(coord).unwrap();
        assert_eq!(tile.properties["food"], 5.0);
        assert_eq!(tile.properties["movement_cost"], 2.0);
        assert_eq!(tile.owner, Some(0));
    }

    // ── Adjacency: Square4 ──

    #[test]
    fn square4_center_has_4_neighbors() {
        let map = TileMap::new(5, 5, 1.0, Topology::Square4);
        let n = map.neighbors(TileCoord::new(2, 2));
        assert_eq!(n.len(), 4);
        assert!(n.contains(&TileCoord::new(2, 1))); // N
        assert!(n.contains(&TileCoord::new(3, 2))); // E
        assert!(n.contains(&TileCoord::new(2, 3))); // S
        assert!(n.contains(&TileCoord::new(1, 2))); // W
    }

    #[test]
    fn square4_corner_has_2_neighbors() {
        let map = TileMap::new(5, 5, 1.0, Topology::Square4);
        let n = map.neighbors(TileCoord::new(0, 0));
        assert_eq!(n.len(), 2);
    }

    // ── Adjacency: Square8 ──

    #[test]
    fn square8_center_has_8_neighbors() {
        let map = TileMap::new(5, 5, 1.0, Topology::Square8);
        let n = map.neighbors(TileCoord::new(2, 2));
        assert_eq!(n.len(), 8);
    }

    #[test]
    fn square8_corner_has_3_neighbors() {
        let map = TileMap::new(5, 5, 1.0, Topology::Square8);
        let n = map.neighbors(TileCoord::new(0, 0));
        assert_eq!(n.len(), 3);
    }

    // ── Adjacency: Hex ──

    #[test]
    fn hex_center_has_6_neighbors() {
        let map = TileMap::new(8, 8, 1.0, Topology::Hex);
        // Even row center
        let n = map.neighbors(TileCoord::new(3, 2));
        assert_eq!(n.len(), 6);
        // Odd row center
        let n = map.neighbors(TileCoord::new(3, 3));
        assert_eq!(n.len(), 6);
    }

    #[test]
    fn hex_corner_has_fewer_neighbors() {
        let map = TileMap::new(5, 5, 1.0, Topology::Hex);
        let n = map.neighbors(TileCoord::new(0, 0));
        // (0,0) even row: offsets are (0,-1),(-1,-1),(1,0),(-1,0),(0,1),(-1,1)
        // Valid: (1,0) and (0,1) → 2 neighbors
        assert!(n.len() < 6);
        assert!(n.len() >= 2);
    }

    #[test]
    fn hex_neighbors_are_symmetric() {
        let map = TileMap::new(8, 8, 1.0, Topology::Hex);
        let a = TileCoord::new(3, 2);
        for b in map.neighbors(a) {
            let b_neighbors = map.neighbors(b);
            assert!(
                b_neighbors.contains(&a),
                "{a:?} lists {b:?} as neighbor, but {b:?} does not list {a:?}"
            );
        }
    }

    // ── Resource income system ──

    #[test]
    fn tile_income_accumulates_resources() {
        let mut world = World::new();

        // Create a player entity with a ResourcePool.
        let player = world.spawn(ResourcePool::new());

        // Set up the tile map with one owned tile producing food.
        let mut map = TileMap::new(3, 3, 1.0, Topology::Square4);
        map.get_mut(TileCoord::new(1, 1)).unwrap().owner = Some(0);
        map.get_mut(TileCoord::new(1, 1))
            .unwrap()
            .properties
            .insert("food".into(), 10.0);

        world.insert_resource(map);
        world.insert_resource(TileOwnerTable {
            entities: vec![player],
        });

        // Run one second of income.
        tile_income_system(&mut world, 1.0);

        let pool = world.get::<ResourcePool>(player).unwrap();
        assert!((pool.resources["food"] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn tile_income_scales_with_dt() {
        let mut world = World::new();
        let player = world.spawn(ResourcePool::new());

        let mut map = TileMap::new(2, 2, 1.0, Topology::Square4);
        map.get_mut(TileCoord::new(0, 0)).unwrap().owner = Some(0);
        map.get_mut(TileCoord::new(0, 0))
            .unwrap()
            .properties
            .insert("gold".into(), 100.0);

        world.insert_resource(map);
        world.insert_resource(TileOwnerTable {
            entities: vec![player],
        });

        // Half-second tick.
        tile_income_system(&mut world, 0.5);

        let pool = world.get::<ResourcePool>(player).unwrap();
        assert!((pool.resources["gold"] - 50.0).abs() < 1e-9);
    }

    #[test]
    fn tile_income_no_owner_no_income() {
        let mut world = World::new();
        let player = world.spawn(ResourcePool::new());

        let mut map = TileMap::new(2, 2, 1.0, Topology::Square4);
        // Tile has properties but no owner.
        map.get_mut(TileCoord::new(0, 0))
            .unwrap()
            .properties
            .insert("food".into(), 10.0);

        world.insert_resource(map);
        world.insert_resource(TileOwnerTable {
            entities: vec![player],
        });

        tile_income_system(&mut world, 1.0);

        let pool = world.get::<ResourcePool>(player).unwrap();
        assert!(pool.resources.is_empty());
    }

    #[test]
    fn tile_income_without_tilemap_is_noop() {
        let mut world = World::new();
        // No TileMap resource — should not panic.
        tile_income_system(&mut world, 1.0);
    }

    #[test]
    fn tile_income_multiple_tiles_same_owner() {
        let mut world = World::new();
        let player = world.spawn(ResourcePool::new());

        let mut map = TileMap::new(3, 3, 1.0, Topology::Square4);
        for y in 0..3 {
            for x in 0..3 {
                let tile = map.get_mut(TileCoord::new(x, y)).unwrap();
                tile.owner = Some(0);
                tile.properties.insert("wood".into(), 1.0);
            }
        }

        world.insert_resource(map);
        world.insert_resource(TileOwnerTable {
            entities: vec![player],
        });

        tile_income_system(&mut world, 1.0);

        let pool = world.get::<ResourcePool>(player).unwrap();
        assert!((pool.resources["wood"] - 9.0).abs() < 1e-9);
    }
}
