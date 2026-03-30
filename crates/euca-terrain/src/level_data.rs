//! Universal level data format for the Euca engine.
//!
//! [`LevelData`] describes everything a level needs in a single, serialisable
//! struct: terrain geometry, surface types, walkability, entity placements,
//! trigger volumes, camera defaults, and navigation configuration.  It is the
//! canonical interchange format between the editor, runtime, and asset
//! pipeline.
//!
//! Supports both human-readable (JSON) and compact binary (bincode) encoding.

use std::collections::HashMap;

use euca_math::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::Heightmap;

// ── Surface & terrain mode ──────────────────────────────────────────────────

/// The physical/visual surface material applied to a terrain cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SurfaceType {
    Grass,
    Dirt,
    Stone,
    Water,
    Sand,
    Snow,
    Mud,
    Road,
    Cliff,
    Void,
    /// Application-defined surface type identified by a numeric tag.
    Custom(u16),
}

impl Default for SurfaceType {
    fn default() -> Self {
        Self::Grass
    }
}

/// How the terrain geometry is represented.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerrainMode {
    /// Classic heightfield terrain.
    #[default]
    Heightmap,
    /// Discrete tile grid (e.g. strategy or RPG maps).
    Tiles,
    /// Flat polygonal zones with optional elevation offsets.
    FlatZones,
    /// Arbitrary static mesh used as terrain.
    StaticMesh,
    /// No terrain — pure sky/void levels.
    None,
}

// ── Placement types ─────────────────────────────────────────────────────────

/// An entity placed in the level.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntityPlacement {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// Identifies the kind of entity (e.g. `"tree_oak"`, `"enemy_grunt"`).
    pub entity_type: String,
    /// Arbitrary key-value properties interpreted by gameplay systems.
    pub properties: HashMap<String, String>,
}

/// A trigger volume placed in the level.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerPlacement {
    pub position: Vec3,
    pub radius: f32,
    /// Identifies the kind of trigger (e.g. `"checkpoint"`, `"damage_zone"`).
    pub trigger_type: String,
    /// Arbitrary key-value properties interpreted by gameplay systems.
    pub properties: HashMap<String, String>,
}

// ── Config structs ──────────────────────────────────────────────────────────

/// Navigation mesh / pathfinding configuration stored per-level.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NavConfig {
    /// Maximum slope (in degrees) that agents can traverse.
    pub max_slope_degrees: f32,
    /// Minimum vertical clearance required for an agent.
    pub agent_height: f32,
    /// Horizontal radius of the navigation agent capsule.
    pub agent_radius: f32,
    /// Maximum step height an agent can climb without jumping.
    pub step_height: f32,
}

impl Default for NavConfig {
    fn default() -> Self {
        Self {
            max_slope_degrees: 45.0,
            agent_height: 1.8,
            agent_radius: 0.3,
            step_height: 0.4,
        }
    }
}

/// Default camera parameters baked into the level.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraConfig {
    pub position: Vec3,
    pub look_at: Vec3,
    /// Vertical field-of-view in degrees.
    pub fov_degrees: f32,
    pub near_plane: f32,
    pub far_plane: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 10.0, -10.0),
            look_at: Vec3::ZERO,
            fov_degrees: 60.0,
            near_plane: 0.1,
            far_plane: 1000.0,
        }
    }
}

// ── LevelData ───────────────────────────────────────────────────────────────

/// The universal, serialisable description of a level.
///
/// Five data layers live side-by-side on the same `width * height` grid:
///
/// | Layer        | Storage              | Per-cell meaning                |
/// |--------------|----------------------|---------------------------------|
/// | heightmap    | `Vec<f32>`           | Normalised elevation `[0, 1]`   |
/// | surface      | `Vec<SurfaceType>`   | Physical surface material       |
/// | walkable     | `Vec<bool>`          | Can an agent stand here?        |
/// | entities     | `Vec<EntityPlacement>` | Objects placed in the level   |
/// | triggers     | `Vec<TriggerPlacement>` | Volume triggers              |
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LevelData {
    // ── Grid dimensions ──
    /// Number of columns (X axis).
    pub width: u32,
    /// Number of rows (Z axis).
    pub height: u32,
    /// World-space distance between adjacent grid cells.
    pub cell_size: f32,

    // ── Terrain layer ──
    pub terrain_mode: TerrainMode,
    /// Row-major elevation data, each value in `[0, 1]`.
    pub heightmap: Vec<f32>,

    // ── Surface layer ──
    pub surface: Vec<SurfaceType>,

    // ── Walkability layer ──
    pub walkable: Vec<bool>,

    // ── Entity layer ──
    pub entities: Vec<EntityPlacement>,

    // ── Trigger layer ──
    pub triggers: Vec<TriggerPlacement>,

    // ── Global configs ──
    pub camera: CameraConfig,
    pub nav_config: NavConfig,

    /// Free-form metadata (level name, author, version, etc.).
    pub metadata: HashMap<String, String>,
}

impl LevelData {
    /// Create an empty level of the given grid dimensions.
    ///
    /// All heights are zero, all surfaces default to [`SurfaceType::Grass`],
    /// and every cell is walkable.
    pub fn new(width: u32, height: u32, cell_size: f32) -> Self {
        let count = (width as usize) * (height as usize);
        Self {
            width,
            height,
            cell_size,
            terrain_mode: TerrainMode::default(),
            heightmap: vec![0.0; count],
            surface: vec![SurfaceType::default(); count],
            walkable: vec![true; count],
            entities: Vec::new(),
            triggers: Vec::new(),
            camera: CameraConfig::default(),
            nav_config: NavConfig::default(),
            metadata: HashMap::new(),
        }
    }

    // ── Dimensions ──────────────────────────────────────────────────────

    /// Total number of cells in the grid.
    #[inline]
    pub fn cell_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Total world-space extent along the X axis.
    #[inline]
    pub fn world_width(&self) -> f32 {
        self.width.saturating_sub(1) as f32 * self.cell_size
    }

    /// Total world-space extent along the Z axis.
    #[inline]
    pub fn world_depth(&self) -> f32 {
        self.height.saturating_sub(1) as f32 * self.cell_size
    }

    // ── Cell access ─────────────────────────────────────────────────────

    /// Row-major index for the cell at `(col, row)`, or `None` if out of bounds.
    #[inline]
    fn cell_index(&self, col: u32, row: u32) -> Option<usize> {
        if col < self.width && row < self.height {
            Some((row as usize) * (self.width as usize) + (col as usize))
        } else {
            None
        }
    }

    /// Height at integer grid coordinates, or `0.0` if out of bounds.
    #[inline]
    pub fn height_at(&self, col: u32, row: u32) -> f32 {
        self.cell_index(col, row)
            .map(|i| self.heightmap[i])
            .unwrap_or(0.0)
    }

    /// Surface type at integer grid coordinates, or [`SurfaceType::Void`] if
    /// out of bounds.
    #[inline]
    pub fn surface_at(&self, col: u32, row: u32) -> SurfaceType {
        self.cell_index(col, row)
            .map(|i| self.surface[i])
            .unwrap_or(SurfaceType::Void)
    }

    /// Whether the cell is walkable, or `false` if out of bounds.
    #[inline]
    pub fn is_walkable(&self, col: u32, row: u32) -> bool {
        self.cell_index(col, row)
            .map(|i| self.walkable[i])
            .unwrap_or(false)
    }

    // ── Coordinate conversions ──────────────────────────────────────────

    /// Convert a world-space `(x, z)` position to the nearest grid cell.
    ///
    /// Returns `None` when the position falls outside the grid.
    pub fn world_to_grid(&self, x: f32, z: f32) -> Option<(u32, u32)> {
        if self.cell_size <= 0.0 {
            return None;
        }
        let col = (x / self.cell_size).round();
        let row = (z / self.cell_size).round();
        if col < 0.0 || row < 0.0 {
            return None;
        }
        let col = col as u32;
        let row = row as u32;
        if col < self.width && row < self.height {
            Some((col, row))
        } else {
            None
        }
    }

    /// Convert grid coordinates to the world-space centre of that cell.
    pub fn grid_to_world(&self, col: u32, row: u32) -> Option<Vec3> {
        if col >= self.width || row >= self.height {
            return None;
        }
        let x = col as f32 * self.cell_size;
        let y = self.height_at(col, row);
        let z = row as f32 * self.cell_size;
        Some(Vec3::new(x, y, z))
    }

    // ── Generators ──────────────────────────────────────────────────────

    /// Recompute the walkability grid from a height-slope threshold.
    ///
    /// A cell is marked walkable when:
    /// 1. Its surface type is not [`SurfaceType::Water`] or [`SurfaceType::Void`].
    /// 2. The absolute height difference to every cardinal neighbour is at most
    ///    `max_slope`.
    pub fn generate_walkability_grid(&mut self, max_slope: f32) {
        let w = self.width;
        let h = self.height;
        let mut grid = vec![true; self.cell_count()];

        for row in 0..h {
            for col in 0..w {
                let idx = (row as usize) * (w as usize) + (col as usize);
                let surface = self.surface[idx];
                if surface == SurfaceType::Water || surface == SurfaceType::Void {
                    grid[idx] = false;
                    continue;
                }

                let center = self.heightmap[idx];
                let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dc, dr) in neighbours {
                    let nc = col as i64 + dc;
                    let nr = row as i64 + dr;
                    if nc >= 0 && nc < w as i64 && nr >= 0 && nr < h as i64 {
                        let ni = (nr as usize) * (w as usize) + (nc as usize);
                        if (self.heightmap[ni] - center).abs() > max_slope {
                            grid[idx] = false;
                            break;
                        }
                    }
                }
            }
        }

        self.walkable = grid;
    }

    // ── Conversion helpers ──────────────────────────────────────────────

    /// Build a [`Heightmap`] from this level's elevation data.
    pub fn to_heightmap(&self) -> Heightmap {
        Heightmap::from_raw(self.width, self.height, self.heightmap.clone())
            .with_cell_size(self.cell_size)
    }

    // ── Serialisation ───────────────────────────────────────────────────

    /// Encode to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Decode from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Encode to a compact binary representation (bincode).
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Decode from a binary representation (bincode).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_level() -> LevelData {
        let mut level = LevelData::new(4, 4, 2.0);
        level.heightmap[0] = 0.5;
        level.heightmap[5] = 1.0;
        level.surface[3] = SurfaceType::Water;
        level.walkable[3] = false;
        level.metadata.insert("name".into(), "test_arena".into());
        level
    }

    // ── Construction & dimensions ───────────────────────────────────────

    #[test]
    fn new_creates_correct_size() {
        let level = LevelData::new(8, 6, 1.5);
        assert_eq!(level.cell_count(), 48);
        assert_eq!(level.heightmap.len(), 48);
        assert_eq!(level.surface.len(), 48);
        assert_eq!(level.walkable.len(), 48);
    }

    #[test]
    fn world_extents() {
        let level = LevelData::new(5, 3, 2.0);
        assert!((level.world_width() - 8.0).abs() < 1e-6); // (5-1)*2
        assert!((level.world_depth() - 4.0).abs() < 1e-6); // (3-1)*2
    }

    #[test]
    fn world_extents_single_cell() {
        let level = LevelData::new(1, 1, 5.0);
        assert!((level.world_width() - 0.0).abs() < 1e-6);
        assert!((level.world_depth() - 0.0).abs() < 1e-6);
    }

    // ── Cell access ─────────────────────────────────────────────────────

    #[test]
    fn height_at_in_bounds() {
        let level = sample_level();
        assert!((level.height_at(0, 0) - 0.5).abs() < 1e-6);
        assert!((level.height_at(1, 1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn height_at_out_of_bounds() {
        let level = sample_level();
        assert!((level.height_at(99, 99) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn surface_at_in_bounds() {
        let level = sample_level();
        assert_eq!(level.surface_at(0, 0), SurfaceType::Grass);
        assert_eq!(level.surface_at(3, 0), SurfaceType::Water);
    }

    #[test]
    fn surface_at_out_of_bounds() {
        let level = sample_level();
        assert_eq!(level.surface_at(99, 0), SurfaceType::Void);
    }

    #[test]
    fn is_walkable_returns_false_for_water() {
        let level = sample_level();
        assert!(!level.is_walkable(3, 0));
    }

    #[test]
    fn is_walkable_out_of_bounds() {
        let level = sample_level();
        assert!(!level.is_walkable(99, 99));
    }

    // ── Coordinate conversions ──────────────────────────────────────────

    #[test]
    fn world_to_grid_roundtrip() {
        let level = LevelData::new(8, 8, 2.0);
        assert_eq!(level.world_to_grid(4.0, 6.0), Some((2, 3)));
    }

    #[test]
    fn world_to_grid_out_of_bounds() {
        let level = LevelData::new(4, 4, 1.0);
        assert_eq!(level.world_to_grid(-1.0, 0.0), None);
        assert_eq!(level.world_to_grid(0.0, 10.0), None);
    }

    #[test]
    fn grid_to_world_basic() {
        let level = LevelData::new(4, 4, 3.0);
        let pos = level.grid_to_world(2, 1).unwrap();
        assert!((pos.x - 6.0).abs() < 1e-6);
        assert!((pos.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn grid_to_world_out_of_bounds() {
        let level = LevelData::new(4, 4, 1.0);
        assert!(level.grid_to_world(10, 0).is_none());
    }

    // ── Walkability generation ──────────────────────────────────────────

    #[test]
    fn generate_walkability_marks_water() {
        let mut level = LevelData::new(3, 3, 1.0);
        level.surface[4] = SurfaceType::Water;
        level.generate_walkability_grid(10.0);
        assert!(!level.is_walkable(1, 1));
        assert!(level.is_walkable(0, 0));
    }

    #[test]
    fn generate_walkability_slope_threshold() {
        let mut level = LevelData::new(3, 3, 1.0);
        // Create a steep height cliff at centre.
        level.heightmap[4] = 5.0;
        level.generate_walkability_grid(1.0);
        // Centre's neighbours all differ by >1.0, so centre is unwalkable.
        assert!(!level.is_walkable(1, 1));
        // Corner (0,0) has a neighbour at (1,0)=0 and (0,1)=0, both flat.
        // But (1,0) borders centre 5.0 — however we only check the cell
        // itself vs its neighbours, so (0,0) whose neighbour (1,0) has
        // height 0.0 is fine.
        assert!(level.is_walkable(0, 0));
    }

    // ── Heightmap conversion ────────────────────────────────────────────

    #[test]
    fn to_heightmap_preserves_data() {
        let level = sample_level();
        let hm = level.to_heightmap();
        assert_eq!(hm.width, level.width);
        assert_eq!(hm.height, level.height);
        assert!((hm.cell_size - level.cell_size).abs() < 1e-6);
        assert!((hm.data[0] - 0.5).abs() < 1e-6);
    }

    // ── JSON round-trip ─────────────────────────────────────────────────

    #[test]
    fn json_roundtrip() {
        let level = sample_level();
        let json = level.to_json().expect("serialize");
        let restored = LevelData::from_json(&json).expect("deserialize");
        assert_eq!(level, restored);
    }

    // ── Bincode round-trip ──────────────────────────────────────────────

    #[test]
    fn bincode_roundtrip() {
        let level = sample_level();
        let bytes = level.to_bytes().expect("serialize");
        let restored = LevelData::from_bytes(&bytes).expect("deserialize");
        assert_eq!(level, restored);
    }

    // ── Entity & trigger placements ─────────────────────────────────────

    #[test]
    fn entity_placement_serde() {
        let entity = EntityPlacement {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            entity_type: "tree_oak".into(),
            properties: HashMap::from([("health".into(), "100".into())]),
        };
        let json = serde_json::to_string(&entity).unwrap();
        let restored: EntityPlacement = serde_json::from_str(&json).unwrap();
        assert_eq!(entity, restored);
    }

    #[test]
    fn trigger_placement_serde() {
        let trigger = TriggerPlacement {
            position: Vec3::new(5.0, 0.0, 5.0),
            radius: 3.0,
            trigger_type: "checkpoint".into(),
            properties: HashMap::new(),
        };
        let json = serde_json::to_string(&trigger).unwrap();
        let restored: TriggerPlacement = serde_json::from_str(&json).unwrap();
        assert_eq!(trigger, restored);
    }

    #[test]
    fn level_with_entities_roundtrip() {
        let mut level = LevelData::new(2, 2, 1.0);
        level.entities.push(EntityPlacement {
            position: Vec3::new(0.5, 0.0, 0.5),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            entity_type: "crate".into(),
            properties: HashMap::new(),
        });
        level.triggers.push(TriggerPlacement {
            position: Vec3::ZERO,
            radius: 10.0,
            trigger_type: "spawn_zone".into(),
            properties: HashMap::from([("team".into(), "red".into())]),
        });
        let bytes = level.to_bytes().unwrap();
        let restored = LevelData::from_bytes(&bytes).unwrap();
        assert_eq!(level.entities.len(), restored.entities.len());
        assert_eq!(level.triggers.len(), restored.triggers.len());
        assert_eq!(restored.triggers[0].trigger_type, "spawn_zone");
    }

    // ── Defaults ────────────────────────────────────────────────────────

    #[test]
    fn default_surface_is_grass() {
        assert_eq!(SurfaceType::default(), SurfaceType::Grass);
    }

    #[test]
    fn default_terrain_mode_is_heightmap() {
        assert_eq!(TerrainMode::default(), TerrainMode::Heightmap);
    }

    #[test]
    fn custom_surface_type() {
        let s = SurfaceType::Custom(42);
        let json = serde_json::to_string(&s).unwrap();
        let restored: SurfaceType = serde_json::from_str(&json).unwrap();
        assert_eq!(s, restored);
    }
}
