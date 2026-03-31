//! Importer for [Tiled](https://www.mapeditor.org/) map files.
//!
//! Converts Tiled's JSON export format into [`LevelData`]. Supports orthogonal
//! maps with tile layers (for surface types) and object layers (for entities
//! and triggers).
//!
//! # Mapping conventions
//!
//! * The first tile layer is interpreted as the surface layer. Tile GIDs are
//!   mapped to [`SurfaceType`] by a configurable callback or by using the
//!   default mapping: GID 0 → Void, 1 → Grass, 2 → Dirt, 3 → Stone,
//!   4 → Water, 5 → Sand, 6 → Snow, 7 → Mud, 8 → Road, 9 → Cliff.
//!
//! * Each object in an object layer becomes an [`EntityPlacement`] or a
//!   [`TriggerPlacement`] depending on whether a `"radius"` property is
//!   present.
//!
//! * The heightmap layer is flat (all zeros) unless a tile property named
//!   `"height"` is present per tile.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::level_data::{
    EntityPlacement, LevelData, SurfaceType, TriggerPlacement,
};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur when importing a Tiled map.
#[derive(Debug)]
pub enum TiledImportError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedOrientation(String),
}

impl std::fmt::Display for TiledImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Json(e) => write!(f, "JSON parse error: {e}"),
            Self::UnsupportedOrientation(o) => {
                write!(f, "unsupported map orientation: {o}")
            }
        }
    }
}

impl std::error::Error for TiledImportError {}

impl From<std::io::Error> for TiledImportError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for TiledImportError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

// ---------------------------------------------------------------------------
// Tiled JSON structures (subset)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TiledMap {
    width: u32,
    height: u32,
    #[serde(default = "default_tile_width")]
    tilewidth: u32,
    #[serde(default = "default_tile_height")]
    tileheight: u32,
    #[serde(default)]
    orientation: String,
    #[serde(default)]
    layers: Vec<TiledLayer>,
}

fn default_tile_width() -> u32 {
    32
}
fn default_tile_height() -> u32 {
    32
}

#[derive(Deserialize)]
struct TiledLayer {
    #[serde(rename = "type")]
    layer_type: String,
    #[serde(default)]
    data: Vec<u32>,
    #[serde(default)]
    objects: Vec<TiledObject>,
}

#[derive(Deserialize)]
struct TiledObject {
    #[serde(default)]
    x: f64,
    #[serde(default)]
    y: f64,
    #[serde(default)]
    width: f64,
    #[serde(default)]
    height: f64,
    #[serde(default, rename = "type")]
    object_type: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    properties: Vec<TiledProperty>,
}

#[derive(Deserialize)]
struct TiledProperty {
    name: String,
    #[serde(default)]
    value: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Default GID → SurfaceType mapping
// ---------------------------------------------------------------------------

/// Map a Tiled tile GID to a surface type using the default convention.
pub fn default_gid_to_surface(gid: u32) -> SurfaceType {
    match gid {
        0 => SurfaceType::Void,
        1 => SurfaceType::Grass,
        2 => SurfaceType::Dirt,
        3 => SurfaceType::Stone,
        4 => SurfaceType::Water,
        5 => SurfaceType::Sand,
        6 => SurfaceType::Snow,
        7 => SurfaceType::Mud,
        8 => SurfaceType::Road,
        9 => SurfaceType::Cliff,
        other => SurfaceType::Custom(other as u16),
    }
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

/// Load a Tiled JSON map file and convert it to [`LevelData`].
///
/// `cell_size` sets the world-space distance between grid cells (Tiled's
/// `tilewidth` / `tileheight` are in pixels and are not directly used as
/// world units).
///
/// Uses [`default_gid_to_surface`] for tile → surface mapping.
pub fn load_tiled_json(path: &Path, cell_size: f32) -> Result<LevelData, TiledImportError> {
    load_tiled_json_with(path, cell_size, default_gid_to_surface)
}

/// Load a Tiled JSON map file with a custom GID → surface mapping function.
pub fn load_tiled_json_with(
    path: &Path,
    cell_size: f32,
    gid_to_surface: fn(u32) -> SurfaceType,
) -> Result<LevelData, TiledImportError> {
    let text = std::fs::read_to_string(path)?;
    parse_tiled_json(&text, cell_size, gid_to_surface)
}

/// Parse a Tiled JSON string into [`LevelData`].
pub fn parse_tiled_json(
    json: &str,
    cell_size: f32,
    gid_to_surface: fn(u32) -> SurfaceType,
) -> Result<LevelData, TiledImportError> {
    let map: TiledMap = serde_json::from_str(json)?;

    if !map.orientation.is_empty()
        && map.orientation != "orthogonal"
    {
        return Err(TiledImportError::UnsupportedOrientation(
            map.orientation.clone(),
        ));
    }

    let width = map.width;
    let height = map.height;
    let count = (width as usize) * (height as usize);

    let mut level = LevelData::new(width, height, cell_size);
    level.interpolate_height = false; // Tiled maps are flat per-tile.

    // Process layers.
    let mut found_tile_layer = false;
    for layer in &map.layers {
        match layer.layer_type.as_str() {
            "tilelayer" => {
                if !found_tile_layer && !layer.data.is_empty() {
                    // First tile layer → surface types.
                    found_tile_layer = true;
                    for (i, &gid) in layer.data.iter().enumerate().take(count) {
                        level.surface[i] = gid_to_surface(gid);
                        level.walkable[i] = !matches!(
                            level.surface[i],
                            SurfaceType::Water | SurfaceType::Void
                        );
                    }
                }
            }
            "objectgroup" => {
                for obj in &layer.objects {
                    let props: HashMap<String, String> = obj
                        .properties
                        .iter()
                        .map(|p| (p.name.clone(), p.value.to_string()))
                        .collect();

                    // Convert pixel coordinates to world coordinates.
                    let x = (obj.x as f32 / map.tilewidth as f32) * cell_size;
                    let z = (obj.y as f32 / map.tileheight as f32) * cell_size;

                    // Check if this is a trigger (has radius property).
                    let has_radius = props.contains_key("radius");

                    if has_radius {
                        let radius = props
                            .get("radius")
                            .and_then(|v| v.trim_matches('"').parse::<f32>().ok())
                            .unwrap_or(1.0);
                        level.triggers.push(TriggerPlacement {
                            position: euca_math::Vec3::new(x, 0.0, z),
                            radius,
                            trigger_type: obj.object_type.clone(),
                            properties: props,
                        });
                    } else {
                        let entity_type = if obj.object_type.is_empty() {
                            obj.name.clone()
                        } else {
                            obj.object_type.clone()
                        };
                        level.entities.push(EntityPlacement {
                            position: euca_math::Vec3::new(x, 0.0, z),
                            rotation: euca_math::Quat::IDENTITY,
                            scale: euca_math::Vec3::ONE,
                            entity_type,
                            properties: props,
                        });
                    }
                }
            }
            _ => {} // Ignore image layers, group layers, etc.
        }
    }

    Ok(level)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_tiled_map() {
        let json = r#"{
            "width": 3,
            "height": 2,
            "tilewidth": 32,
            "tileheight": 32,
            "orientation": "orthogonal",
            "layers": [
                {
                    "type": "tilelayer",
                    "data": [1, 2, 4, 1, 1, 3]
                }
            ]
        }"#;

        let level = parse_tiled_json(json, 1.0, default_gid_to_surface).unwrap();
        assert_eq!(level.width, 3);
        assert_eq!(level.height, 2);
        assert_eq!(level.surface[0], SurfaceType::Grass);
        assert_eq!(level.surface[1], SurfaceType::Dirt);
        assert_eq!(level.surface[2], SurfaceType::Water);
        assert!(!level.walkable[2]); // Water is not walkable.
        assert!(level.walkable[0]); // Grass is walkable.
        assert!(!level.interpolate_height);
    }

    #[test]
    fn parse_tiled_with_objects() {
        let json = r#"{
            "width": 4,
            "height": 4,
            "tilewidth": 32,
            "tileheight": 32,
            "layers": [
                {
                    "type": "objectgroup",
                    "objects": [
                        {
                            "x": 64.0,
                            "y": 96.0,
                            "type": "tree",
                            "name": "oak",
                            "properties": []
                        },
                        {
                            "x": 32.0,
                            "y": 32.0,
                            "type": "spawn_zone",
                            "name": "red_spawn",
                            "properties": [
                                { "name": "radius", "value": 5.0 },
                                { "name": "team", "value": "red" }
                            ]
                        }
                    ]
                }
            ]
        }"#;

        let level = parse_tiled_json(json, 2.0, default_gid_to_surface).unwrap();
        assert_eq!(level.entities.len(), 1);
        assert_eq!(level.entities[0].entity_type, "tree");
        assert!((level.entities[0].position.x - 4.0).abs() < 1e-3); // 64/32 * 2.0
        assert!((level.entities[0].position.z - 6.0).abs() < 1e-3); // 96/32 * 2.0

        assert_eq!(level.triggers.len(), 1);
        assert_eq!(level.triggers[0].trigger_type, "spawn_zone");
    }

    #[test]
    fn unsupported_orientation_errors() {
        let json = r#"{
            "width": 2,
            "height": 2,
            "orientation": "isometric",
            "layers": []
        }"#;
        let err = parse_tiled_json(json, 1.0, default_gid_to_surface).unwrap_err();
        assert!(matches!(err, TiledImportError::UnsupportedOrientation(_)));
    }

    #[test]
    fn default_gid_mapping() {
        assert_eq!(default_gid_to_surface(0), SurfaceType::Void);
        assert_eq!(default_gid_to_surface(1), SurfaceType::Grass);
        assert_eq!(default_gid_to_surface(4), SurfaceType::Water);
        assert_eq!(default_gid_to_surface(100), SurfaceType::Custom(100));
    }
}
