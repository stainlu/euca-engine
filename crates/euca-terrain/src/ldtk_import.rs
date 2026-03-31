//! Importer for [LDtk](https://ldtk.io/) level files.
//!
//! Converts LDtk's JSON format into [`LevelData`]. Supports the IntGrid layer
//! (for surface types / walkability), Entity layers, and AutoLayer tile data.
//!
//! # Mapping conventions
//!
//! * The first IntGrid layer is mapped to the surface layer. Int values follow
//!   the same convention as Tiled GIDs: 0 → Void, 1 → Grass, 2 → Dirt, etc.
//!   (See [`default_intgrid_to_surface`].)
//!
//! * Entity instances become [`EntityPlacement`] entries. The entity
//!   `__identifier` field is used as the `entity_type`.
//!
//! * LDtk stores multiple levels in one file — this importer picks the
//!   **first** level by default, or a specific level by UID.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::level_data::{EntityPlacement, LevelData, SurfaceType};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur when importing an LDtk file.
#[derive(Debug)]
pub enum LdtkImportError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NoLevels,
    LevelNotFound(i64),
}

impl std::fmt::Display for LdtkImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Json(e) => write!(f, "JSON parse error: {e}"),
            Self::NoLevels => write!(f, "LDtk file contains no levels"),
            Self::LevelNotFound(uid) => write!(f, "level with uid {uid} not found"),
        }
    }
}

impl std::error::Error for LdtkImportError {}

impl From<std::io::Error> for LdtkImportError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for LdtkImportError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

// ---------------------------------------------------------------------------
// LDtk JSON structures (subset)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct LdtkRoot {
    #[serde(default)]
    levels: Vec<LdtkLevel>,
    #[serde(default, rename = "defaultGridSize")]
    default_grid_size: Option<u32>,
}

#[derive(Deserialize)]
struct LdtkLevel {
    #[serde(default)]
    uid: i64,
    #[serde(default, rename = "pxWid")]
    px_wid: u32,
    #[serde(default, rename = "pxHei")]
    px_hei: u32,
    #[serde(default, rename = "layerInstances")]
    layer_instances: Option<Vec<LdtkLayerInstance>>,
}

#[derive(Deserialize)]
struct LdtkLayerInstance {
    #[serde(default, rename = "__type")]
    layer_type: String,
    #[serde(default, rename = "__cWid")]
    c_wid: u32,
    #[serde(default, rename = "__cHei")]
    c_hei: u32,
    #[serde(default, rename = "__gridSize")]
    grid_size: u32,
    #[serde(default, rename = "intGridCsv")]
    int_grid_csv: Vec<i32>,
    #[serde(default, rename = "entityInstances")]
    entity_instances: Vec<LdtkEntityInstance>,
}

#[derive(Deserialize)]
struct LdtkEntityInstance {
    #[serde(default, rename = "__identifier")]
    identifier: String,
    #[serde(default)]
    px: [f64; 2],
    #[serde(default, rename = "fieldInstances")]
    field_instances: Vec<LdtkFieldInstance>,
}

#[derive(Deserialize)]
struct LdtkFieldInstance {
    #[serde(default, rename = "__identifier")]
    identifier: String,
    #[serde(default, rename = "__value")]
    value: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Default IntGrid → SurfaceType mapping
// ---------------------------------------------------------------------------

/// Map an LDtk IntGrid value to a surface type.
pub fn default_intgrid_to_surface(value: i32) -> SurfaceType {
    match value {
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
        other if other > 0 => SurfaceType::Custom(other as u16),
        _ => SurfaceType::Void,
    }
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

/// Load the first level from an LDtk JSON file.
pub fn load_ldtk_json(path: &Path, cell_size: f32) -> Result<LevelData, LdtkImportError> {
    let text = std::fs::read_to_string(path)?;
    parse_ldtk_json(&text, cell_size, None)
}

/// Load a specific level (by UID) from an LDtk JSON file.
pub fn load_ldtk_json_by_uid(
    path: &Path,
    cell_size: f32,
    uid: i64,
) -> Result<LevelData, LdtkImportError> {
    let text = std::fs::read_to_string(path)?;
    parse_ldtk_json(&text, cell_size, Some(uid))
}

/// Parse an LDtk JSON string into [`LevelData`].
///
/// If `uid` is `None`, the first level is used.
pub fn parse_ldtk_json(
    json: &str,
    cell_size: f32,
    uid: Option<i64>,
) -> Result<LevelData, LdtkImportError> {
    let root: LdtkRoot = serde_json::from_str(json)?;

    let ldtk_level = if let Some(uid) = uid {
        root.levels
            .iter()
            .find(|l| l.uid == uid)
            .ok_or(LdtkImportError::LevelNotFound(uid))?
    } else {
        root.levels.first().ok_or(LdtkImportError::NoLevels)?
    };

    let layers = ldtk_level
        .layer_instances
        .as_deref()
        .unwrap_or(&[]);

    // Determine grid dimensions from the first layer that has them, or from
    // the pixel dimensions + default grid size.
    let (grid_w, grid_h, grid_px) = layers
        .iter()
        .find(|l| l.c_wid > 0 && l.c_hei > 0)
        .map(|l| (l.c_wid, l.c_hei, l.grid_size))
        .unwrap_or_else(|| {
            let gs = root.default_grid_size.unwrap_or(16);
            let w = if gs > 0 { ldtk_level.px_wid / gs } else { 1 };
            let h = if gs > 0 { ldtk_level.px_hei / gs } else { 1 };
            (w.max(1), h.max(1), gs)
        });

    let mut level = LevelData::new(grid_w, grid_h, cell_size);
    level.interpolate_height = false; // LDtk maps are flat per-cell.

    // Process layers (LDtk stores layers bottom-to-top; we iterate as-is).
    let mut found_intgrid = false;
    for layer in layers {
        match layer.layer_type.as_str() {
            "IntGrid" => {
                if !found_intgrid && !layer.int_grid_csv.is_empty() {
                    found_intgrid = true;
                    let count = level.cell_count();
                    for (i, &val) in layer.int_grid_csv.iter().enumerate().take(count) {
                        level.surface[i] = default_intgrid_to_surface(val);
                        level.walkable[i] = !matches!(
                            level.surface[i],
                            SurfaceType::Water | SurfaceType::Void
                        );
                    }
                }
            }
            "Entities" => {
                for ent in &layer.entity_instances {
                    // Convert pixel position to world coordinates.
                    let x = (ent.px[0] as f32 / grid_px.max(1) as f32) * cell_size;
                    let z = (ent.px[1] as f32 / grid_px.max(1) as f32) * cell_size;

                    let mut props = HashMap::new();
                    for field in &ent.field_instances {
                        props.insert(
                            field.identifier.clone(),
                            field.value.to_string(),
                        );
                    }

                    level.entities.push(EntityPlacement {
                        position: euca_math::Vec3::new(x, 0.0, z),
                        rotation: euca_math::Quat::IDENTITY,
                        scale: euca_math::Vec3::ONE,
                        entity_type: ent.identifier.clone(),
                        properties: props,
                    });
                }
            }
            _ => {} // AutoLayer, Tiles — not yet mapped.
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
    fn parse_minimal_ldtk() {
        let json = r#"{
            "defaultGridSize": 16,
            "levels": [
                {
                    "uid": 1,
                    "pxWid": 48,
                    "pxHei": 32,
                    "layerInstances": [
                        {
                            "__type": "IntGrid",
                            "__cWid": 3,
                            "__cHei": 2,
                            "__gridSize": 16,
                            "intGridCsv": [1, 2, 4, 1, 1, 3],
                            "entityInstances": []
                        }
                    ]
                }
            ]
        }"#;

        let level = parse_ldtk_json(json, 1.0, None).unwrap();
        assert_eq!(level.width, 3);
        assert_eq!(level.height, 2);
        assert_eq!(level.surface[0], SurfaceType::Grass);
        assert_eq!(level.surface[1], SurfaceType::Dirt);
        assert_eq!(level.surface[2], SurfaceType::Water);
        assert!(!level.walkable[2]);
        assert!(!level.interpolate_height);
    }

    #[test]
    fn parse_ldtk_with_entities() {
        let json = r#"{
            "levels": [
                {
                    "uid": 1,
                    "pxWid": 64,
                    "pxHei": 64,
                    "layerInstances": [
                        {
                            "__type": "Entities",
                            "__cWid": 4,
                            "__cHei": 4,
                            "__gridSize": 16,
                            "intGridCsv": [],
                            "entityInstances": [
                                {
                                    "__identifier": "PlayerSpawn",
                                    "px": [32.0, 48.0],
                                    "fieldInstances": [
                                        { "__identifier": "team", "__value": "blue" }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }"#;

        let level = parse_ldtk_json(json, 2.0, None).unwrap();
        assert_eq!(level.entities.len(), 1);
        assert_eq!(level.entities[0].entity_type, "PlayerSpawn");
        // 32px / 16px * 2.0 = 4.0
        assert!((level.entities[0].position.x - 4.0).abs() < 1e-3);
        // 48px / 16px * 2.0 = 6.0
        assert!((level.entities[0].position.z - 6.0).abs() < 1e-3);
        assert!(level.entities[0].properties.contains_key("team"));
    }

    #[test]
    fn load_by_uid_not_found() {
        let json = r#"{ "levels": [{ "uid": 1, "pxWid": 16, "pxHei": 16 }] }"#;
        let err = parse_ldtk_json(json, 1.0, Some(999)).unwrap_err();
        assert!(matches!(err, LdtkImportError::LevelNotFound(999)));
    }

    #[test]
    fn no_levels_error() {
        let json = r#"{ "levels": [] }"#;
        let err = parse_ldtk_json(json, 1.0, None).unwrap_err();
        assert!(matches!(err, LdtkImportError::NoLevels));
    }

    #[test]
    fn default_intgrid_mapping() {
        assert_eq!(default_intgrid_to_surface(0), SurfaceType::Void);
        assert_eq!(default_intgrid_to_surface(1), SurfaceType::Grass);
        assert_eq!(default_intgrid_to_surface(4), SurfaceType::Water);
        assert_eq!(default_intgrid_to_surface(-1), SurfaceType::Void);
        assert_eq!(default_intgrid_to_surface(100), SurfaceType::Custom(100));
    }
}
