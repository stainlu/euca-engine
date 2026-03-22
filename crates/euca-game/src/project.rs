//! Project configuration file (`.eucaproject.json`).
//!
//! Defines game metadata: name, version, default level, window settings,
//! and directory conventions for levels and assets.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Window configuration for the standalone game runner.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowConfig {
    /// Window title bar text.
    pub title: String,
    /// Initial window width in pixels.
    pub width: u32,
    /// Initial window height in pixels.
    pub height: u32,
    /// Start in fullscreen mode.
    #[serde(default)]
    pub fullscreen: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Euca Game".to_string(),
            width: 1280,
            height: 720,
            fullscreen: false,
        }
    }
}

/// Top-level project configuration.
///
/// Loaded from `.eucaproject.json` in the game's root directory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectConfig {
    /// Human-readable game name.
    pub name: String,
    /// Semantic version string.
    #[serde(default = "default_version")]
    pub version: String,
    /// Path to the default level file (relative to project root).
    pub default_level: String,
    /// Window configuration.
    #[serde(default)]
    pub window: WindowConfig,
    /// Directory containing level files (relative to project root).
    #[serde(default = "default_levels_dir")]
    pub levels_dir: String,
    /// Directory containing game assets (relative to project root).
    #[serde(default = "default_assets_dir")]
    pub assets_dir: String,
}

fn default_version() -> String {
    "0.1.0".to_string()
}
fn default_levels_dir() -> String {
    "levels".to_string()
}
fn default_assets_dir() -> String {
    "assets".to_string()
}

/// Standard project file name.
pub const PROJECT_FILE_NAME: &str = ".eucaproject.json";

impl ProjectConfig {
    /// Load a project config from a JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
        serde_json::from_str(&data)
            .map_err(|e| format!("Invalid project JSON in {}: {e}", path.display()))
    }

    /// Try to find and load `.eucaproject.json` in the given directory.
    pub fn discover(dir: impl AsRef<Path>) -> Option<Self> {
        let path = dir.as_ref().join(PROJECT_FILE_NAME);
        if path.exists() {
            Self::load(&path).ok()
        } else {
            None
        }
    }

    /// Save the project config to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("Serialization failed: {e}"))?;
        std::fs::write(path.as_ref(), json).map_err(|e| format!("Write failed: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_minimal() {
        let json = r#"{"name": "Test Game", "default_level": "test.json"}"#;
        let config: ProjectConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.name, "Test Game");
        assert_eq!(config.default_level, "test.json");
        assert_eq!(config.version, "0.1.0");
        assert_eq!(config.window.width, 1280);
        assert_eq!(config.levels_dir, "levels");
    }

    #[test]
    fn deserialize_full() {
        let json = r#"{
            "name": "MOBA",
            "version": "1.0.0",
            "default_level": "moba.json",
            "window": {"title": "My MOBA", "width": 1920, "height": 1080, "fullscreen": true},
            "levels_dir": "maps",
            "assets_dir": "content"
        }"#;
        let config: ProjectConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.name, "MOBA");
        assert_eq!(config.version, "1.0.0");
        assert!(config.window.fullscreen);
        assert_eq!(config.window.width, 1920);
        assert_eq!(config.levels_dir, "maps");
    }

    #[test]
    fn roundtrip_save_load() {
        let config = ProjectConfig {
            name: "Test".to_string(),
            version: "0.2.0".to_string(),
            default_level: "level.json".to_string(),
            window: WindowConfig::default(),
            levels_dir: "levels".to_string(),
            assets_dir: "assets".to_string(),
        };
        let tmp = std::env::temp_dir().join("euca_test_project.json");
        config.save(&tmp).unwrap();
        let loaded = ProjectConfig::load(&tmp).unwrap();
        assert_eq!(loaded.name, "Test");
        assert_eq!(loaded.version, "0.2.0");
        std::fs::remove_file(tmp).ok();
    }
}
