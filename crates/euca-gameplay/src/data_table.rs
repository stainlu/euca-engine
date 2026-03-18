//! Data tables — JSON-driven game configuration.
//!
//! Resources: `DataTable`.
//! Load game balance, weapon stats, level configs from JSON files.

use serde::de::DeserializeOwned;
use std::collections::HashMap;

/// JSON-loaded game data. Stored as a World resource.
///
/// Tables are loaded from JSON files and queried by path (e.g. "weapons.sword.damage").
#[derive(Clone, Debug, Default)]
pub struct DataTable {
    tables: HashMap<String, serde_json::Value>,
}

impl DataTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load a JSON file into a named table.
    pub fn load(&mut self, name: &str, path: &str) -> Result<(), String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("Cannot read {path}: {e}"))?;
        let value: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("Invalid JSON in {path}: {e}"))?;
        self.tables.insert(name.to_string(), value);
        log::info!("DataTable: loaded '{name}' from {path}");
        Ok(())
    }

    /// Load a JSON file, using the filename (without extension) as the table name.
    pub fn load_auto(&mut self, path: &str) -> Result<(), String> {
        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default")
            .to_string();
        self.load(&name, path)
    }

    /// Get a value by dotted path (e.g. "weapons.sword.damage").
    pub fn get_raw(&self, path: &str) -> Option<&serde_json::Value> {
        let parts: Vec<&str> = path.splitn(2, '.').collect();
        if parts.is_empty() {
            return None;
        }
        let table = self.tables.get(parts[0])?;
        if parts.len() == 1 {
            return Some(table);
        }
        // Navigate nested path
        let mut current = table;
        for key in parts[1].split('.') {
            current = current.get(key)?;
        }
        Some(current)
    }

    /// Get a typed value by dotted path.
    pub fn get<T: DeserializeOwned>(&self, path: &str) -> Option<T> {
        let raw = self.get_raw(path)?;
        serde_json::from_value(raw.clone()).ok()
    }

    /// Get a float value by dotted path.
    pub fn get_f32(&self, path: &str) -> Option<f32> {
        self.get_raw(path)?.as_f64().map(|v| v as f32)
    }

    /// Get a string value by dotted path.
    pub fn get_str(&self, path: &str) -> Option<String> {
        self.get_raw(path)?.as_str().map(|s| s.to_string())
    }

    /// List all loaded table names.
    pub fn table_names(&self) -> Vec<&str> {
        self.tables.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_nested_value() {
        let mut dt = DataTable::new();
        dt.tables.insert(
            "weapons".to_string(),
            serde_json::json!({
                "sword": {
                    "damage": 25.0,
                    "speed": 1.5,
                    "name": "Iron Sword"
                },
                "bow": {
                    "damage": 15.0,
                    "range": 30.0
                }
            }),
        );

        assert_eq!(dt.get_f32("weapons.sword.damage"), Some(25.0));
        assert_eq!(dt.get_str("weapons.sword.name"), Some("Iron Sword".into()));
        assert_eq!(dt.get_f32("weapons.bow.range"), Some(30.0));
    }

    #[test]
    fn missing_path_returns_none() {
        let dt = DataTable::new();
        assert_eq!(dt.get_f32("nonexistent.path"), None);
    }

    #[test]
    fn table_names_listed() {
        let mut dt = DataTable::new();
        dt.tables
            .insert("weapons".to_string(), serde_json::json!({}));
        dt.tables
            .insert("config".to_string(), serde_json::json!({}));

        let names = dt.table_names();
        assert!(names.contains(&"weapons"));
        assert!(names.contains(&"config"));
    }
}
