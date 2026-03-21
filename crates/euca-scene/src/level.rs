use serde::{Deserialize, Serialize};

// ── Data structs ──

/// Top-level level file format. Describes an entire level (entities, rules,
/// camera, and game mode) in a serialisable form suitable for JSON on disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LevelData {
    pub name: String,
    pub entities: Vec<EntityData>,
    #[serde(default)]
    pub rules: Vec<RuleData>,
    #[serde(default)]
    pub camera: Option<CameraData>,
    #[serde(default)]
    pub game: Option<GameData>,
}

/// Describes a single entity to be spawned when the level loads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EntityData {
    #[serde(default)]
    pub position: Option<[f32; 3]>,
    #[serde(default)]
    pub scale: Option<[f32; 3]>,
    /// Mesh primitive name, e.g. `"cube"`, `"sphere"`.
    #[serde(default)]
    pub mesh: Option<String>,
    /// Colour name, e.g. `"blue"`, `"red"`, `"cyan"`.
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub health: Option<f32>,
    #[serde(default)]
    pub team: Option<u8>,
    /// Physics body type: `"Static"`, `"Kinematic"`, or `"Dynamic"`.
    #[serde(default)]
    pub physics: Option<String>,
    /// Gameplay role: `"hero"`, `"minion"`, `"tower"`, `"structure"`.
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub combat: Option<CombatData>,
    #[serde(default)]
    pub spawn_point: Option<u8>,
    #[serde(default)]
    pub player: Option<bool>,
    #[serde(default)]
    pub gold: Option<i32>,
    #[serde(default)]
    pub gold_bounty: Option<i32>,
    #[serde(default)]
    pub xp_bounty: Option<u32>,
    /// Collider descriptor, e.g. `"sphere:0.6"`, `"aabb:1,1,1"`.
    #[serde(default)]
    pub collider: Option<String>,
}

/// Combat parameters attached to an entity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CombatData {
    pub damage: f32,
    pub range: f32,
    pub speed: f32,
    pub cooldown: f32,
    /// `"melee"` or `"stationary"`.
    #[serde(default)]
    pub style: Option<String>,
}

/// A reactive rule that fires when a condition is met.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuleData {
    /// Trigger expression, e.g. `"timer:20"`, `"death"`.
    pub when: String,
    /// Action expression, e.g. `"spawn cube ..."`.
    pub action: String,
    /// Optional entity filter, e.g. `"team:1"`.
    #[serde(default)]
    pub filter: Option<String>,
}

/// Initial camera placement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CameraData {
    pub eye: [f32; 3],
    pub target: [f32; 3],
}

/// Game mode configuration embedded in the level file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GameData {
    pub mode: String,
    #[serde(default)]
    pub score_limit: Option<u32>,
}

// ── Load / Save ──

/// Deserialise a [`LevelData`] from a JSON file at `path`.
pub fn load_level(path: &str) -> Result<LevelData, String> {
    let contents =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read {path}: {e}"))?;
    serde_json::from_str(&contents).map_err(|e| format!("failed to parse {path}: {e}"))
}

/// Serialise a [`LevelData`] to a JSON file at `path` (pretty-printed).
pub fn save_level(data: &LevelData, path: &str) -> Result<(), String> {
    let json = serde_json::to_string_pretty(data)
        .map_err(|e| format!("failed to serialise level: {e}"))?;
    std::fs::write(path, json).map_err(|e| format!("failed to write {path}: {e}"))
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal level for testing.
    fn sample_level() -> LevelData {
        LevelData {
            name: "test_arena".into(),
            entities: vec![
                EntityData {
                    position: Some([1.0, 2.0, 3.0]),
                    mesh: Some("cube".into()),
                    color: Some("blue".into()),
                    health: Some(100.0),
                    team: Some(1),
                    physics: Some("Static".into()),
                    role: Some("tower".into()),
                    combat: Some(CombatData {
                        damage: 25.0,
                        range: 8.0,
                        speed: 0.0,
                        cooldown: 1.5,
                        style: Some("stationary".into()),
                    }),
                    spawn_point: Some(0),
                    player: Some(false),
                    gold: Some(0),
                    gold_bounty: Some(150),
                    xp_bounty: Some(200),
                    collider: Some("sphere:0.6".into()),
                    scale: Some([2.0, 2.0, 2.0]),
                },
                EntityData {
                    position: Some([5.0, 0.0, 0.0]),
                    mesh: Some("sphere".into()),
                    color: Some("red".into()),
                    health: Some(50.0),
                    team: Some(2),
                    physics: Some("Dynamic".into()),
                    role: Some("minion".into()),
                    combat: None,
                    spawn_point: None,
                    player: None,
                    gold: None,
                    gold_bounty: None,
                    xp_bounty: None,
                    collider: None,
                    scale: None,
                },
            ],
            rules: vec![RuleData {
                when: "timer:20".into(),
                action: "spawn cube at 0,0,0".into(),
                filter: Some("team:1".into()),
            }],
            camera: Some(CameraData {
                eye: [0.0, 10.0, 10.0],
                target: [0.0, 0.0, 0.0],
            }),
            game: Some(GameData {
                mode: "arena".into(),
                score_limit: Some(10),
            }),
        }
    }

    // ── Test 1: round-trip serialize -> deserialize ──

    #[test]
    fn roundtrip_json() {
        let level = sample_level();
        let json = serde_json::to_string_pretty(&level).unwrap();
        let parsed: LevelData = serde_json::from_str(&json).unwrap();
        assert_eq!(level, parsed);
    }

    // ── Test 2: missing optional fields default correctly ──

    #[test]
    fn missing_optional_fields() {
        let json = r#"{
            "name": "bare",
            "entities": [
                { "mesh": "cube" }
            ]
        }"#;
        let level: LevelData = serde_json::from_str(json).unwrap();
        assert_eq!(level.name, "bare");
        assert_eq!(level.entities.len(), 1);
        assert!(level.entities[0].position.is_none());
        assert!(level.entities[0].health.is_none());
        assert!(level.entities[0].combat.is_none());
        assert!(level.rules.is_empty());
        assert!(level.camera.is_none());
        assert!(level.game.is_none());
    }

    // ── Test 3: combat data serialises correctly ──

    #[test]
    fn combat_data_roundtrip() {
        let combat = CombatData {
            damage: 30.0,
            range: 5.0,
            speed: 1.2,
            cooldown: 0.8,
            style: Some("melee".into()),
        };
        let json = serde_json::to_string(&combat).unwrap();
        let parsed: CombatData = serde_json::from_str(&json).unwrap();
        assert_eq!(combat, parsed);
    }

    // ── Test 4: save then load file ──

    #[test]
    fn save_and_load_file() {
        let dir = std::env::temp_dir().join("euca_level_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_level.json");
        let path_str = path.to_str().unwrap();

        let level = sample_level();
        save_level(&level, path_str).unwrap();
        let loaded = load_level(path_str).unwrap();
        assert_eq!(level, loaded);

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    // ── Test 5: load non-existent file returns error ──

    #[test]
    fn load_missing_file_returns_error() {
        let result = load_level("/tmp/euca_does_not_exist_12345.json");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to read"));
    }

    // ── Test 6: malformed JSON returns parse error ──

    #[test]
    fn load_malformed_json() {
        let dir = std::env::temp_dir().join("euca_level_malformed");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.json");
        std::fs::write(&path, "{ not valid json }").unwrap();

        let result = load_level(path.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to parse"));

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    // ── Test 7: empty entities list is valid ──

    #[test]
    fn empty_entities_list() {
        let json = r#"{ "name": "empty", "entities": [] }"#;
        let level: LevelData = serde_json::from_str(json).unwrap();
        assert_eq!(level.name, "empty");
        assert!(level.entities.is_empty());
    }
}
