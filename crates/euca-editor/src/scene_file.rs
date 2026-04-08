use std::collections::HashMap;

use euca_ecs::{Entity, Query, World};
use euca_reflect::TypeRegistry;
use euca_reflect::json::{reflect_from_json, reflect_to_json};
use euca_render::{MaterialRef, MeshRenderer};
use euca_scene::{GlobalTransform, LocalTransform};
use serde::{Deserialize, Serialize};

/// Current scene file format version.
pub const SCENE_VERSION: u32 = 2;

/// A serializable scene file format with versioning.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SceneFile {
    /// Format version for migration support.
    #[serde(default = "default_version")]
    pub version: u32,
    pub entities: Vec<SceneEntity>,
}

fn default_version() -> u32 {
    1
}

/// A serializable entity with all supported components.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SceneEntity {
    pub position: [f32; 3],
    #[serde(default = "default_scale")]
    pub scale: [f32; 3],
    #[serde(default = "default_rotation")]
    pub rotation: [f32; 4],
    /// Mesh type name ("cube", "sphere", "plane", or "none").
    pub mesh: String,
    /// Material index (refers to upload order in the editor).
    pub material: u32,
    /// Health (current, max). None if entity has no Health component.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub health: Option<[f32; 2]>,
    /// Team ID. None if entity has no Team component.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub team: Option<u8>,
    /// Physics body type ("Dynamic", "Static", "Kinematic"). None if no PhysicsBody.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physics_body: Option<String>,
    /// Has AutoCombat component.
    #[serde(default)]
    pub combat: bool,
}

fn default_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn default_rotation() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

impl SceneFile {
    /// Capture the current world state as a scene file.
    pub fn capture(world: &World) -> Self {
        let query = Query::<(Entity, &LocalTransform)>::new(world);
        let entities = query
            .iter()
            .map(|(e, lt)| {
                let t = &lt.0;
                let mesh = world
                    .get::<MeshRenderer>(e)
                    .map_or("none".to_string(), |m| format!("mesh_{}", m.mesh.0));
                let material = world.get::<MaterialRef>(e).map_or(0, |m| m.handle.0);

                let health = world
                    .get::<euca_gameplay::Health>(e)
                    .map(|h| [h.current, h.max]);
                let team = world.get::<euca_gameplay::Team>(e).map(|t| t.0);
                let physics_body =
                    world
                        .get::<euca_physics::PhysicsBody>(e)
                        .map(|pb| match pb.body_type {
                            euca_physics::RigidBodyType::Dynamic => "Dynamic".to_string(),
                            euca_physics::RigidBodyType::Static => "Static".to_string(),
                            euca_physics::RigidBodyType::Kinematic => "Kinematic".to_string(),
                        });
                let combat = world.get::<euca_gameplay::AutoCombat>(e).is_some();

                SceneEntity {
                    position: [t.translation.x, t.translation.y, t.translation.z],
                    scale: [t.scale.x, t.scale.y, t.scale.z],
                    rotation: [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w],
                    mesh,
                    material,
                    health,
                    team,
                    physics_body,
                    combat,
                }
            })
            .collect();
        Self {
            version: SCENE_VERSION,
            entities,
        }
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Scene serialization failed")
    }

    /// Deserialize from JSON with automatic version migration.
    ///
    /// Detects the format version:
    /// - v1/v2: parsed as [`SceneFile`] (legacy hardcoded components)
    /// - v3+: parsed as [`SceneFileV3`] (reflection-based), **not** returned
    ///   here. Use [`SceneFileV3::from_json`] directly for v3 scenes.
    ///
    /// Returns `Err` if the JSON is v3+ (callers should use `SceneFileV3::from_json`).
    pub fn from_json(json: &str) -> Result<Self, String> {
        // Peek at version to reject v3+ scenes that should use SceneFileV3.
        if let Ok(v) = serde_json::from_str::<VersionProbe>(json)
            && v.version >= 3
        {
            return Err(format!(
                "Scene version {} requires SceneFileV3::from_json",
                v.version
            ));
        }
        let mut scene: SceneFile =
            serde_json::from_str(json).map_err(|e| format!("Scene deserialization failed: {e}"))?;
        scene.migrate();
        Ok(scene)
    }

    /// Apply version migrations to bring the scene up to current format.
    fn migrate(&mut self) {
        if self.version < 2 {
            // v1 → v2: add rotation, health, team, physics_body, combat fields
            // serde defaults handle missing fields, so just bump version
            log::info!("Migrating scene from v{} to v{SCENE_VERSION}", self.version);
            self.version = SCENE_VERSION;
        }
    }

    /// Save to a file.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = self.to_json();
        std::fs::write(path, json).map_err(|e| format!("Failed to write scene: {e}"))
    }

    /// Load from a file.
    pub fn load(path: &str) -> Result<Self, String> {
        let json =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read scene: {e}"))?;
        Self::from_json(&json)
    }
}

/// Rebuild world entities from a scene file. Returns the list of spawned entities.
pub fn load_scene_into_world(
    world: &mut World,
    scene: &SceneFile,
    mesh_lookup: &dyn Fn(&str) -> Option<euca_render::MeshHandle>,
    material_count: u32,
) -> Vec<Entity> {
    let mut spawned = Vec::new();
    for se in &scene.entities {
        let pos = euca_math::Vec3::new(se.position[0], se.position[1], se.position[2]);
        let scl = euca_math::Vec3::new(se.scale[0], se.scale[1], se.scale[2]);
        let rot = euca_math::Quat::from_xyzw(
            se.rotation[0],
            se.rotation[1],
            se.rotation[2],
            se.rotation[3],
        );
        let mut transform = euca_math::Transform::from_translation(pos);
        transform.scale = scl;
        transform.rotation = rot;

        let e = world.spawn(LocalTransform(transform));
        world.insert(e, GlobalTransform::default());

        if let Some(mesh_handle) = mesh_lookup(&se.mesh) {
            world.insert(e, MeshRenderer { mesh: mesh_handle });
        }

        if se.material < material_count {
            world.insert(
                e,
                MaterialRef {
                    handle: euca_render::MaterialHandle(se.material),
                },
            );
        }

        // Restore gameplay components
        if let Some([current, max]) = se.health {
            let mut h = euca_gameplay::Health::new(max);
            h.current = current;
            world.insert(e, h);
        }
        if let Some(team_id) = se.team {
            world.insert(e, euca_gameplay::Team(team_id));
        }
        if let Some(ref body_type) = se.physics_body {
            let pb = match body_type.as_str() {
                "Static" => euca_physics::PhysicsBody::fixed(),
                "Kinematic" => euca_physics::PhysicsBody {
                    body_type: euca_physics::RigidBodyType::Kinematic,
                },
                _ => euca_physics::PhysicsBody::dynamic(),
            };
            world.insert(e, pb);
        }
        if se.combat {
            world.insert(e, euca_gameplay::AutoCombat::new());
        }

        spawned.push(e);
    }
    spawned
}

// ── Prefab System ──

/// A prefab: a named, reusable entity template stored as scene data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prefab {
    pub name: String,
    pub entities: Vec<SceneEntity>,
}

/// World resource: stores named prefabs.
#[derive(Clone, Debug, Default)]
pub struct PrefabRegistry {
    pub prefabs: std::collections::HashMap<String, Prefab>,
}

impl PrefabRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a prefab from scene entities.
    pub fn register(&mut self, name: impl Into<String>, entities: Vec<SceneEntity>) {
        let name = name.into();
        self.prefabs.insert(name.clone(), Prefab { name, entities });
    }

    /// Instantiate a prefab at a position offset.
    pub fn instantiate(
        &self,
        name: &str,
        world: &mut World,
        offset: euca_math::Vec3,
        mesh_lookup: &dyn Fn(&str) -> Option<euca_render::MeshHandle>,
        material_count: u32,
    ) -> Option<Vec<Entity>> {
        let prefab = self.prefabs.get(name)?;

        // Offset all entity positions
        let mut offset_scene = SceneFile {
            version: SCENE_VERSION,
            entities: prefab.entities.clone(),
        };
        for entity in &mut offset_scene.entities {
            entity.position[0] += offset.x;
            entity.position[1] += offset.y;
            entity.position[2] += offset.z;
        }

        Some(load_scene_into_world(
            world,
            &offset_scene,
            mesh_lookup,
            material_count,
        ))
    }
}

// ── V3 Scene Format (reflection-driven) ──

/// Helper for peeking at the `version` field without fully parsing.
#[derive(Deserialize)]
struct VersionProbe {
    #[serde(default = "default_version")]
    version: u32,
}

/// V3 entity: stores components as `type_name -> JSON object`.
///
/// Each value is the output of [`reflect_to_json`], which includes a
/// `__type` discriminator for struct-typed components.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReflectSceneEntity {
    pub components: HashMap<String, serde_json::Value>,
}

/// Reflection-based scene file (v3+).
///
/// Unlike the legacy [`SceneFile`] which hardcodes a fixed set of
/// components, `SceneFileV3` serializes *any* component that has been
/// registered for reflection via [`World::register_reflect`].
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SceneFileV3 {
    /// Format version. Always `3` for this format.
    pub version: u32,
    /// Entities and their reflected components.
    pub entities: Vec<ReflectSceneEntity>,
}

impl SceneFileV3 {
    /// Capture the current world state using reflection.
    ///
    /// For each alive entity, iterates all reflection-registered component
    /// names and serializes whichever ones the entity actually has.
    /// Components that are registered but absent on a given entity are
    /// silently skipped.
    pub fn capture(world: &World) -> Self {
        let entities = world
            .all_entities()
            .into_iter()
            .filter_map(|entity| {
                let mut components = HashMap::new();
                for name in world.reflect_component_names() {
                    if let Some(val) = world.get_reflect(entity, name)
                        && let Ok(json) = reflect_to_json(val.as_ref())
                    {
                        components.insert(name.to_string(), json);
                    }
                }
                // Only include entities that have at least one reflected component.
                if components.is_empty() {
                    None
                } else {
                    Some(ReflectSceneEntity { components })
                }
            })
            .collect();

        Self {
            version: 3,
            entities,
        }
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Scene V3 serialization failed")
    }

    /// Deserialize a v3 scene from JSON.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let scene: SceneFileV3 = serde_json::from_str(json)
            .map_err(|e| format!("Scene V3 deserialization failed: {e}"))?;
        if scene.version < 3 {
            return Err(format!(
                "Expected scene version >= 3, got {}. Use SceneFile::from_json for v1/v2.",
                scene.version
            ));
        }
        Ok(scene)
    }
}

/// Load a v3 scene into the world using reflection.
///
/// For each entity in the scene, spawns a new empty entity and inserts
/// components via [`World::insert_reflect`]. Components whose `__type`
/// is not found in the `type_registry` are skipped with a log warning.
///
/// Returns the list of spawned entities.
pub fn load_scene_v3_into_world(
    world: &mut World,
    scene: &SceneFileV3,
    type_registry: &TypeRegistry,
) -> Vec<Entity> {
    let mut spawned = Vec::new();
    for re in &scene.entities {
        let entity = world.spawn_empty();
        for json_val in re.components.values() {
            // The __type field inside the JSON object tells us the type name.
            let type_name = match json_val.get("__type").and_then(|v| v.as_str()) {
                Some(tn) => tn,
                None => {
                    // Primitive component stored without __type — skip.
                    log::info!("Skipping component without __type field");
                    continue;
                }
            };
            match reflect_from_json(json_val, type_registry) {
                Some(val) => {
                    if !world.insert_reflect(entity, type_name, val) {
                        log::info!(
                            "Skipping unregistered reflect component '{type_name}' during scene load"
                        );
                    }
                }
                None => {
                    log::info!("Failed to deserialize component '{type_name}' from scene JSON");
                }
            }
        }
        spawned.push(entity);
    }
    spawned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_file_roundtrip() {
        let scene = SceneFile {
            version: SCENE_VERSION,
            entities: vec![SceneEntity {
                position: [1.0, 2.0, 3.0],
                scale: [1.0, 1.0, 1.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                mesh: "cube".to_string(),
                material: 0,
                health: Some([80.0, 100.0]),
                team: Some(1),
                physics_body: Some("Dynamic".into()),
                combat: true,
            }],
        };
        let json = scene.to_json();
        let restored = SceneFile::from_json(&json).unwrap();
        assert_eq!(restored.entities.len(), 1);
        assert_eq!(restored.entities[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(restored.entities[0].health, Some([80.0, 100.0]));
        assert_eq!(restored.entities[0].team, Some(1));
        assert!(restored.entities[0].combat);
        assert_eq!(restored.version, SCENE_VERSION);
    }

    #[test]
    fn v1_scene_migration() {
        // Simulate a v1 scene (no version field, no health/team/combat)
        let v1_json = r#"{"entities": [{"position": [1, 2, 3], "scale": [1, 1, 1], "mesh": "cube", "material": 0}]}"#;
        let scene = SceneFile::from_json(v1_json).unwrap();
        assert_eq!(scene.version, SCENE_VERSION);
        assert!(scene.entities[0].health.is_none());
        assert!(!scene.entities[0].combat);
    }

    #[test]
    fn prefab_registry() {
        let mut registry = PrefabRegistry::new();
        registry.register(
            "soldier",
            vec![SceneEntity {
                position: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                mesh: "cube".into(),
                material: 0,
                health: Some([100.0, 100.0]),
                team: Some(1),
                physics_body: None,
                combat: true,
            }],
        );

        assert!(registry.prefabs.contains_key("soldier"));
        assert_eq!(registry.prefabs["soldier"].entities.len(), 1);
    }

    // ── V3 reflection-driven tests ──

    /// Test component with Reflect derive.
    #[derive(Clone, Debug, Default, PartialEq, euca_reflect::Reflect)]
    struct TestHealth {
        current: f32,
        max: f32,
    }

    /// Test tuple-struct component with Reflect derive.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, euca_reflect::Reflect)]
    struct TestTeam(u8);

    #[test]
    fn v3_capture_and_load_roundtrip() {
        let mut world = World::new();
        let mut type_reg = TypeRegistry::new();

        // Register types for both the ECS reflection bridge and the type registry.
        world.register_reflect::<TestHealth>();
        world.register_reflect::<TestTeam>();
        type_reg.register::<TestHealth>();
        type_reg.register::<TestTeam>();

        // Spawn an entity with both components.
        let e = world.spawn(TestHealth {
            current: 80.0,
            max: 100.0,
        });
        world.insert(e, TestTeam(2));

        // Capture.
        let scene = SceneFileV3::capture(&world);
        assert_eq!(scene.version, 3);
        assert_eq!(scene.entities.len(), 1);
        assert!(scene.entities[0].components.contains_key("TestHealth"));
        assert!(scene.entities[0].components.contains_key("TestTeam"));

        // Serialize and deserialize.
        let json = scene.to_json();
        let restored = SceneFileV3::from_json(&json).unwrap();
        assert_eq!(restored.entities.len(), 1);

        // Load into a fresh world.
        let mut world2 = World::new();
        world2.register_reflect::<TestHealth>();
        world2.register_reflect::<TestTeam>();
        let spawned = load_scene_v3_into_world(&mut world2, &restored, &type_reg);
        assert_eq!(spawned.len(), 1);

        let loaded_health = world2
            .get_reflect(spawned[0], "TestHealth")
            .expect("TestHealth should exist");
        let h = loaded_health.as_any().downcast_ref::<TestHealth>().unwrap();
        assert_eq!(h.current, 80.0);
        assert_eq!(h.max, 100.0);

        let loaded_team = world2
            .get_reflect(spawned[0], "TestTeam")
            .expect("TestTeam should exist");
        let t = loaded_team.as_any().downcast_ref::<TestTeam>().unwrap();
        assert_eq!(t.0, 2);
    }

    #[test]
    fn v2_scene_still_loads() {
        // V2 JSON should parse normally.
        let v2_json = r#"{"version": 2, "entities": [{"position": [1, 2, 3], "scale": [1, 1, 1], "rotation": [0, 0, 0, 1], "mesh": "cube", "material": 0}]}"#;
        let scene = SceneFile::from_json(v2_json).unwrap();
        assert_eq!(scene.version, SCENE_VERSION);
        assert_eq!(scene.entities.len(), 1);
    }

    #[test]
    fn v3_json_rejected_by_v2_parser() {
        let v3_json = r#"{"version": 3, "entities": []}"#;
        assert!(SceneFile::from_json(v3_json).is_err());
    }

    #[test]
    fn v3_multiple_components_on_same_entity() {
        #[derive(Clone, Debug, Default, PartialEq, euca_reflect::Reflect)]
        struct Score {
            value: f32,
        }

        let mut world = World::new();
        let mut type_reg = TypeRegistry::new();

        world.register_reflect::<TestHealth>();
        world.register_reflect::<TestTeam>();
        world.register_reflect::<Score>();
        type_reg.register::<TestHealth>();
        type_reg.register::<TestTeam>();
        type_reg.register::<Score>();

        let e = world.spawn(TestHealth {
            current: 50.0,
            max: 50.0,
        });
        world.insert(e, TestTeam(1));
        world.insert(e, Score { value: 42.0 });

        let scene = SceneFileV3::capture(&world);
        assert_eq!(scene.entities[0].components.len(), 3);

        let json = scene.to_json();
        let restored = SceneFileV3::from_json(&json).unwrap();

        let mut world2 = World::new();
        world2.register_reflect::<TestHealth>();
        world2.register_reflect::<TestTeam>();
        world2.register_reflect::<Score>();
        let spawned = load_scene_v3_into_world(&mut world2, &restored, &type_reg);

        let s = world2
            .get_reflect(spawned[0], "Score")
            .unwrap()
            .as_any()
            .downcast_ref::<Score>()
            .unwrap()
            .clone();
        assert_eq!(s.value, 42.0);
    }

    #[test]
    fn unregistered_components_skipped_in_capture() {
        #[derive(Clone, Debug, Default)]
        struct Invisible {
            _data: f32,
        }

        let mut world = World::new();
        let type_reg = TypeRegistry::new();

        world.register_reflect::<TestHealth>();

        let e = world.spawn(TestHealth {
            current: 10.0,
            max: 10.0,
        });
        // Invisible is not registered for reflection.
        world.insert(e, Invisible { _data: 999.0 });

        let scene = SceneFileV3::capture(&world);
        assert_eq!(scene.entities.len(), 1);
        // Only TestHealth should be captured.
        assert!(scene.entities[0].components.contains_key("TestHealth"));
        assert_eq!(scene.entities[0].components.len(), 1);
    }

    #[test]
    fn get_reflect_returns_none_without_component() {
        let mut world = World::new();
        world.register_reflect::<TestHealth>();
        let e = world.spawn_empty();
        assert!(world.get_reflect(e, "TestHealth").is_none());
    }

    #[test]
    fn insert_reflect_returns_false_for_unregistered() {
        let mut world = World::new();
        let e = world.spawn_empty();
        let val: Box<dyn euca_reflect::Reflect> = Box::new(42_f32);
        assert!(!world.insert_reflect(e, "NotRegistered", val));
    }
}
