use euca_ecs::{Entity, Query, World};
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
    pub fn from_json(json: &str) -> Result<Self, String> {
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
}
