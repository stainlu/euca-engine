use euca_ecs::{Entity, Query, World};
use euca_render::{MaterialRef, MeshRenderer};
#[allow(unused_imports)]
use euca_scene::{GlobalTransform, LocalTransform};
use serde::{Deserialize, Serialize};

/// A serializable scene file format.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SceneFile {
    pub entities: Vec<SceneEntity>,
}

/// A serializable entity with named components.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SceneEntity {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    /// Mesh type name ("cube", "sphere", or "none").
    pub mesh: String,
    /// Material index (refers to upload order in the editor).
    pub material: u32,
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
                SceneEntity {
                    position: [t.translation.x, t.translation.y, t.translation.z],
                    scale: [t.scale.x, t.scale.y, t.scale.z],
                    mesh,
                    material,
                }
            })
            .collect();
        Self { entities }
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Scene serialization failed")
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("Scene deserialization failed: {e}"))
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
) -> Vec<euca_ecs::Entity> {
    let mut spawned = Vec::new();
    for se in &scene.entities {
        let pos = euca_math::Vec3::new(se.position[0], se.position[1], se.position[2]);
        let scl = euca_math::Vec3::new(se.scale[0], se.scale[1], se.scale[2]);
        let mut transform = euca_math::Transform::from_translation(pos);
        transform.scale = scl;

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

        spawned.push(e);
    }
    spawned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_file_roundtrip() {
        let scene = SceneFile {
            entities: vec![SceneEntity {
                position: [1.0, 2.0, 3.0],
                scale: [1.0, 1.0, 1.0],
                mesh: "cube".to_string(),
                material: 0,
            }],
        };
        let json = scene.to_json();
        let restored = SceneFile::from_json(&json).unwrap();
        assert_eq!(restored.entities.len(), 1);
        assert_eq!(restored.entities[0].position, [1.0, 2.0, 3.0]);
    }
}
