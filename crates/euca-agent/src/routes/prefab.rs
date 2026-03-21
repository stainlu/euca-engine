use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use euca_scene::PrefabRegistry;

use crate::state::SharedWorld;

#[derive(Deserialize)]
pub struct PrefabSpawnRequest {
    pub name: String,
    #[serde(default)]
    pub position: Option<[f32; 3]>,
}

/// POST /prefab/spawn — spawn a named prefab at a position
pub async fn prefab_spawn(
    State(world): State<SharedWorld>,
    Json(req): Json<PrefabSpawnRequest>,
) -> Json<serde_json::Value> {
    let result = world.with(|w, _| {
        // Use WorldPrefabExt to temporarily borrow the registry
        let entity = euca_scene::WorldPrefabExt::spawn_prefab(w, &req.name)?;

        // Override position if provided
        if let Some(pos) = req.position {
            let transform = euca_math::Transform::from_translation(euca_math::Vec3::new(
                pos[0], pos[1], pos[2],
            ));
            if let Some(lt) = w.get_mut::<euca_scene::LocalTransform>(entity) {
                lt.0 = transform;
            } else {
                w.insert(entity, euca_scene::LocalTransform(transform));
            }
            if let Some(gt) = w.get_mut::<euca_scene::GlobalTransform>(entity) {
                gt.0 = transform;
            } else {
                w.insert(entity, euca_scene::GlobalTransform(transform));
            }
        }

        Some(entity)
    });

    match result {
        Some(entity) => Json(serde_json::json!({
            "ok": true,
            "entity_id": entity.index(),
            "entity_generation": entity.generation(),
            "name": req.name,
        })),
        None => Json(serde_json::json!({
            "ok": false,
            "error": format!("Prefab '{}' not found in registry", req.name),
        })),
    }
}

/// GET /prefab/list — list registered prefabs
pub async fn prefab_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let names = world.with_world(|w| {
        w.resource::<PrefabRegistry>()
            .map(|registry| {
                registry
                    .names()
                    .map(|name| serde_json::json!(name))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });

    Json(serde_json::json!({
        "ok": true,
        "count": names.len(),
        "prefabs": names,
    }))
}
