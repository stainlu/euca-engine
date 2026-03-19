use axum::Json;
use axum::extract::State;

use euca_math::Vec3;
use euca_physics::Velocity;
use euca_render::{MaterialRef, MeshRenderer};
use euca_scene::{GlobalTransform, LocalTransform};

use crate::state::{Owner, SharedWorld};

use super::{
    DefaultAssets, SpawnRequest, TemplateRegistry, apply_collider, apply_physics_body,
    apply_velocity,
};

/// POST /template/create — define a named entity template
pub async fn template_create(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let name = req
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if name.is_empty() {
        return Json(serde_json::json!({"ok": false, "error": "Template name required"}));
    }

    let template: SpawnRequest = match serde_json::from_value(req.clone()) {
        Ok(t) => t,
        Err(e) => {
            return Json(
                serde_json::json!({"ok": false, "error": format!("Invalid template: {e}")}),
            );
        }
    };

    world.with(|w, _| {
        if let Some(registry) = w.resource_mut::<TemplateRegistry>() {
            registry.templates.insert(name.clone(), template);
        }
    });

    Json(serde_json::json!({"ok": true, "template": name}))
}

/// POST /template/spawn — instantiate a template at a position
pub async fn template_spawn(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let name = req
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let position = req.get("position").and_then(|v| v.as_array()).map(|a| {
        [
            a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        ]
    });

    let template = world.with_world(|w| {
        w.resource::<TemplateRegistry>()
            .and_then(|r| r.templates.get(&name).cloned())
    });

    let template = match template {
        Some(t) => t,
        None => {
            return Json(
                serde_json::json!({"ok": false, "error": format!("Template '{name}' not found")}),
            );
        }
    };

    let mut spawn_req = template;
    if let Some(pos) = position {
        spawn_req.position = Some(pos);
    }

    let entity_id = world.with(|w, _| {
        let pos = spawn_req.position.unwrap_or([0.0, 0.0, 0.0]);
        let scl = spawn_req.scale.unwrap_or([1.0, 1.0, 1.0]);
        let mut transform =
            euca_math::Transform::from_translation(Vec3::new(pos[0], pos[1], pos[2]));
        transform.scale = Vec3::new(scl[0], scl[1], scl[2]);

        let entity = w.spawn(LocalTransform(transform));
        w.insert(entity, GlobalTransform::default());

        if let Some(agent_id) = spawn_req.agent_id {
            w.insert(entity, Owner(agent_id));
        }

        if let Some(assets) = w.resource::<DefaultAssets>().cloned()
            && let Some(mesh_name) = &spawn_req.mesh
            && let Some(mesh) = assets.mesh(mesh_name)
        {
            w.insert(entity, MeshRenderer { mesh });
            let mat = spawn_req
                .color
                .as_deref()
                .and_then(|c| assets.material(c))
                .unwrap_or(assets.default_material);
            w.insert(entity, MaterialRef { handle: mat });
        }

        if let Some(v) = &spawn_req.velocity {
            apply_velocity(w, entity, v);
        }
        if let Some(c) = &spawn_req.collider {
            apply_collider(w, entity, c);
        }
        if let Some(pb) = &spawn_req.physics_body {
            apply_physics_body(w, entity, pb);
            if pb == "Dynamic" && w.get::<Velocity>(entity).is_none() {
                w.insert(entity, Velocity::default());
            }
        }
        if let Some(max_health) = spawn_req.health {
            w.insert(entity, euca_gameplay::Health::new(max_health));
        }
        if let Some(team_id) = spawn_req.team {
            w.insert(entity, euca_gameplay::Team(team_id));
        }
        if spawn_req.combat == Some(true) {
            w.insert(entity, euca_gameplay::AutoCombat::new());
        }

        entity.index()
    });

    Json(serde_json::json!({"ok": true, "entity_id": entity_id, "template": name}))
}

/// GET /template/list — list all defined templates
pub async fn template_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let templates = world.with_world(|w| {
        w.resource::<TemplateRegistry>()
            .map(|r| r.templates.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default()
    });
    Json(serde_json::json!({"templates": templates, "count": templates.len()}))
}
