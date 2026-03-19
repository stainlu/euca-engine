use axum::Json;
use axum::extract::State;

use euca_math::Vec3;

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

/// POST /navmesh/generate — build navmesh from scene colliders
pub async fn navmesh_generate(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let cell_size = req.get("cell_size").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let min_x = req.get("min_x").and_then(|v| v.as_f64()).unwrap_or(-50.0) as f32;
    let min_z = req.get("min_z").and_then(|v| v.as_f64()).unwrap_or(-50.0) as f32;
    let max_x = req.get("max_x").and_then(|v| v.as_f64()).unwrap_or(50.0) as f32;
    let max_z = req.get("max_z").and_then(|v| v.as_f64()).unwrap_or(50.0) as f32;

    let config = euca_nav::GridConfig {
        min: [min_x, min_z],
        max: [max_x, max_z],
        cell_size,
        ground_y: 0.0,
    };

    let (width, height, blocked) = world.with(|w, _| {
        let mesh = euca_nav::navmesh::build_navmesh_from_world(w, config);
        let blocked = mesh.walkable.iter().filter(|&&w| !w).count();
        let width = mesh.width;
        let height = mesh.height;
        w.insert_resource(mesh);
        (width, height, blocked)
    });

    Json(serde_json::json!({
        "ok": true,
        "width": width,
        "height": height,
        "blocked_cells": blocked,
        "total_cells": width * height,
    }))
}

/// POST /path/compute — compute A* path between two points
pub async fn path_compute(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let from = req
        .get("from")
        .and_then(|v| v.as_array())
        .map(|a| {
            Vec3::new(
                a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            )
        })
        .unwrap_or(Vec3::ZERO);

    let to = req
        .get("to")
        .and_then(|v| v.as_array())
        .map(|a| {
            Vec3::new(
                a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            )
        })
        .unwrap_or(Vec3::ZERO);

    let path = world.with_world(|w| {
        let mesh = w.resource::<euca_nav::NavMesh>()?;
        euca_nav::find_path(mesh, from, to)
    });

    match path {
        Some(waypoints) => {
            let wp_json: Vec<_> = waypoints
                .iter()
                .map(|w| serde_json::json!([w.x, w.y, w.z]))
                .collect();
            Json(serde_json::json!({
                "ok": true,
                "waypoints": wp_json,
                "count": wp_json.len(),
            }))
        }
        None => Json(serde_json::json!({
            "ok": false,
            "error": "No path found (no navmesh or blocked)",
        })),
    }
}

/// POST /path/set — set pathfinding goal on an entity
pub async fn path_set(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let target = req
        .get("target")
        .and_then(|v| v.as_array())
        .map(|a| {
            Vec3::new(
                a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            )
        })
        .unwrap_or(Vec3::ZERO);
    let speed = req.get("speed").and_then(|v| v.as_f64()).unwrap_or(5.0) as f32;

    let ok = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return false,
        };

        // Ensure NavAgent exists
        if w.get::<euca_nav::NavAgent>(entity).is_none() {
            w.insert(entity, euca_nav::NavAgent::new(speed));
        }

        // Ensure Velocity exists
        if w.get::<euca_physics::Velocity>(entity).is_none() {
            w.insert(entity, euca_physics::Velocity::default());
        }

        w.insert(entity, euca_nav::PathGoal::new(target));
        true
    });

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!(
                "Set path goal for entity {entity_id} to ({:.1}, {:.1}, {:.1})",
                target.x, target.y, target.z
            )
        } else {
            format!("Entity {entity_id} not found")
        }),
    })
}
