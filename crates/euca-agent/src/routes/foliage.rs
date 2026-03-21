use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use euca_math::Vec3;
use euca_render::{FoliageLayer, FoliageLayers, scatter_foliage};

use crate::state::SharedWorld;

use super::DefaultAssets;

#[derive(Deserialize)]
pub struct FoliageScatterRequest {
    #[serde(default = "default_mesh")]
    pub mesh_name: String,
    #[serde(default = "default_density")]
    pub density: f32,
    #[serde(default = "default_area_min")]
    pub area_min: [f32; 3],
    #[serde(default = "default_area_max")]
    pub area_max: [f32; 3],
    #[serde(default = "default_min_scale")]
    pub min_scale: f32,
    #[serde(default = "default_max_scale")]
    pub max_scale: f32,
    #[serde(default = "default_max_distance")]
    pub max_distance: f32,
}

fn default_mesh() -> String {
    "cube".to_string()
}
fn default_density() -> f32 {
    0.5
}
fn default_area_min() -> [f32; 3] {
    [-20.0, 0.0, -20.0]
}
fn default_area_max() -> [f32; 3] {
    [20.0, 0.0, 20.0]
}
fn default_min_scale() -> f32 {
    0.8
}
fn default_max_scale() -> f32 {
    1.2
}
fn default_max_distance() -> f32 {
    100.0
}

/// POST /foliage/scatter
pub async fn foliage_scatter(
    State(world): State<SharedWorld>,
    Json(req): Json<FoliageScatterRequest>,
) -> Json<serde_json::Value> {
    let result = world.with(|w, _| {
        let assets = match w.resource::<DefaultAssets>() {
            Some(a) => a.clone(),
            None => return Err("DefaultAssets resource not found"),
        };
        let mesh = match assets.mesh(&req.mesh_name) {
            Some(m) => m,
            None => return Err("Unknown mesh name"),
        };
        let material = assets.material("green").unwrap_or(assets.default_material);
        let mut layer = FoliageLayer {
            mesh,
            material,
            density: req.density,
            min_scale: req.min_scale,
            max_scale: req.max_scale,
            max_distance: req.max_distance,
            instances: Vec::new(),
        };
        let area_min = Vec3::new(req.area_min[0], req.area_min[1], req.area_min[2]);
        let area_max = Vec3::new(req.area_max[0], req.area_max[1], req.area_max[2]);
        scatter_foliage(&mut layer, area_min, area_max, 42);
        let instance_count = layer.instances.len();
        if let Some(layers) = w.resource_mut::<FoliageLayers>() {
            layers.layers.push(layer);
        } else {
            w.insert_resource(FoliageLayers {
                layers: vec![layer],
            });
        }
        Ok(instance_count)
    });
    match result {
        Ok(count) => Json(
            serde_json::json!({"ok": true, "instance_count": count, "mesh": req.mesh_name, "density": req.density}),
        ),
        Err(msg) => Json(serde_json::json!({"ok": false, "error": msg})),
    }
}

/// GET /foliage/list
pub async fn foliage_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let layers_info = world.with_world(|w| {
        let layers = match w.resource::<FoliageLayers>() {
            Some(l) => l,
            None => return vec![],
        };
        layers.layers.iter().enumerate().map(|(i, layer)| {
            serde_json::json!({"index": i, "instance_count": layer.instances.len(), "density": layer.density, "max_distance": layer.max_distance})
        }).collect::<Vec<_>>()
    });
    Json(serde_json::json!({"layers": layers_info, "count": layers_info.len()}))
}
