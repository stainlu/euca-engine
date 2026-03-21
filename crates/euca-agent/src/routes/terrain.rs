use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use crate::state::SharedWorld;

use super::MessageResponse;

// ── Request types ──

#[derive(Deserialize)]
pub struct TerrainCreateRequest {
    #[serde(default = "default_terrain_size")]
    pub width: u32,
    #[serde(default = "default_terrain_size")]
    pub height: u32,
    #[serde(default = "default_cell_size")]
    pub cell_size: f32,
}

fn default_terrain_size() -> u32 {
    64
}
fn default_cell_size() -> f32 {
    1.0
}

#[derive(Deserialize)]
pub struct TerrainEditRequest {
    /// Brush operation: "raise", "lower", "flatten", "smooth"
    #[serde(default = "default_op")]
    pub op: String,
    pub x: f32,
    pub z: f32,
    #[serde(default = "default_radius")]
    pub radius: f32,
    #[serde(default = "default_amount")]
    pub amount: f32,
}

fn default_op() -> String {
    "raise".to_string()
}
fn default_radius() -> f32 {
    3.0
}
fn default_amount() -> f32 {
    0.5
}

/// POST /terrain/create — create a flat heightmap terrain
pub async fn terrain_create(
    State(world): State<SharedWorld>,
    Json(req): Json<TerrainCreateRequest>,
) -> Json<serde_json::Value> {
    let (entity_id, world_width, world_depth) = world.with(|w, _| {
        let heightmap = euca_terrain::Heightmap::flat(req.width, req.height)
            .with_cell_size(req.cell_size)
            .with_max_height(50.0);
        let ww = heightmap.world_width();
        let wd = heightmap.world_depth();
        let terrain = euca_terrain::TerrainComponent::new(heightmap, 32);
        let entity = w.spawn(terrain);
        (entity.index(), ww, wd)
    });

    Json(serde_json::json!({
        "ok": true,
        "entity_id": entity_id,
        "width": req.width,
        "height": req.height,
        "cell_size": req.cell_size,
        "world_width": world_width,
        "world_depth": world_depth,
    }))
}

/// POST /terrain/edit — raise/lower/flatten/smooth at a position
pub async fn terrain_edit(
    State(world): State<SharedWorld>,
    Json(req): Json<TerrainEditRequest>,
) -> Json<MessageResponse> {
    let ok = world.with(|w, _| {
        // Find the first entity with a TerrainComponent
        let query =
            euca_ecs::Query::<(euca_ecs::Entity, &mut euca_terrain::TerrainComponent)>::new(w);
        let entities: Vec<euca_ecs::Entity> = query.iter().map(|(e, _)| e).collect();
        drop(query);

        let entity = match entities.first() {
            Some(e) => *e,
            None => return false,
        };

        let terrain = match w.get_mut::<euca_terrain::TerrainComponent>(entity) {
            Some(t) => t,
            None => return false,
        };

        match req.op.as_str() {
            "raise" => euca_terrain::raise_terrain(
                &mut terrain.heightmap,
                req.x,
                req.z,
                req.radius,
                req.amount,
            ),
            "lower" => euca_terrain::lower_terrain(
                &mut terrain.heightmap,
                req.x,
                req.z,
                req.radius,
                req.amount,
            ),
            "flatten" => euca_terrain::flatten_terrain(
                &mut terrain.heightmap,
                req.x,
                req.z,
                req.radius,
                0.5,
                req.amount,
            ),
            "smooth" => euca_terrain::smooth_terrain(
                &mut terrain.heightmap,
                req.x,
                req.z,
                req.radius,
                req.amount,
            ),
            _ => return false,
        }

        true
    });

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!(
                "{} terrain at ({}, {}) r={}",
                req.op, req.x, req.z, req.radius
            )
        } else {
            "No terrain entity found or invalid operation".to_string()
        }),
    })
}
