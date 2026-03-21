use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use euca_render::VolumetricFogSettings;

use crate::state::SharedWorld;

#[derive(Deserialize)]
pub struct FogUpdateRequest {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub density: Option<f32>,
    #[serde(default)]
    pub scattering: Option<f32>,
    #[serde(default)]
    pub absorption: Option<f32>,
    #[serde(default)]
    pub height_falloff: Option<f32>,
    #[serde(default)]
    pub max_distance: Option<f32>,
    #[serde(default)]
    pub color: Option<[f32; 3]>,
    #[serde(default)]
    pub light_contribution: Option<f32>,
}

/// GET /fog/settings -- get current volumetric fog settings
pub async fn fog_get(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let settings = world.with_world(|w| w.resource::<VolumetricFogSettings>().cloned());

    match settings {
        Some(s) => Json(serde_json::json!({
            "ok": true,
            "enabled": s.enabled,
            "density": s.density,
            "scattering": s.scattering,
            "absorption": s.absorption,
            "height_falloff": s.height_falloff,
            "max_distance": s.max_distance,
            "color": s.color,
            "light_contribution": s.light_contribution,
        })),
        None => Json(serde_json::json!({
            "ok": false,
            "error": "VolumetricFogSettings resource not found",
        })),
    }
}

/// POST /fog/settings -- update volumetric fog settings
pub async fn fog_set(
    State(world): State<SharedWorld>,
    Json(req): Json<FogUpdateRequest>,
) -> Json<serde_json::Value> {
    let ok = world.with(|w, _| {
        if w.resource::<VolumetricFogSettings>().is_none() {
            w.insert_resource(VolumetricFogSettings::default());
        }

        let settings = match w.resource_mut::<VolumetricFogSettings>() {
            Some(s) => s,
            None => return false,
        };

        if let Some(v) = req.enabled {
            settings.enabled = v;
        }
        if let Some(v) = req.density {
            settings.density = v;
        }
        if let Some(v) = req.scattering {
            settings.scattering = v;
        }
        if let Some(v) = req.absorption {
            settings.absorption = v;
        }
        if let Some(v) = req.height_falloff {
            settings.height_falloff = v;
        }
        if let Some(v) = req.max_distance {
            settings.max_distance = v;
        }
        if let Some(v) = req.color {
            settings.color = v;
        }
        if let Some(v) = req.light_contribution {
            settings.light_contribution = v;
        }

        true
    });

    if ok {
        fog_get(State(world)).await
    } else {
        Json(serde_json::json!({
            "ok": false,
            "error": "Failed to update VolumetricFogSettings",
        }))
    }
}
