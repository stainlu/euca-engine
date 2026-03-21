use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use euca_render::{PostProcessSettings, RenderQuality};

use crate::state::SharedWorld;

#[derive(Deserialize)]
pub struct PostProcessUpdateRequest {
    #[serde(default)]
    pub ssao_enabled: Option<bool>,
    #[serde(default)]
    pub ssao_radius: Option<f32>,
    #[serde(default)]
    pub ssao_intensity: Option<f32>,
    #[serde(default)]
    pub fxaa_enabled: Option<bool>,
    #[serde(default)]
    pub bloom_enabled: Option<bool>,
    #[serde(default)]
    pub bloom_threshold: Option<f32>,
    #[serde(default)]
    pub exposure: Option<f32>,
    #[serde(default)]
    pub contrast: Option<f32>,
    #[serde(default)]
    pub saturation: Option<f32>,
    #[serde(default)]
    pub temperature: Option<f32>,
}

/// GET /postprocess/settings — get current post-processing settings
pub async fn postprocess_get(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let settings = world.with_world(|w| w.resource::<PostProcessSettings>().cloned());

    match settings {
        Some(s) => Json(serde_json::json!({
            "ok": true,
            "ssao_enabled": s.ssao_enabled,
            "ssao_radius": s.ssao_radius,
            "ssao_intensity": s.ssao_intensity,
            "fxaa_enabled": s.fxaa_enabled,
            "bloom_enabled": s.bloom_enabled,
            "bloom_threshold": s.bloom_threshold,
            "exposure": s.exposure,
            "contrast": s.contrast,
            "saturation": s.saturation,
            "temperature": s.temperature,
        })),
        None => Json(serde_json::json!({
            "ok": false,
            "error": "PostProcessSettings resource not found",
        })),
    }
}

/// POST /postprocess/settings — update post-processing settings
pub async fn postprocess_set(
    State(world): State<SharedWorld>,
    Json(req): Json<PostProcessUpdateRequest>,
) -> Json<serde_json::Value> {
    let ok = world.with(|w, _| {
        // Ensure the resource exists
        if w.resource::<PostProcessSettings>().is_none() {
            w.insert_resource(PostProcessSettings::default());
        }

        let settings = match w.resource_mut::<PostProcessSettings>() {
            Some(s) => s,
            None => return false,
        };

        if let Some(v) = req.ssao_enabled {
            settings.ssao_enabled = v;
        }
        if let Some(v) = req.ssao_radius {
            settings.ssao_radius = v;
        }
        if let Some(v) = req.ssao_intensity {
            settings.ssao_intensity = v;
        }
        if let Some(v) = req.fxaa_enabled {
            settings.fxaa_enabled = v;
        }
        if let Some(v) = req.bloom_enabled {
            settings.bloom_enabled = v;
        }
        if let Some(v) = req.bloom_threshold {
            settings.bloom_threshold = v;
        }
        if let Some(v) = req.exposure {
            settings.exposure = v;
        }
        if let Some(v) = req.contrast {
            settings.contrast = v;
        }
        if let Some(v) = req.saturation {
            settings.saturation = v;
        }
        if let Some(v) = req.temperature {
            settings.temperature = v;
        }

        true
    });

    if ok {
        // Return the updated settings
        postprocess_get(State(world)).await
    } else {
        Json(serde_json::json!({
            "ok": false,
            "error": "Failed to update PostProcessSettings",
        }))
    }
}

#[derive(Deserialize)]
pub struct PresetRequest {
    pub quality: String,
}

/// POST /postprocess/preset — apply a named quality preset
pub async fn postprocess_preset(
    State(world): State<SharedWorld>,
    Json(req): Json<PresetRequest>,
) -> Json<serde_json::Value> {
    let quality = match RenderQuality::from_name(&req.quality) {
        Some(q) => q,
        None => {
            return Json(serde_json::json!({
                "ok": false,
                "error": format!(
                    "Unknown quality preset '{}'. Use low, medium, high, or ultra.",
                    req.quality
                ),
            }));
        }
    };

    let settings = quality.to_settings();

    let ok = world.with(|w, _| {
        if w.resource::<PostProcessSettings>().is_none() {
            w.insert_resource(settings.clone());
        } else if let Some(existing) = w.resource_mut::<PostProcessSettings>() {
            *existing = settings.clone();
        } else {
            return false;
        }
        true
    });

    if ok {
        postprocess_get(State(world)).await
    } else {
        Json(serde_json::json!({
            "ok": false,
            "error": "Failed to apply quality preset",
        }))
    }
}
