use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use euca_render::{AlphaMode, Material};

use crate::state::SharedWorld;

use super::{MessageResponse, find_entity};

#[derive(Deserialize)]
pub struct MaterialSetRequest {
    pub entity_id: u32,
    /// Emissive color as [r, g, b]
    #[serde(default)]
    pub emissive: Option<[f32; 3]>,
    /// Alpha mode: "opaque", "blend", or "mask:0.5"
    #[serde(default)]
    pub alpha_mode: Option<String>,
    /// Metallic factor (0.0 - 1.0)
    #[serde(default)]
    pub metallic: Option<f32>,
    /// Roughness factor (0.0 - 1.0)
    #[serde(default)]
    pub roughness: Option<f32>,
    /// Albedo color as [r, g, b, a]
    #[serde(default)]
    pub albedo: Option<[f32; 4]>,
}

/// POST /material/set — set material properties on an entity
pub async fn material_set(
    State(world): State<SharedWorld>,
    Json(req): Json<MaterialSetRequest>,
) -> Json<MessageResponse> {
    let ok = world.with(|w, _| {
        let entity = match find_entity(w, req.entity_id) {
            Some(e) => e,
            None => return false,
        };

        // Get or create the Material component
        let has_material = w.get::<Material>(entity).is_some();
        if !has_material {
            w.insert(entity, Material::default());
        }

        let mat = match w.get_mut::<Material>(entity) {
            Some(m) => m,
            None => return false,
        };

        if let Some(emissive) = req.emissive {
            mat.emissive = emissive;
        }

        if let Some(ref mode_str) = req.alpha_mode {
            mat.alpha_mode = parse_alpha_mode(mode_str);
        }

        if let Some(metallic) = req.metallic {
            mat.metallic = metallic;
        }

        if let Some(roughness) = req.roughness {
            mat.roughness = roughness;
        }

        if let Some(albedo) = req.albedo {
            mat.albedo = albedo;
        }

        true
    });

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("Material updated on entity {}", req.entity_id)
        } else {
            format!("Entity {} not found", req.entity_id)
        }),
    })
}

fn parse_alpha_mode(s: &str) -> AlphaMode {
    match s {
        "opaque" => AlphaMode::Opaque,
        "blend" => AlphaMode::Blend,
        s if s.starts_with("mask:") => {
            let cutoff = s[5..].parse::<f32>().unwrap_or(0.5);
            AlphaMode::Mask { cutoff }
        }
        _ => AlphaMode::Opaque,
    }
}
