mod animation;
mod audio;
mod camera;
mod entity;
mod gameplay;
mod nav;
mod particle;
mod scene_auth;
mod sim;
mod templates;
mod ui;

pub use animation::*;
pub use audio::*;
pub use camera::*;
pub use entity::*;
pub use gameplay::*;
pub use nav::*;
pub use particle::*;
pub use scene_auth::*;
pub use sim::*;
pub use templates::*;
pub use ui::*;

use serde::{Deserialize, Serialize};

use euca_ecs::Entity;
use euca_math::Vec3;
use euca_physics::{Collider, ColliderShape, PhysicsBody, RigidBodyType, Velocity};
use euca_render::{MaterialHandle, MeshHandle};
use euca_scene::GlobalTransform;

/// Pre-uploaded mesh/material handles stored as a World resource.
/// Allows the HTTP handler to assign visuals to spawned entities.
#[derive(Clone)]
pub struct DefaultAssets {
    pub meshes: std::collections::HashMap<String, MeshHandle>,
    pub materials: std::collections::HashMap<String, MaterialHandle>,
    pub default_material: MaterialHandle,
}

impl DefaultAssets {
    pub fn mesh(&self, name: &str) -> Option<MeshHandle> {
        self.meshes.get(name).copied()
    }

    /// Resolve a color string to a material handle.
    /// Accepts named colors ("red", "gold") or RGB ("0.5,0.2,0.8") mapped to nearest preset.
    pub fn material(&self, color: &str) -> Option<MaterialHandle> {
        // Try exact name match first
        if let Some(h) = self.materials.get(color) {
            return Some(*h);
        }
        // Try RGB parsing → nearest preset
        let parts: Vec<f32> = color
            .split(',')
            .filter_map(|p| p.trim().parse().ok())
            .collect();
        if parts.len() == 3 {
            return Some(self.nearest_color(parts[0], parts[1], parts[2]));
        }
        None
    }

    fn nearest_color(&self, r: f32, g: f32, b: f32) -> MaterialHandle {
        let presets: &[(&str, [f32; 3])] = &[
            ("red", [0.9, 0.1, 0.1]),
            ("blue", [0.1, 0.2, 0.9]),
            ("green", [0.2, 0.8, 0.2]),
            ("gold", [1.0, 0.84, 0.0]),
            ("silver", [0.95, 0.95, 0.95]),
            ("gray", [0.5, 0.5, 0.5]),
            ("white", [1.0, 1.0, 1.0]),
            ("black", [0.05, 0.05, 0.05]),
            ("yellow", [1.0, 1.0, 0.0]),
            ("cyan", [0.0, 0.9, 0.9]),
            ("magenta", [0.9, 0.0, 0.9]),
            ("orange", [1.0, 0.5, 0.0]),
        ];
        let mut best = "blue";
        let mut best_dist = f32::MAX;
        for (name, rgb) in presets {
            let dist = (r - rgb[0]).powi(2) + (g - rgb[1]).powi(2) + (b - rgb[2]).powi(2);
            if dist < best_dist {
                best_dist = dist;
                best = name;
            }
        }
        self.materials
            .get(best)
            .copied()
            .unwrap_or(self.default_material)
    }
}

/// Named entity templates for quick spawning. Stored as World resource.
#[derive(Clone, Default)]
pub struct TemplateRegistry {
    pub templates: std::collections::HashMap<String, SpawnRequest>,
}

impl TemplateRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

// ── Serializable component representations ──

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TransformData {
    #[serde(default)]
    pub position: Option<[f32; 3]>,
    #[serde(default)]
    pub rotation: Option<[f32; 4]>,
    #[serde(default)]
    pub scale: Option<[f32; 3]>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VelocityData {
    pub linear: [f32; 3],
    #[serde(default)]
    pub angular: [f32; 3],
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "shape")]
pub enum ColliderData {
    Aabb { hx: f32, hy: f32, hz: f32 },
    Sphere { radius: f32 },
    Capsule { radius: f32, half_height: f32 },
}

// ── Rich entity representation ──

#[derive(Serialize)]
pub struct RichEntityData {
    pub id: u32,
    pub generation: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform: Option<TransformData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity: Option<VelocityData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collider: Option<ColliderData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physics_body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health: Option<[f32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dead: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_zone: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projectile: Option<bool>,
}

pub(crate) fn read_entity_data(w: &euca_ecs::World, entity: Entity) -> RichEntityData {
    let transform = w.get::<GlobalTransform>(entity).map(|gt| {
        let t = &gt.0;
        TransformData {
            position: Some([t.translation.x, t.translation.y, t.translation.z]),
            rotation: Some([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]),
            scale: Some([t.scale.x, t.scale.y, t.scale.z]),
        }
    });

    let velocity = w.get::<Velocity>(entity).map(|v| VelocityData {
        linear: [v.linear.x, v.linear.y, v.linear.z],
        angular: [v.angular.x, v.angular.y, v.angular.z],
    });

    let collider = w.get::<Collider>(entity).map(|c| match &c.shape {
        ColliderShape::Aabb { hx, hy, hz } => ColliderData::Aabb {
            hx: *hx,
            hy: *hy,
            hz: *hz,
        },
        ColliderShape::Sphere { radius } => ColliderData::Sphere { radius: *radius },
        ColliderShape::Capsule {
            radius,
            half_height,
        } => ColliderData::Capsule {
            radius: *radius,
            half_height: *half_height,
        },
    });

    let physics_body = w.get::<PhysicsBody>(entity).map(|pb| match pb.body_type {
        RigidBodyType::Dynamic => "Dynamic".to_string(),
        RigidBodyType::Static => "Static".to_string(),
        RigidBodyType::Kinematic => "Kinematic".to_string(),
    });

    let health = w
        .get::<euca_gameplay::Health>(entity)
        .map(|h| [h.current, h.max]);
    let team = w.get::<euca_gameplay::Team>(entity).map(|t| t.0);
    let dead = if w.get::<euca_gameplay::Dead>(entity).is_some() {
        Some(true)
    } else {
        None
    };

    let ai = w
        .get::<euca_gameplay::AiGoal>(entity)
        .map(|g| match &g.behavior {
            euca_gameplay::AiBehavior::Idle => "idle".to_string(),
            euca_gameplay::AiBehavior::Patrol => {
                format!("patrol ({} waypoints)", g.waypoints.len())
            }
            euca_gameplay::AiBehavior::Chase => format!(
                "chase (target: {})",
                g.target
                    .map(|t| t.index().to_string())
                    .unwrap_or("none".into())
            ),
            euca_gameplay::AiBehavior::Flee => "flee".to_string(),
        });
    let trigger_zone = w
        .get::<euca_gameplay::TriggerZone>(entity)
        .map(|tz| format!("{:?}", tz.action));
    let projectile = if w.get::<euca_gameplay::Projectile>(entity).is_some() {
        Some(true)
    } else {
        None
    };

    RichEntityData {
        id: entity.index(),
        generation: entity.generation(),
        transform,
        velocity,
        collider,
        physics_body,
        health,
        team,
        dead,
        ai,
        trigger_zone,
        projectile,
    }
}

// ── Response / Request types ──

#[derive(Serialize)]
pub struct StatusResponse {
    pub engine: &'static str,
    pub version: &'static str,
    pub entity_count: u32,
    pub archetype_count: usize,
    pub tick: u64,
}

#[derive(Serialize)]
pub struct ObserveResponse {
    pub tick: u64,
    pub entity_count: u32,
    pub entities: Vec<RichEntityData>,
}

#[derive(Deserialize)]
pub struct StepRequest {
    #[serde(default = "default_ticks")]
    pub ticks: u64,
}
fn default_ticks() -> u64 {
    1
}

#[derive(Serialize)]
pub struct StepResponse {
    pub ticks_advanced: u64,
    pub new_tick: u64,
    pub entity_count: u32,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct SpawnRequest {
    #[serde(default)]
    pub agent_id: Option<u32>,
    #[serde(default)]
    pub mesh: Option<String>,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub position: Option<[f32; 3]>,
    #[serde(default)]
    pub scale: Option<[f32; 3]>,
    #[serde(default)]
    pub velocity: Option<VelocityData>,
    #[serde(default)]
    pub collider: Option<ColliderData>,
    #[serde(default)]
    pub physics_body: Option<String>,
    #[serde(default)]
    pub health: Option<f32>,
    #[serde(default)]
    pub team: Option<u8>,
    #[serde(default)]
    pub combat: Option<bool>,
}

#[derive(Serialize)]
pub struct SpawnResponse {
    pub entity_id: u32,
    pub entity_generation: u32,
}

#[derive(Deserialize)]
pub struct DespawnRequest {
    pub entity_id: u32,
    pub entity_generation: u32,
    #[serde(default)]
    pub agent_id: Option<u32>,
}

#[derive(Serialize)]
pub struct MessageResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Deserialize)]
pub struct ComponentPatch {
    #[serde(default)]
    pub agent_id: Option<u32>,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub transform: Option<TransformData>,
    #[serde(default)]
    pub velocity: Option<VelocityData>,
    #[serde(default)]
    pub collider: Option<ColliderData>,
    #[serde(default)]
    pub physics_body: Option<String>,
}

// ── Helpers ──

pub(crate) fn find_entity(w: &euca_ecs::World, id: u32) -> Option<Entity> {
    for g in 0..16u32 {
        let e = Entity::from_raw(id, g);
        if w.is_alive(e) {
            return Some(e);
        }
    }
    None
}

pub(crate) fn apply_velocity(w: &mut euca_ecs::World, entity: Entity, v: &VelocityData) {
    let vel = Velocity {
        linear: Vec3::new(v.linear[0], v.linear[1], v.linear[2]),
        angular: Vec3::new(v.angular[0], v.angular[1], v.angular[2]),
    };
    if w.get::<Velocity>(entity).is_some() {
        if let Some(existing) = w.get_mut::<Velocity>(entity) {
            *existing = vel;
        }
    } else {
        w.insert(entity, vel);
    }
}

pub(crate) fn apply_collider(w: &mut euca_ecs::World, entity: Entity, c: &ColliderData) {
    let collider = match c {
        ColliderData::Aabb { hx, hy, hz } => Collider::aabb(*hx, *hy, *hz),
        ColliderData::Sphere { radius } => Collider::sphere(*radius),
        ColliderData::Capsule {
            radius,
            half_height,
        } => Collider::capsule(*radius, *half_height),
    };
    if w.get::<Collider>(entity).is_some() {
        if let Some(existing) = w.get_mut::<Collider>(entity) {
            *existing = collider;
        }
    } else {
        w.insert(entity, collider);
    }
}

pub(crate) fn apply_physics_body(w: &mut euca_ecs::World, entity: Entity, body_type: &str) {
    let pb = match body_type {
        "Static" => PhysicsBody::fixed(),
        "Kinematic" => PhysicsBody {
            body_type: RigidBodyType::Kinematic,
        },
        _ => PhysicsBody::dynamic(),
    };
    if w.get::<PhysicsBody>(entity).is_some() {
        if let Some(existing) = w.get_mut::<PhysicsBody>(entity) {
            *existing = pb;
        }
    } else {
        w.insert(entity, pb);
    }
}
