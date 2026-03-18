use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use euca_ecs::{Entity, Query};
use euca_math::Vec3;
use euca_physics::{Collider, ColliderShape, PhysicsBody, RigidBodyType, Velocity};
use euca_render::{MaterialHandle, MaterialRef, MeshHandle, MeshRenderer};
use euca_scene::{GlobalTransform, LocalTransform};

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
        // Known preset RGB values (must match what's uploaded in editor setup_scene)
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

use crate::state::{Owner, SharedWorld};

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
}

fn read_entity_data(w: &euca_ecs::World, entity: Entity) -> RichEntityData {
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

#[derive(Deserialize)]
pub struct SpawnRequest {
    /// Agent that owns this entity. If set, only this agent can despawn/modify it.
    #[serde(default)]
    pub agent_id: Option<u32>,
    /// Mesh to attach: "cube", "sphere", "plane", "cylinder", "cone"
    #[serde(default)]
    pub mesh: Option<String>,
    /// Color: named ("red", "gold") or RGB ("0.5,0.2,0.8")
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
    /// Initial health (adds Health component)
    #[serde(default)]
    pub health: Option<f32>,
    /// Team ID (adds Team component)
    #[serde(default)]
    pub team: Option<u8>,
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
    /// Agent requesting despawn. Must match entity's Owner if set.
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
    /// Agent requesting the patch. Must match entity's Owner if set.
    #[serde(default)]
    pub agent_id: Option<u32>,
    /// Color: named ("red", "gold") or RGB ("0.5,0.2,0.8")
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

fn find_entity(w: &euca_ecs::World, id: u32) -> Option<Entity> {
    for g in 0..16u32 {
        let e = Entity::from_raw(id, g);
        if w.is_alive(e) {
            return Some(e);
        }
    }
    None
}

fn apply_velocity(w: &mut euca_ecs::World, entity: Entity, v: &VelocityData) {
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

fn apply_collider(w: &mut euca_ecs::World, entity: Entity, c: &ColliderData) {
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

fn apply_physics_body(w: &mut euca_ecs::World, entity: Entity, body_type: &str) {
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

// ── Route handlers ──

/// GET / — engine status
pub async fn status(State(world): State<SharedWorld>) -> Json<StatusResponse> {
    let resp = world.with_world(|w| StatusResponse {
        engine: "Euca Engine",
        version: env!("CARGO_PKG_VERSION"),
        entity_count: w.entity_count(),
        archetype_count: w.archetype_count(),
        tick: w.current_tick(),
    });
    Json(resp)
}

/// POST /observe — query full world state
pub async fn observe(State(world): State<SharedWorld>) -> Json<ObserveResponse> {
    let resp = world.with_world(|w| {
        let entities: Vec<RichEntityData> = {
            let query = Query::<Entity>::new(w);
            query.iter().map(|e| read_entity_data(w, e)).collect()
        };
        ObserveResponse {
            tick: w.current_tick(),
            entity_count: w.entity_count(),
            entities,
        }
    });
    Json(resp)
}

/// GET /entities/:id — query single entity
pub async fn get_entity(
    State(world): State<SharedWorld>,
    Path(id): Path<u32>,
) -> Result<Json<RichEntityData>, StatusCode> {
    let result = world.with_world(|w| find_entity(w, id).map(|e| read_entity_data(w, e)));
    match result {
        Some(data) => Ok(Json(data)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// POST /entities/:id/components — add/update components on an entity
pub async fn patch_entity(
    State(world): State<SharedWorld>,
    Path(id): Path<u32>,
    Json(patch): Json<ComponentPatch>,
) -> Result<Json<MessageResponse>, StatusCode> {
    world.with(|w, _| {
        let entity = match find_entity(w, id) {
            Some(e) => e,
            None => return Err(StatusCode::NOT_FOUND),
        };

        // Ownership check
        if let Some(owner) = w.get::<Owner>(entity) {
            match patch.agent_id {
                Some(aid) if aid == owner.0 => {}
                _ => {
                    return Ok(Json(MessageResponse {
                        ok: false,
                        message: Some("Permission denied: entity owned by another agent".into()),
                    }));
                }
            }
        }

        if let Some(t) = &patch.transform
            && let Some(lt) = w.get_mut::<LocalTransform>(entity)
        {
            if let Some(pos) = t.position {
                lt.0.translation = Vec3::new(pos[0], pos[1], pos[2]);
            }
            if let Some(scl) = t.scale {
                lt.0.scale = Vec3::new(scl[0], scl[1], scl[2]);
            }
            if let Some(rot) = t.rotation {
                lt.0.rotation = euca_math::Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]);
            }
        }
        if let Some(color) = &patch.color
            && let Some(assets) = w.resource::<DefaultAssets>().cloned()
            && let Some(mat) = assets.material(color)
        {
            if w.get::<MaterialRef>(entity).is_some() {
                if let Some(mr) = w.get_mut::<MaterialRef>(entity) {
                    mr.handle = mat;
                }
            } else {
                w.insert(entity, MaterialRef { handle: mat });
            }
        }
        if let Some(v) = &patch.velocity {
            apply_velocity(w, entity, v);
        }
        if let Some(c) = &patch.collider {
            apply_collider(w, entity, c);
        }
        if let Some(pb) = &patch.physics_body {
            apply_physics_body(w, entity, pb);
        }

        Ok(Json(MessageResponse {
            ok: true,
            message: None,
        }))
    })
}

/// POST /step — advance simulation
pub async fn step(
    State(world): State<SharedWorld>,
    Json(req): Json<StepRequest>,
) -> Json<StepResponse> {
    let resp = world.with(|w, schedule| {
        let ticks = req.ticks.min(10000);
        for _ in 0..ticks {
            schedule.run(w);
        }
        StepResponse {
            ticks_advanced: ticks,
            new_tick: w.current_tick(),
            entity_count: w.entity_count(),
        }
    });
    Json(resp)
}

/// POST /spawn — create entity with optional components
pub async fn spawn(
    State(world): State<SharedWorld>,
    Json(req): Json<SpawnRequest>,
) -> (StatusCode, Json<SpawnResponse>) {
    let resp = world.with(|w, _| {
        let pos = req.position.unwrap_or([0.0, 0.0, 0.0]);
        let scl = req.scale.unwrap_or([1.0, 1.0, 1.0]);
        let mut transform =
            euca_math::Transform::from_translation(Vec3::new(pos[0], pos[1], pos[2]));
        transform.scale = Vec3::new(scl[0], scl[1], scl[2]);

        let entity = w.spawn(LocalTransform(transform));
        w.insert(entity, GlobalTransform::default());

        // Track ownership if agent_id provided
        if let Some(agent_id) = req.agent_id {
            w.insert(entity, Owner(agent_id));
        }

        // Assign mesh + material from pre-uploaded assets
        if let Some(assets) = w.resource::<DefaultAssets>().cloned()
            && let Some(mesh_name) = &req.mesh
            && let Some(mesh) = assets.mesh(mesh_name)
        {
            w.insert(entity, MeshRenderer { mesh });
            // Material: use --color if provided, else default
            let mat = req
                .color
                .as_deref()
                .and_then(|c| assets.material(c))
                .unwrap_or(assets.default_material);
            w.insert(entity, MaterialRef { handle: mat });
        }

        if let Some(v) = &req.velocity {
            apply_velocity(w, entity, v);
        }
        if let Some(c) = &req.collider {
            apply_collider(w, entity, c);
        }
        if let Some(pb) = &req.physics_body {
            apply_physics_body(w, entity, pb);
            // Dynamic bodies need Velocity for physics simulation (gravity)
            if pb == "Dynamic" && w.get::<Velocity>(entity).is_none() {
                w.insert(entity, Velocity::default());
            }
        }

        // Gameplay components
        if let Some(max_health) = req.health {
            w.insert(entity, euca_gameplay::Health::new(max_health));
        }
        if let Some(team_id) = req.team {
            w.insert(entity, euca_gameplay::Team(team_id));
        }

        SpawnResponse {
            entity_id: entity.index(),
            entity_generation: entity.generation(),
        }
    });
    (StatusCode::CREATED, Json(resp))
}

/// POST /despawn — remove an entity
pub async fn despawn(
    State(world): State<SharedWorld>,
    Json(req): Json<DespawnRequest>,
) -> Json<MessageResponse> {
    let resp = world.with(|w, _| {
        let entity = Entity::from_raw(req.entity_id, req.entity_generation);

        // Ownership check: if entity has an Owner, agent_id must match
        if let Some(owner) = w.get::<Owner>(entity) {
            match req.agent_id {
                Some(aid) if aid == owner.0 => {} // authorized
                _ => {
                    return MessageResponse {
                        ok: false,
                        message: Some("Permission denied: entity owned by another agent".into()),
                    };
                }
            }
        }

        if w.despawn(entity) {
            MessageResponse {
                ok: true,
                message: None,
            }
        } else {
            MessageResponse {
                ok: false,
                message: Some("Entity not found or already despawned".into()),
            }
        }
    });
    Json(resp)
}

/// POST /reset — reset the world
pub async fn reset(State(world): State<SharedWorld>) -> Json<MessageResponse> {
    let resp = world.with(|w, _| {
        let entities: Vec<Entity> = {
            let query = Query::<Entity>::new(w);
            query.iter().collect()
        };
        for entity in entities {
            w.despawn(entity);
        }
        MessageResponse {
            ok: true,
            message: Some(format!("World reset. Tick: {}", w.current_tick())),
        }
    });
    Json(resp)
}

/// GET /camera — get current camera state
pub async fn camera_get(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let data = world.with_world(|w| {
        w.resource::<euca_render::Camera>().map(|cam| {
            serde_json::json!({
                "eye": [cam.eye.x, cam.eye.y, cam.eye.z],
                "target": [cam.target.x, cam.target.y, cam.target.z],
                "fov_y": cam.fov_y,
                "orthographic": cam.orthographic,
                "ortho_size": cam.ortho_size,
            })
        })
    });
    Json(data.unwrap_or(serde_json::json!({"error": "No camera"})))
}

/// POST /camera — set camera position and target
pub async fn camera_set(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    world.with(|w, _| {
        if let Some(cam) = w.resource_mut::<euca_render::Camera>() {
            if let Some(eye) = req.get("eye").and_then(|v| v.as_array())
                && eye.len() == 3
            {
                cam.eye = Vec3::new(
                    eye[0].as_f64().unwrap_or(0.0) as f32,
                    eye[1].as_f64().unwrap_or(0.0) as f32,
                    eye[2].as_f64().unwrap_or(0.0) as f32,
                );
            }
            if let Some(target) = req.get("target").and_then(|v| v.as_array())
                && target.len() == 3
            {
                cam.target = Vec3::new(
                    target[0].as_f64().unwrap_or(0.0) as f32,
                    target[1].as_f64().unwrap_or(0.0) as f32,
                    target[2].as_f64().unwrap_or(0.0) as f32,
                );
            }
        }
    });
    // Set CameraOverride so editor doesn't override with mouse orbit
    world.with_world(|w| {
        if let Some(co) = w.resource::<crate::control::CameraOverride>() {
            co.set();
        }
    });

    Json(MessageResponse {
        ok: true,
        message: Some("Camera updated".into()),
    })
}

/// POST /camera/view — apply a named view preset (top, front, right, etc.)
pub async fn camera_view(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let view = req
        .get("view")
        .and_then(|v| v.as_str())
        .unwrap_or("perspective");

    let ok = world.with(|w, _| {
        if let Some(cam) = w.resource_mut::<euca_render::Camera>() {
            cam.apply_preset(view)
        } else {
            false
        }
    });

    if ok {
        world.with_world(|w| {
            if let Some(co) = w.resource::<crate::control::CameraOverride>() {
                co.set();
            }
        });
    }

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("Camera set to {view} view")
        } else {
            format!("Unknown view: {view}. Use: top, front, back, right, left, perspective")
        }),
    })
}

/// POST /camera/focus — focus camera on a specific entity
pub async fn camera_focus(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req
        .get("entity_id")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let entity_id = match entity_id {
        Some(id) => id,
        None => {
            return Json(MessageResponse {
                ok: false,
                message: Some("Missing entity_id".into()),
            });
        }
    };

    let result = world.with(|w, _| {
        let entity = find_entity(w, entity_id)?;
        let pos = w
            .get::<GlobalTransform>(entity)
            .map(|gt| gt.0.translation)?;
        let cam = w.resource_mut::<euca_render::Camera>()?;
        cam.target = pos;
        // Position camera at a reasonable distance from the entity
        let offset = cam.eye - cam.target;
        let dist = offset.length().clamp(5.0, 20.0);
        let dir = if offset.length() > 0.001 {
            offset.normalize()
        } else {
            Vec3::new(0.6, 0.5, 0.6).normalize()
        };
        cam.eye = pos + dir * dist;
        cam.orthographic = false;
        Some(pos)
    });

    if let Some(pos) = result {
        world.with_world(|w| {
            if let Some(co) = w.resource::<crate::control::CameraOverride>() {
                co.set();
            }
        });
        Json(MessageResponse {
            ok: true,
            message: Some(format!(
                "Focused on entity {} at ({:.1}, {:.1}, {:.1})",
                entity_id, pos.x, pos.y, pos.z
            )),
        })
    } else {
        Json(MessageResponse {
            ok: false,
            message: Some(format!("Entity {entity_id} not found")),
        })
    }
}

/// POST /scene/save — save current world state as JSON
pub async fn scene_save(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let path = req
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("scene.json")
        .to_string();

    // Capture all entities as JSON via the observe handler's data format
    let scene_data = world.with_world(|w| {
        let entities: Vec<RichEntityData> = {
            let query = Query::<Entity>::new(w);
            query.iter().map(|e| read_entity_data(w, e)).collect()
        };
        serde_json::json!({
            "version": 1,
            "tick": w.current_tick(),
            "entities": entities,
        })
    });

    match std::fs::write(&path, serde_json::to_string_pretty(&scene_data).unwrap()) {
        Ok(()) => Json(MessageResponse {
            ok: true,
            message: Some(format!("Scene saved to {path}")),
        }),
        Err(e) => Json(MessageResponse {
            ok: false,
            message: Some(format!("Save failed: {e}")),
        }),
    }
}

/// POST /scene/load — load scene from JSON file
pub async fn scene_load(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let path = req
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("scene.json")
        .to_string();

    let data = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            return Json(MessageResponse {
                ok: false,
                message: Some(format!("Cannot read {path}: {e}")),
            });
        }
    };

    let scene: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => {
            return Json(MessageResponse {
                ok: false,
                message: Some(format!("Invalid JSON: {e}")),
            });
        }
    };

    let entities = scene["entities"].as_array();
    let count = entities.map(|e| e.len()).unwrap_or(0);

    world.with(|w, _| {
        // Clear existing entities
        let existing: Vec<Entity> = {
            let query = Query::<Entity>::new(w);
            query.iter().collect()
        };
        for entity in existing {
            w.despawn(entity);
        }

        // Recreate entities from scene data
        if let Some(entities) = entities {
            for ent in entities {
                let pos = ent["transform"]["position"]
                    .as_array()
                    .map(|a| {
                        [
                            a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                            a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                            a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        ]
                    })
                    .unwrap_or([0.0, 0.0, 0.0]);

                let mut transform =
                    euca_math::Transform::from_translation(Vec3::new(pos[0], pos[1], pos[2]));

                if let Some(scl) = ent["transform"]["scale"].as_array() {
                    transform.scale = Vec3::new(
                        scl.first().and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                        scl.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                        scl.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                    );
                }

                let entity = w.spawn(LocalTransform(transform));
                w.insert(entity, GlobalTransform::default());

                if let Some(pb) = ent["physics_body"].as_str() {
                    apply_physics_body(w, entity, pb);
                }
            }
        }
    });

    Json(MessageResponse {
        ok: true,
        message: Some(format!("Loaded {count} entities from {path}")),
    })
}

/// POST /play — start simulation
pub async fn play(State(world): State<SharedWorld>) -> Json<MessageResponse> {
    world.with_world(|w| {
        if let Some(ctrl) = w.resource::<crate::control::EngineControl>() {
            ctrl.set_playing(true);
        }
    });
    Json(MessageResponse {
        ok: true,
        message: Some("Simulation playing".into()),
    })
}

/// POST /pause — pause simulation
pub async fn pause(State(world): State<SharedWorld>) -> Json<MessageResponse> {
    world.with_world(|w| {
        if let Some(ctrl) = w.resource::<crate::control::EngineControl>() {
            ctrl.set_playing(false);
        }
    });
    Json(MessageResponse {
        ok: true,
        message: Some("Simulation paused".into()),
    })
}

/// POST /screenshot — capture 3D viewport as PNG
pub async fn screenshot(
    State(world): State<SharedWorld>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let rx = world.with_world(|w| {
        w.resource::<crate::control::ScreenshotChannel>()
            .map(|ch| ch.request())
    });

    let rx = match rx {
        Some(rx) => rx,
        None => {
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
    };

    // Wait for the render loop to capture the frame (timeout 2s)
    match tokio::time::timeout(std::time::Duration::from_secs(2), rx).await {
        Ok(Ok(png_bytes)) => {
            // Write to temp file
            let path = std::env::temp_dir().join(format!(
                "euca_screenshot_{}.png",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis()
            ));
            if let Err(e) = std::fs::write(&path, &png_bytes) {
                log::error!("Failed to write screenshot: {e}");
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
            Ok(Json(serde_json::json!({
                "ok": true,
                "path": path.to_string_lossy(),
                "size_bytes": png_bytes.len(),
            })))
        }
        Ok(Err(_)) => {
            // Sender dropped (render loop didn't capture)
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(_) => {
            // Timeout
            Err(StatusCode::GATEWAY_TIMEOUT)
        }
    }
}

/// POST /auth/login — authenticate via nit Ed25519 signature
pub async fn auth_login(
    State(world): State<SharedWorld>,
    Json(payload): Json<crate::auth::LoginPayload>,
) -> Result<Json<crate::auth::LoginResponse>, (StatusCode, Json<crate::auth::AuthError>)> {
    let auth_store = world.with_world(|w| w.resource::<crate::auth::AuthStore>().cloned());

    let auth_store = match auth_store {
        Some(store) => store,
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(crate::auth::AuthError {
                    ok: false,
                    error: "Auth not configured".into(),
                }),
            ));
        }
    };

    match auth_store.login(&payload) {
        Ok(token) => Ok(Json(crate::auth::LoginResponse {
            ok: true,
            session_token: token,
            agent_id: payload.agent_id,
        })),
        Err(e) => Err((
            StatusCode::UNAUTHORIZED,
            Json(crate::auth::AuthError {
                ok: false,
                error: e,
            }),
        )),
    }
}

/// GET /auth/status — check current auth session
pub async fn auth_status(
    State(world): State<SharedWorld>,
    headers: axum::http::HeaderMap,
) -> Json<serde_json::Value> {
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    let auth_store = world.with_world(|w| w.resource::<crate::auth::AuthStore>().cloned());

    if let (Some(token), Some(store)) = (token, auth_store)
        && let Some(agent_id) = store.validate(token)
    {
        return Json(serde_json::json!({
            "authenticated": true,
            "agent_id": agent_id,
        }));
    }

    Json(serde_json::json!({
        "authenticated": false,
    }))
}

/// GET /schema — dynamic schema: all component types and actions
pub async fn schema() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "components": {
            "LocalTransform": {
                "fields": {"position": "[f32; 3]", "rotation": "[f32; 4] (xyzw)", "scale": "[f32; 3]"}
            },
            "GlobalTransform": {
                "fields": {"position": "[f32; 3]", "rotation": "[f32; 4] (xyzw)", "scale": "[f32; 3]"},
                "note": "Read-only. Computed from LocalTransform hierarchy."
            },
            "Velocity": {
                "fields": {"linear": "[f32; 3]", "angular": "[f32; 3]"}
            },
            "PhysicsBody": {
                "fields": {"body_type": "Dynamic | Static | Kinematic"}
            },
            "Collider": {
                "variants": {
                    "Aabb": {"hx": "f32", "hy": "f32", "hz": "f32"},
                    "Sphere": {"radius": "f32"}
                }
            }
        },
        "endpoints": {
            "GET /": "Engine status",
            "POST /observe": "Full world state (all entities with all components)",
            "GET /entities/:id": "Single entity with all components",
            "POST /entities/:id/components": "Add/update components on entity",
            "POST /spawn": "Create entity with optional components (position, scale, velocity, collider, physics_body)",
            "POST /despawn": "Remove entity by id + generation",
            "POST /step": "Advance simulation N ticks",
            "POST /reset": "Despawn all entities",
            "GET /schema": "This endpoint",
            "POST /entity/damage": "Apply damage to entity",
            "POST /entity/heal": "Heal entity",
            "POST /game/create": "Create match with mode and config",
            "GET /game/state": "Get match state and scores",
            "POST /trigger/create": "Create trigger zone",
            "POST /projectile/spawn": "Spawn projectile",
            "POST /ai/set": "Set AI behavior on entity"
        }
    }))
}

// ── HUD endpoints ──

/// POST /ui/text — add text to HUD
pub async fn ui_text(
    State(world): State<SharedWorld>,
    Json(req): Json<crate::hud::HudElement>,
) -> Json<MessageResponse> {
    world.with(|w, _| {
        if let Some(canvas) = w.resource_mut::<crate::hud::HudCanvas>() {
            canvas.add(req.clone());
        }
    });
    Json(MessageResponse {
        ok: true,
        message: None,
    })
}

/// POST /ui/bar — add a bar to HUD
pub async fn ui_bar(
    State(world): State<SharedWorld>,
    Json(req): Json<crate::hud::HudElement>,
) -> Json<MessageResponse> {
    world.with(|w, _| {
        if let Some(canvas) = w.resource_mut::<crate::hud::HudCanvas>() {
            canvas.add(req.clone());
        }
    });
    Json(MessageResponse {
        ok: true,
        message: None,
    })
}

/// POST /ui/clear — remove all HUD elements
pub async fn ui_clear(State(world): State<SharedWorld>) -> Json<MessageResponse> {
    world.with(|w, _| {
        if let Some(canvas) = w.resource_mut::<crate::hud::HudCanvas>() {
            canvas.clear();
        }
    });
    Json(MessageResponse {
        ok: true,
        message: Some("HUD cleared".into()),
    })
}

/// GET /ui/list — list current HUD elements
pub async fn ui_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let elements = world.with_world(|w| {
        w.resource::<crate::hud::HudCanvas>()
            .map(|c| {
                c.elements
                    .iter()
                    .map(|e| serde_json::to_value(e).unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });
    Json(serde_json::json!({"elements": elements, "count": elements.len()}))
}

// ── Gameplay endpoints ──

/// POST /entity/damage — apply damage to an entity
pub async fn entity_damage(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let amount = req.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

    world.with(|w, _| {
        if let Some(entity) = find_entity(w, entity_id)
            && let Some(events) = w.resource_mut::<euca_ecs::Events>()
        {
            events.send(euca_gameplay::DamageEvent {
                target: entity,
                amount,
                source: None,
            });
        }
    });

    Json(MessageResponse {
        ok: true,
        message: Some(format!("Applied {amount} damage to entity {entity_id}")),
    })
}

/// POST /entity/heal — heal an entity
pub async fn entity_heal(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let amount = req.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

    let ok = world.with(|w, _| {
        if let Some(entity) = find_entity(w, entity_id) {
            euca_gameplay::health::heal(w, entity, amount);
            true
        } else {
            false
        }
    });

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("Healed entity {entity_id} by {amount}")
        } else {
            format!("Entity {entity_id} not found")
        }),
    })
}

/// POST /game/create — create a match with config
pub async fn game_create(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let mode = req
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("deathmatch")
        .to_string();
    let score_limit = req
        .get("score_limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(10) as i32;
    let time_limit = req
        .get("time_limit")
        .and_then(|v| v.as_f64())
        .unwrap_or(300.0) as f32;
    let respawn_delay = req
        .get("respawn_delay")
        .and_then(|v| v.as_f64())
        .unwrap_or(3.0) as f32;

    world.with(|w, _| {
        let config = euca_gameplay::MatchConfig {
            mode: mode.clone(),
            score_limit,
            time_limit,
            respawn_delay,
        };
        let mut state = euca_gameplay::GameState::new(config);
        state.start();
        w.insert_resource(state);
    });

    Json(MessageResponse {
        ok: true,
        message: Some(format!("Match created: {mode}, score limit {score_limit}")),
    })
}

/// GET /game/state — get match state and scores
pub async fn game_state(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let data = world.with_world(|w| {
        w.resource::<euca_gameplay::GameState>().map(|state| {
            let phase = match &state.phase {
                euca_gameplay::GamePhase::Lobby => "lobby",
                euca_gameplay::GamePhase::Countdown { .. } => "countdown",
                euca_gameplay::GamePhase::Playing => "playing",
                euca_gameplay::GamePhase::PostMatch { .. } => "post_match",
            };
            serde_json::json!({
                "phase": phase,
                "mode": state.config.mode,
                "elapsed": state.elapsed,
                "scores": state.scoreboard().iter()
                    .map(|(idx, score)| serde_json::json!({"entity": idx, "score": score}))
                    .collect::<Vec<_>>(),
            })
        })
    });

    Json(
        data.unwrap_or(serde_json::json!({"error": "No game state. Use POST /game/create first."})),
    )
}

/// POST /trigger/create — create a trigger zone entity
pub async fn trigger_create(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let pos = req
        .get("position")
        .and_then(|v| v.as_array())
        .map(|a| {
            Vec3::new(
                a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            )
        })
        .unwrap_or(Vec3::ZERO);

    let half = req
        .get("zone")
        .and_then(|v| v.as_array())
        .map(|a| {
            Vec3::new(
                a.first().and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            )
        })
        .unwrap_or(Vec3::new(1.0, 1.0, 1.0));

    let action_str = req
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("damage:10");

    let action = if let Some(rest) = action_str.strip_prefix("damage:") {
        let amount = rest.parse::<f32>().unwrap_or(10.0);
        euca_gameplay::TriggerAction::Damage { amount }
    } else if let Some(rest) = action_str.strip_prefix("heal:") {
        let amount = rest.parse::<f32>().unwrap_or(10.0);
        euca_gameplay::TriggerAction::Heal { amount }
    } else {
        euca_gameplay::TriggerAction::Damage { amount: 10.0 }
    };

    let entity_id = world.with(|w, _| {
        let transform = euca_math::Transform::from_translation(pos);
        let entity = w.spawn(LocalTransform(transform));
        w.insert(entity, GlobalTransform::default());
        w.insert(entity, euca_gameplay::TriggerZone::new(half, action));
        entity.index()
    });

    Json(serde_json::json!({
        "ok": true,
        "entity_id": entity_id,
        "message": format!("Trigger zone created at ({}, {}, {})", pos.x, pos.y, pos.z),
    }))
}

/// POST /projectile/spawn — spawn a projectile
pub async fn projectile_spawn(
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

    let direction = req
        .get("direction")
        .and_then(|v| v.as_array())
        .map(|a| {
            Vec3::new(
                a.first().and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            )
        })
        .unwrap_or(Vec3::new(1.0, 0.0, 0.0));

    let speed = req.get("speed").and_then(|v| v.as_f64()).unwrap_or(20.0) as f32;
    let damage = req.get("damage").and_then(|v| v.as_f64()).unwrap_or(25.0) as f32;
    let lifetime = req.get("lifetime").and_then(|v| v.as_f64()).unwrap_or(3.0) as f32;

    let entity_id = world.with(|w, _| {
        // Use a dummy owner (Entity 0) if not specified
        let owner = Entity::from_raw(0, 0);
        let transform = euca_math::Transform::from_translation(from);
        let entity = w.spawn(LocalTransform(transform));
        w.insert(entity, GlobalTransform::default());
        w.insert(
            entity,
            euca_gameplay::Projectile::new(direction, speed, damage, lifetime, owner),
        );
        entity.index()
    });

    Json(serde_json::json!({
        "ok": true,
        "entity_id": entity_id,
    }))
}

/// POST /ai/set — set AI behavior on an entity
pub async fn ai_set(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<MessageResponse> {
    let entity_id = req.get("entity_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let behavior = req
        .get("behavior")
        .and_then(|v| v.as_str())
        .unwrap_or("idle");
    let target_id = req.get("target").and_then(|v| v.as_u64()).map(|v| v as u32);
    let speed = req.get("speed").and_then(|v| v.as_f64()).unwrap_or(3.0) as f32;

    let ok = world.with(|w, _| {
        let entity = match find_entity(w, entity_id) {
            Some(e) => e,
            None => return false,
        };

        let goal = match behavior {
            "idle" => {
                let pos = w
                    .get::<LocalTransform>(entity)
                    .map(|lt| lt.0.translation)
                    .unwrap_or(Vec3::ZERO);
                euca_gameplay::AiGoal::idle(pos)
            }
            "chase" => {
                let target = target_id
                    .and_then(|id| find_entity(w, id))
                    .unwrap_or(Entity::from_raw(0, 0));
                euca_gameplay::AiGoal::chase(target, speed)
            }
            "patrol" => {
                // Parse waypoints from request if provided
                let waypoints = req
                    .get("waypoints")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|wp| {
                                wp.as_array().map(|a| {
                                    Vec3::new(
                                        a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                                        a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                                        a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                                    )
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                euca_gameplay::AiGoal::patrol(waypoints, speed)
            }
            _ => {
                let pos = w
                    .get::<LocalTransform>(entity)
                    .map(|lt| lt.0.translation)
                    .unwrap_or(Vec3::ZERO);
                euca_gameplay::AiGoal::idle(pos)
            }
        };

        // Ensure entity has Velocity for AI movement
        if w.get::<Velocity>(entity).is_none() {
            w.insert(entity, Velocity::default());
        }
        w.insert(entity, goal);
        true
    });

    Json(MessageResponse {
        ok,
        message: Some(if ok {
            format!("Set entity {entity_id} AI to {behavior}")
        } else {
            format!("Entity {entity_id} not found")
        }),
    })
}
