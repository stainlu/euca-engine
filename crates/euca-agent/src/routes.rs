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
    pub cube_mesh: MeshHandle,
    pub sphere_mesh: MeshHandle,
    pub default_material: MaterialHandle,
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

    RichEntityData {
        id: entity.index(),
        generation: entity.generation(),
        transform,
        velocity,
        collider,
        physics_body,
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
    /// Mesh to attach: "cube" or "sphere"
    #[serde(default)]
    pub mesh: Option<String>,
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

        // Assign mesh + default material from pre-uploaded assets
        if let Some(mesh_name) = &req.mesh
            && let Some(assets) = w.resource::<DefaultAssets>()
        {
            let mesh_handle = match mesh_name.as_str() {
                "cube" => Some(assets.cube_mesh),
                "sphere" => Some(assets.sphere_mesh),
                _ => None,
            };
            let mat = assets.default_material;
            if let Some(mesh) = mesh_handle {
                w.insert(entity, MeshRenderer { mesh });
                w.insert(entity, MaterialRef { handle: mat });
            }
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
            "GET /schema": "This endpoint"
        }
    }))
}
