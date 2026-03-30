mod animation;
mod assertions;
mod asset_gen;
mod audio;
mod camera;
mod debug;
mod entity;
mod fog;
mod foliage;
mod gameplay;
mod gpu;
mod hero;
mod input;
mod inventory;
pub mod level;
mod manifest;
mod material;
mod nav;
mod net;
mod particle;
mod postprocess;
mod prefab;
mod probe;
mod profile;
mod scene_auth;
mod script;
mod shop;
mod sim;
mod snapshot;
mod status_effects;
mod templates;
mod terrain;
mod ui;

pub use animation::{
    animation_list, animation_load, animation_montage, animation_play, animation_state_machine,
    animation_stop,
};
pub use assertions::{assert_create, assert_delete, assert_evaluate, assert_list, assert_results};
pub use asset_gen::{asset_generate, asset_generated, asset_providers, asset_status};
pub use audio::{audio_list, audio_play, audio_stop};
pub use camera::{camera_focus, camera_get, camera_set, camera_view};
pub use debug::{diagnose, events_list};
pub use entity::{
    despawn, entity_damage, entity_heal, get_entity, observe, patch_entity, reset, schema, spawn,
    status, tag_set, view_filter_set,
};
pub use fog::{fog_get, fog_set};
pub use foliage::{foliage_list, foliage_scatter};
pub use gameplay::{
    ability_list, ability_use, ai_set, game_create, game_state, projectile_spawn, rule_create,
    rule_list, trigger_create,
};
pub use gpu::engine_gpu;
pub use hero::{hero_define, hero_list, hero_select};
pub use input::{input_bind, input_context_pop, input_context_push, input_list, input_unbind};
pub use inventory::{item_define, item_equip, item_give, item_list};
pub use manifest::{manifest_feature_update, manifest_get, manifest_set};
pub use material::material_set;
pub use nav::{navmesh_generate, path_compute, path_set};
pub use net::net_status;
pub use particle::{particle_create, particle_list, particle_stop};
pub use postprocess::{postprocess_get, postprocess_preset, postprocess_set};
pub use prefab::{prefab_list, prefab_spawn};
pub use probe::probe;
pub use profile::profile;
pub use scene_auth::{auth_login, auth_status, scene_load, scene_save, screenshot};
pub use script::{script_list, script_load};
pub use shop::{shop_buy, shop_list, shop_sell};
pub use sim::{pause, play, step};
pub use snapshot::{game_summary, snapshot_create, snapshot_diff, snapshot_latest, snapshot_list};
pub use status_effects::{effect_apply, effect_cleanse, effect_list};
pub use templates::{template_create, template_list, template_spawn};
pub use terrain::{terrain_create, terrain_edit};
pub use ui::{ui_bar, ui_clear, ui_list, ui_text};

use serde::{Deserialize, Serialize};

use euca_ecs::Entity;
use euca_math::Vec3;
use euca_physics::{Collider, ColliderShape, PhysicsBody, RigidBodyType, Velocity};
use euca_render::{MaterialHandle, Mesh, MeshHandle};
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
    /// Returns white material as default if parsing fails.
    pub fn material(&self, color: &str) -> Option<MaterialHandle> {
        // Try exact name match first
        if let Some(h) = self.materials.get(color) {
            return Some(*h);
        }
        // Try RGB parsing → nearest preset
        if let Some((r, g, b)) = Self::parse_rgb(color) {
            return Some(self.nearest_color(r, g, b));
        }
        // Fall back to white for unrecognised color strings
        self.materials.get("white").copied()
    }

    /// Parse an RGB color string of the form "r,g,b" where each component is an f32.
    /// Returns `None` if the string does not contain exactly 3 valid f32 components.
    /// Components are clamped to the 0.0..=1.0 range.
    fn parse_rgb(color: &str) -> Option<(f32, f32, f32)> {
        let parts: Vec<&str> = color.split(',').collect();
        if parts.len() != 3 {
            return None;
        }
        let r: f32 = parts[0].trim().parse().ok()?;
        let g: f32 = parts[1].trim().parse().ok()?;
        let b: f32 = parts[2].trim().parse().ok()?;
        Some((r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)))
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

/// GPU information exposed to the agent HTTP layer.
///
/// Populated once during engine init (after `GpuContext` is created) and
/// inserted as an ECS world resource. The `/engine/gpu` endpoint reads this.
#[derive(Clone, Debug, Serialize)]
pub struct GpuInfo {
    /// Backend identifier (e.g. "wgpu", "metal-native").
    pub backend: String,
    /// Human-readable device name (e.g. "Apple M4 Pro").
    pub device_name: String,
    /// Hardware capabilities queried at startup.
    pub capabilities: GpuCapabilities,
}

/// Subset of GPU capabilities relevant for the agent API.
#[derive(Clone, Debug, Serialize)]
pub struct GpuCapabilities {
    pub unified_memory: bool,
    pub multi_draw_indirect: bool,
    pub multi_draw_indirect_count: bool,
    pub texture_binding_array: bool,
    pub non_uniform_indexing: bool,
    pub max_texture_dimension_2d: u32,
    pub max_bind_groups: u32,
    pub max_bindings_per_bind_group: u32,
    pub max_binding_array_elements: u32,
}

impl GpuInfo {
    /// Build from the render crate's types after GPU initialization.
    pub fn from_render(
        backend: &euca_render::RenderBackend,
        adapter_info: &euca_render::AdapterInfo,
        caps: &euca_render::euca_rhi::Capabilities,
    ) -> Self {
        // RenderBackend::MetalNative only exists when euca-render's "metal-native"
        // feature is enabled; the catch-all uses Debug to name future variants.
        #[allow(unreachable_patterns)]
        let backend = match backend {
            euca_render::RenderBackend::Wgpu => "wgpu".to_string(),
            other => format!("{other:?}").to_lowercase(),
        };
        Self {
            backend,
            device_name: adapter_info.name.clone(),
            capabilities: GpuCapabilities {
                unified_memory: caps.unified_memory,
                multi_draw_indirect: caps.multi_draw_indirect,
                multi_draw_indirect_count: caps.multi_draw_indirect_count,
                texture_binding_array: caps.texture_binding_array,
                non_uniform_indexing: caps.non_uniform_indexing,
                max_texture_dimension_2d: caps.max_texture_dimension_2d,
                max_bind_groups: caps.max_bind_groups,
                max_bindings_per_bind_group: caps.max_bindings_per_bind_group,
                max_binding_array_elements: caps.max_binding_array_elements,
            },
        }
    }
}

/// Cache of GPU-uploaded meshes loaded from file paths (GLB/glTF).
///
/// Stored as a World resource. The spawn handler inserts entries after the
/// render loop uploads them; subsequent spawns with the same path reuse the
/// cached handle.
#[derive(Clone, Default)]
pub struct MeshCache {
    pub meshes: std::collections::HashMap<String, MeshHandle>,
}

/// Queue of meshes loaded from disk that need GPU upload.
///
/// The spawn handler pushes entries here (on the HTTP thread) and the
/// render loop drains them (on the main thread where GPU access is available).
/// In headless mode this queue is never drained — entities simply have no
/// `MeshRenderer`, which is harmless.
#[derive(Default)]
pub struct PendingMeshUpload {
    pub queue: Vec<PendingMeshEntry>,
}

/// A single mesh waiting for GPU upload, with optional material/textures from GLB.
pub struct PendingMeshEntry {
    /// The entity that should receive a `MeshRenderer` after upload.
    pub entity: Entity,
    /// The file path (used as cache key in `MeshCache`).
    pub path: String,
    /// CPU-side mesh geometry, ready for upload.
    pub mesh: Mesh,
    /// PBR material from the GLB file (if available).
    pub material: Option<euca_render::Material>,
    /// Texture images from the GLB (RGBA8 pixels). Indices correspond to
    /// the `albedo_tex_index` etc. fields from `GltfMesh`.
    pub images: Vec<euca_asset::gltf_loader::GltfImage>,
    /// Index into `images` for the albedo texture.
    pub albedo_tex_index: Option<usize>,
    /// Vertical offset to place the mesh bottom on the ground plane.
    /// Computed from the mesh's AABB when loaded from a glTF file.
    pub ground_offset: Option<f32>,
}

/// Returns true if the mesh name looks like a file path rather than a
/// primitive name (cube, sphere, etc.).
pub fn is_file_path_mesh(name: &str) -> bool {
    name.contains('/') || name.contains('\\') || name.ends_with(".glb") || name.ends_with(".gltf")
}

/// Result of resolving a mesh name in the spawn handler.
pub(crate) enum MeshResolution {
    /// Already uploaded — use this handle immediately.
    Ready(MeshHandle),
    /// Queued for GPU upload — entity will receive MeshRenderer later.
    Pending,
    /// Not a file path and not in DefaultAssets — no mesh.
    NotFound,
    /// File path but loading failed.
    LoadError(String),
}

/// Resolve a mesh name to a handle, loading from disk if it's a file path.
///
/// - Primitives ("cube", "sphere", etc.) are resolved via `DefaultAssets`.
/// - File paths are checked in `MeshCache`; if not cached, the GLB is loaded
///   from disk and pushed to `PendingMeshUpload` for deferred GPU upload.
pub(crate) fn resolve_mesh(
    w: &mut euca_ecs::World,
    entity: Entity,
    mesh_name: &str,
) -> MeshResolution {
    // 1. Try DefaultAssets lookup (primitives).
    if let Some(assets) = w.resource::<DefaultAssets>().cloned()
        && let Some(handle) = assets.mesh(mesh_name)
    {
        return MeshResolution::Ready(handle);
    }

    // 2. Only treat as file path if it looks like one.
    if !is_file_path_mesh(mesh_name) {
        return MeshResolution::NotFound;
    }

    // 3. Check mesh cache for previously uploaded file.
    if let Some(cache) = w.resource::<MeshCache>().cloned()
        && let Some(handle) = cache.meshes.get(mesh_name)
    {
        return MeshResolution::Ready(*handle);
    }

    // 4. Verify the file exists before attempting to load.
    if !std::path::Path::new(mesh_name).exists() {
        return MeshResolution::LoadError(format!("File not found: {mesh_name}"));
    }

    // 5. Load the GLB/glTF from disk (CPU-only, no GPU access needed).
    let scene = match euca_asset::load_gltf(mesh_name) {
        Ok(s) => s,
        Err(e) => return MeshResolution::LoadError(e),
    };

    // Take the first mesh from the scene.
    let gltf_mesh = match scene.meshes.into_iter().next() {
        Some(m) => m,
        None => return MeshResolution::LoadError("GLB file contains no meshes".into()),
    };

    // 6. Ensure PendingMeshUpload resource exists.
    if w.resource::<PendingMeshUpload>().is_none() {
        w.insert_resource(PendingMeshUpload::default());
    }

    // 7. Push to pending queue for GPU upload in the render loop.
    let ground_offset = gltf_mesh.bounds.map(|b| b.ground_offset());
    if let Some(pending) = w.resource_mut::<PendingMeshUpload>() {
        pending.queue.push(PendingMeshEntry {
            entity,
            path: mesh_name.to_string(),
            mesh: gltf_mesh.mesh,
            material: Some(gltf_mesh.material),
            images: scene.images,
            albedo_tex_index: gltf_mesh.albedo_tex_index,
            ground_offset,
        });
    }

    MeshResolution::Pending
}

/// Drain the pending mesh upload queue, uploading each mesh to the GPU
/// and attaching `MeshRenderer` to the corresponding entity.
///
/// Call this from the render loop where `Renderer` and `GpuContext` are
/// available. Each uploaded mesh is cached in `MeshCache` so that
/// subsequent spawns with the same path skip the upload.
pub fn drain_pending_mesh_uploads(
    w: &mut euca_ecs::World,
    renderer: &mut euca_render::Renderer,
    gpu: &euca_render::GpuContext,
) {
    // Drain the queue: take all entries, leaving the resource empty.
    let entries: Vec<PendingMeshEntry> = match w.resource_mut::<PendingMeshUpload>() {
        Some(pending) => std::mem::take(&mut pending.queue),
        None => return,
    };

    if entries.is_empty() {
        return;
    }

    // Ensure MeshCache resource exists.
    if w.resource::<MeshCache>().is_none() {
        w.insert_resource(MeshCache::default());
    }

    for entry in entries {
        // Check if another entry in this batch already uploaded the same path.
        let cached = w
            .resource::<MeshCache>()
            .and_then(|c| c.meshes.get(&entry.path).copied());

        let handle = if let Some(h) = cached {
            h
        } else {
            let h = renderer.upload_mesh(gpu, &entry.mesh);
            if let Some(cache) = w.resource_mut::<MeshCache>() {
                cache.meshes.insert(entry.path.clone(), h);
            }
            h
        };

        // Only attach MeshRenderer + MaterialRef if the entity is still alive.
        if w.is_alive(entry.entity) {
            w.insert(entry.entity, euca_render::MeshRenderer { mesh: handle });

            // Attach GroundOffset so the mesh bottom sits on the ground plane.
            if let Some(offset) = entry.ground_offset {
                w.insert(entry.entity, euca_render::GroundOffset(offset));
            }

            // Upload GLB material with textures if available, otherwise use default.
            let mat_handle = if let Some(ref mat) = entry.material {
                // Upload albedo texture from GLB if present.
                let mut uploaded_mat = mat.clone();
                if let Some(tex_idx) = entry.albedo_tex_index
                    && let Some(img) = entry.images.get(tex_idx)
                {
                    let tex_handle =
                        renderer.upload_texture(gpu, img.width, img.height, &img.pixels);
                    uploaded_mat.albedo_texture = Some(tex_handle);
                }
                Some(renderer.upload_material(gpu, &uploaded_mat))
            } else {
                None
            };

            if let Some(mh) = mat_handle {
                w.insert(entry.entity, euca_render::MaterialRef { handle: mh });
            } else if w.get::<euca_render::MaterialRef>(entry.entity).is_none()
                && let Some(assets) = w.resource::<DefaultAssets>()
            {
                w.insert(
                    entry.entity,
                    euca_render::MaterialRef {
                        handle: assets.default_material,
                    },
                );
            }
        }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gold: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mana: Option<[f32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visible_to: Option<Vec<u32>>,
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

    let gold = w.get::<euca_gameplay::Gold>(entity).map(|g| g.0);
    let level = w.get::<euca_gameplay::Level>(entity).map(|l| l.level);
    let role = w
        .get::<euca_gameplay::EntityRole>(entity)
        .map(|r| format!("{:?}", r));
    let mana = w
        .get::<euca_gameplay::Mana>(entity)
        .map(|m| [m.current, m.max]);

    let tags = w.get::<euca_gameplay::Tags>(entity).map(|t| {
        let mut v: Vec<String> = t.0.iter().cloned().collect();
        v.sort();
        v
    });
    let visible_to = w.get::<euca_gameplay::VisibleTo>(entity).map(|vt| {
        let mut v: Vec<u32> = vt.0.iter().map(|e| e.index()).collect();
        v.sort();
        v
    });

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
        gold,
        level,
        role,
        mana,
        tags,
        visible_to,
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
    #[serde(default)]
    pub combat_damage: Option<f32>,
    #[serde(default)]
    pub combat_range: Option<f32>,
    #[serde(default)]
    pub combat_speed: Option<f32>,
    #[serde(default)]
    pub combat_cooldown: Option<f32>,
    /// "melee" (default) or "stationary" (towers)
    #[serde(default)]
    pub combat_style: Option<String>,
    /// AI patrol waypoints as colon-separated "x,y,z:x,y,z"
    #[serde(default)]
    pub ai_patrol: Option<Vec<[f32; 3]>>,
    /// Starting gold
    #[serde(default)]
    pub gold: Option<i32>,
    /// Gold bounty awarded to killer
    #[serde(default)]
    pub gold_bounty: Option<i32>,
    /// XP bounty awarded to killer
    #[serde(default)]
    pub xp_bounty: Option<u32>,
    /// Entity role: hero, minion, tower, structure
    #[serde(default)]
    pub role: Option<String>,
    /// Spawn point for team (marks this entity as a respawn location)
    #[serde(default)]
    pub spawn_point: Option<u8>,
    /// Mark this entity as the player-controlled hero
    #[serde(default)]
    pub player: Option<bool>,
    /// Building type (e.g. "tier1_tower", "melee_barracks", "ancient").
    /// When set, attaches BuildingStats, BackdoorProtection, and TowerAggro.
    #[serde(default)]
    pub building_type: Option<String>,
    /// Lane assignment for buildings: "top", "mid", or "bot".
    #[serde(default)]
    pub lane: Option<String>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use euca_ecs::World;

    #[test]
    fn is_file_path_mesh_detects_paths() {
        assert!(is_file_path_mesh("assets/generated/sword.glb"));
        assert!(is_file_path_mesh("models/hero.gltf"));
        assert!(is_file_path_mesh("path/to/mesh"));
        assert!(is_file_path_mesh("model.glb"));
        assert!(!is_file_path_mesh("cube"));
        assert!(!is_file_path_mesh("sphere"));
        assert!(!is_file_path_mesh("plane"));
    }

    #[test]
    fn resolve_mesh_primitive_with_default_assets() {
        let mut world = World::new();
        let mut meshes = std::collections::HashMap::new();
        meshes.insert("cube".to_string(), MeshHandle(0));
        let materials = std::collections::HashMap::new();
        world.insert_resource(DefaultAssets {
            meshes,
            materials,
            default_material: euca_render::MaterialHandle(0),
        });

        let entity = world.spawn(());
        match resolve_mesh(&mut world, entity, "cube") {
            MeshResolution::Ready(h) => assert_eq!(h, MeshHandle(0)),
            _ => panic!("Expected Ready for primitive mesh"),
        }
    }

    #[test]
    fn resolve_mesh_unknown_primitive_returns_not_found() {
        let mut world = World::new();
        let entity = world.spawn(());
        match resolve_mesh(&mut world, entity, "nonexistent_primitive") {
            MeshResolution::NotFound => {}
            _ => panic!("Expected NotFound for unknown non-path name"),
        }
    }

    #[test]
    fn resolve_mesh_nonexistent_file_returns_load_error() {
        let mut world = World::new();
        let entity = world.spawn(());
        match resolve_mesh(&mut world, entity, "assets/does_not_exist.glb") {
            MeshResolution::LoadError(msg) => {
                assert!(
                    msg.contains("not found") || msg.contains("File not found"),
                    "Unexpected error: {msg}"
                );
            }
            _ => panic!("Expected LoadError for nonexistent file path"),
        }
    }

    #[test]
    fn resolve_mesh_cached_returns_ready() {
        let mut world = World::new();
        let mut cache = MeshCache::default();
        cache
            .meshes
            .insert("assets/cached.glb".to_string(), MeshHandle(42));
        world.insert_resource(cache);

        let entity = world.spawn(());
        match resolve_mesh(&mut world, entity, "assets/cached.glb") {
            MeshResolution::Ready(h) => assert_eq!(h, MeshHandle(42)),
            _ => panic!("Expected Ready for cached mesh"),
        }
    }

    #[test]
    fn mesh_cache_default_is_empty() {
        let cache = MeshCache::default();
        assert!(cache.meshes.is_empty());
    }

    #[test]
    fn pending_mesh_upload_default_is_empty() {
        let pending = PendingMeshUpload::default();
        assert!(pending.queue.is_empty());
    }
}
