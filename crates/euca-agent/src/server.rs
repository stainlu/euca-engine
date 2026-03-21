use crate::routes;
use crate::state::SharedWorld;
use axum::{
    Router,
    routing::{get, post},
};
use euca_ecs::{Schedule, World};

/// HTTP server that exposes the ECS world to external AI agents.
pub struct AgentServer {
    shared: SharedWorld,
    port: u16,
}

impl AgentServer {
    /// Create a new agent server wrapping the given world and schedule.
    pub fn new(world: World, schedule: Schedule, port: u16) -> Self {
        Self {
            shared: SharedWorld::new(world, schedule),
            port,
        }
    }

    /// Create a server backed by an existing SharedWorld.
    ///
    /// Use this when the editor already owns the SharedWorld and wants the
    /// HTTP server to operate on the same world instance.
    pub fn from_shared(shared: SharedWorld, port: u16) -> Self {
        Self { shared, port }
    }

    /// Get a clone of the shared world handle (for external use).
    pub fn shared_world(&self) -> SharedWorld {
        self.shared.clone()
    }

    /// Build the axum router.
    fn router(&self) -> Router {
        Router::new()
            .route("/", get(routes::status))
            .route("/observe", post(routes::observe))
            .route("/entities/{id}", get(routes::get_entity))
            .route("/entities/{id}/components", post(routes::patch_entity))
            .route("/step", post(routes::step))
            .route("/spawn", post(routes::spawn))
            .route("/despawn", post(routes::despawn))
            .route("/reset", post(routes::reset))
            .route("/play", post(routes::play))
            .route("/pause", post(routes::pause))
            .route("/screenshot", post(routes::screenshot))
            .route("/camera", get(routes::camera_get))
            .route("/camera", post(routes::camera_set))
            .route("/camera/view", post(routes::camera_view))
            .route("/camera/focus", post(routes::camera_focus))
            .route("/scene/save", post(routes::scene_save))
            .route("/scene/load", post(routes::scene_load))
            .route("/ui/text", post(routes::ui_text))
            .route("/ui/bar", post(routes::ui_bar))
            .route("/ui/clear", post(routes::ui_clear))
            .route("/ui/list", get(routes::ui_list))
            .route("/entity/damage", post(routes::entity_damage))
            .route("/entity/heal", post(routes::entity_heal))
            .route("/game/create", post(routes::game_create))
            .route("/game/state", get(routes::game_state))
            .route("/trigger/create", post(routes::trigger_create))
            .route("/projectile/spawn", post(routes::projectile_spawn))
            .route("/ai/set", post(routes::ai_set))
            .route("/rule/create", post(routes::rule_create))
            .route("/rule/list", get(routes::rule_list))
            .route("/template/create", post(routes::template_create))
            .route("/template/spawn", post(routes::template_spawn))
            .route("/template/list", get(routes::template_list))
            .route("/auth/login", post(routes::auth_login))
            .route("/auth/status", get(routes::auth_status))
            .route("/schema", get(routes::schema))
            .route("/audio/play", post(routes::audio_play))
            .route("/audio/stop", post(routes::audio_stop))
            .route("/audio/list", get(routes::audio_list))
            .route("/animation/load", post(routes::animation_load))
            .route("/animation/play", post(routes::animation_play))
            .route("/animation/stop", post(routes::animation_stop))
            .route("/animation/list", get(routes::animation_list))
            .route(
                "/animation/state-machine",
                post(routes::animation_state_machine),
            )
            .route("/animation/montage", post(routes::animation_montage))
            .route("/terrain/create", post(routes::terrain_create))
            .route("/terrain/edit", post(routes::terrain_edit))
            .route("/prefab/spawn", post(routes::prefab_spawn))
            .route("/prefab/list", get(routes::prefab_list))
            .route("/material/set", post(routes::material_set))
            .route("/postprocess/settings", get(routes::postprocess_get))
            .route("/postprocess/settings", post(routes::postprocess_set))
            .route("/fog/settings", get(routes::fog_get))
            .route("/fog/settings", post(routes::fog_set))
            .route("/diagnose", get(routes::diagnose))
            .route("/events", get(routes::events_list))
            .route("/profile", get(routes::profile))
            .route("/ability/use", post(routes::ability_use))
            .route("/ability/list/{id}", get(routes::ability_list))
            .route("/particle/create", post(routes::particle_create))
            .route("/particle/stop", post(routes::particle_stop))
            .route("/particle/list", get(routes::particle_list))
            .route("/navmesh/generate", post(routes::navmesh_generate))
            .route("/path/compute", post(routes::path_compute))
            .route("/path/set", post(routes::path_set))
            .route("/input/bind", post(routes::input_bind))
            .route("/input/unbind", post(routes::input_unbind))
            .route("/input/list", get(routes::input_list))
            .route("/input/context/push", post(routes::input_context_push))
            .route("/input/context/pop", post(routes::input_context_pop))
            .route("/foliage/scatter", post(routes::foliage_scatter))
            .route("/foliage/list", get(routes::foliage_list))
            .route("/level/load", post(routes::level_load))
            .route("/level/save", post(routes::level_save))
            .with_state(self.shared.clone())
    }

    /// Run the server (blocking). Call from a tokio runtime.
    pub async fn run(self) {
        let addr = format!("127.0.0.1:{}", self.port);
        let router = self.router();

        log::info!("Euca Agent Server listening on http://{addr}");
        log::info!(
            "Endpoints: GET /, POST /observe, /step, /spawn, /despawn, /reset, GET /schema, /entities/:id"
        );

        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .expect("Failed to bind server address");

        axum::serve(listener, router).await.expect("Server error");
    }
}
