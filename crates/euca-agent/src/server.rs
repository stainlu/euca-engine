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

    /// Get a clone of the shared world handle (for external use).
    pub fn shared_world(&self) -> SharedWorld {
        self.shared.clone()
    }

    /// Build the axum router.
    fn router(&self) -> Router {
        Router::new()
            .route("/", get(routes::status))
            .route("/observe", post(routes::observe))
            .route("/step", post(routes::step))
            .route("/spawn", post(routes::spawn))
            .route("/despawn", post(routes::despawn))
            .route("/reset", post(routes::reset))
            .route("/schema", get(routes::schema))
            .with_state(self.shared.clone())
    }

    /// Run the server (blocking). Call from a tokio runtime.
    pub async fn run(self) {
        let addr = format!("127.0.0.1:{}", self.port);
        let router = self.router();

        log::info!("Euca Agent Server listening on http://{addr}");
        log::info!("Endpoints: GET /, POST /observe, /step, /spawn, /despawn, /reset, GET /schema");

        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .expect("Failed to bind server address");

        axum::serve(listener, router).await.expect("Server error");
    }
}
