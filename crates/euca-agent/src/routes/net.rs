//! Networking status endpoint — show connected peers, tick rate, bandwidth.

use axum::Json;
use axum::extract::State;

use crate::state::SharedWorld;

/// GET /net/status — show networking state
pub async fn net_status(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let result = world.with_world(|w| {
        let mut status = serde_json::Map::new();

        // GameServer info
        if let Some(server) = w.resource::<euca_net::GameServer>() {
            let peer_addrs: Vec<String> = server
                .player_addrs()
                .iter()
                .map(|a| a.to_string())
                .collect();
            status.insert(
                "server".to_string(),
                serde_json::json!({
                    "connected_peers": server.player_count(),
                    "peer_addresses": peer_addrs,
                }),
            );
        }

        // GameClient info
        if let Some(client) = w.resource::<euca_net::GameClient>() {
            status.insert(
                "client".to_string(),
                serde_json::json!({
                    "connected": client.connected,
                    "server_tick": client.server_tick,
                    "player_network_id": client.player_network_id.map(|id| id.0),
                    "replicated_entities": client.entities.len(),
                }),
            );
        }

        // Tick rate config
        if let Some(tick_config) = w.resource::<euca_net::TickRateConfig>() {
            status.insert(
                "tick_rate".to_string(),
                serde_json::json!({
                    "sim_rate": tick_config.sim_rate,
                    "net_rate": tick_config.net_rate,
                    "ticks_per_send": tick_config.ticks_per_send(),
                }),
            );
        }

        // Bandwidth budget
        if let Some(budget) = w.resource::<euca_net::BandwidthBudget>() {
            status.insert(
                "bandwidth".to_string(),
                serde_json::json!({
                    "bytes_per_tick": budget.bytes_per_tick,
                    "bytes_used": budget.bytes_used,
                    "remaining": budget.remaining(),
                }),
            );
        }

        if status.is_empty() {
            serde_json::json!({
                "active": false,
                "message": "No networking resources found in world",
            })
        } else {
            status.insert("active".to_string(), serde_json::json!(true));
            serde_json::Value::Object(status)
        }
    });
    Json(result)
}
