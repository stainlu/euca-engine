//! Standalone matchmaking server for Euca engine games.
//!
//! ```text
//! GET  /health  — health check
//! GET  /stats   — queue sizes, active matches
//! WS   /ws      — WebSocket for matchmaking protocol
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use euca_matchmaking::{
    ClientMessage, MatchAcceptResult, MatchConfig, MatchDeclineResult, Matchmaker, ServerMessage,
};
use serde::Serialize;
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

/// Shared server state behind an `Arc<Mutex<..>>`.
struct AppState {
    matchmaker: Matchmaker,
    /// Map from player_id → sender half so the tick loop can push messages.
    senders: HashMap<String, tokio::sync::mpsc::UnboundedSender<ServerMessage>>,
}

type SharedState = Arc<Mutex<AppState>>;

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

async fn health() -> &'static str {
    "ok"
}

#[derive(Serialize)]
struct Stats {
    total_queued: usize,
    pending_matches: usize,
}

async fn stats(State(state): State<SharedState>) -> Json<Stats> {
    let st = state.lock().await;
    Json(Stats {
        total_queued: st.matchmaker.total_queued(),
        pending_matches: st.matchmaker.pending_match_count(),
    })
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<SharedState>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

// ---------------------------------------------------------------------------
// WebSocket handler
// ---------------------------------------------------------------------------

async fn handle_socket(socket: WebSocket, state: SharedState) {
    use futures_util::{SinkExt, StreamExt};

    let (mut ws_tx, mut ws_rx) = socket.split();

    // Per-connection channel for server-initiated pushes.
    let (push_tx, mut push_rx) = tokio::sync::mpsc::unbounded_channel::<ServerMessage>();

    // Forward push messages to the WebSocket.
    let push_task = tokio::spawn(async move {
        while let Some(msg) = push_rx.recv().await {
            if let Ok(json) = serde_json::to_string(&msg)
                && ws_tx.send(Message::Text(json.into())).await.is_err()
            {
                break;
            }
        }
    });

    // Track the player_id so we can clean up on disconnect.
    let mut current_player_id: Option<String> = None;

    while let Some(Ok(msg)) = ws_rx.next().await {
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };

        let client_msg: ClientMessage = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                let _ = push_tx.send(ServerMessage::Error {
                    message: format!("invalid message: {e}"),
                });
                continue;
            }
        };

        match client_msg {
            ClientMessage::Ping => {
                let _ = push_tx.send(ServerMessage::Pong);
            }
            ClientMessage::JoinQueue {
                player_id,
                display_name,
                mmr,
                game_mode,
            } => {
                let mut st = state.lock().await;

                let player = euca_matchmaking::QueuedPlayer {
                    player_id: player_id.clone(),
                    display_name,
                    mmr,
                    queued_at: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    game_mode,
                };

                let position = st.matchmaker.add_player(player);

                // Register the sender so tick loop can push to this player.
                st.senders.insert(player_id.clone(), push_tx.clone());
                current_player_id = Some(player_id);

                let _ = push_tx.send(ServerMessage::QueueStatus {
                    position,
                    estimated_wait_secs: 0,
                });
            }
            ClientMessage::LeaveQueue { player_id } => {
                let mut st = state.lock().await;
                st.matchmaker.remove_player(&player_id);
                st.senders.remove(&player_id);
                current_player_id = None;
            }
            ClientMessage::AcceptMatch {
                player_id,
                match_id,
            } => {
                let mut st = state.lock().await;
                match st.matchmaker.accept_match(&player_id, &match_id) {
                    MatchAcceptResult::AllAccepted => {
                        let confirmed = ServerMessage::MatchConfirmed {
                            match_id,
                            server_address: "127.0.0.1:7777".to_string(),
                        };
                        let _ = push_tx.send(confirmed);
                    }
                    MatchAcceptResult::Waiting { remaining } => {
                        let _ = push_tx.send(ServerMessage::QueueStatus {
                            position: 0,
                            estimated_wait_secs: remaining,
                        });
                    }
                    MatchAcceptResult::Expired => {
                        let _ = push_tx.send(ServerMessage::MatchCancelled {
                            match_id,
                            reason: "acceptance timed out".to_string(),
                        });
                    }
                    MatchAcceptResult::NotFound => {
                        let _ = push_tx.send(ServerMessage::Error {
                            message: "match not found".to_string(),
                        });
                    }
                }
            }
            ClientMessage::DeclineMatch {
                player_id,
                match_id,
            } => {
                let mut st = state.lock().await;
                match st.matchmaker.decline_match(&player_id, &match_id) {
                    MatchDeclineResult::Cancelled { player_ids } => {
                        // Notify other players in the match.
                        for pid in &player_ids {
                            if let Some(tx) = st.senders.get(pid) {
                                let _ = tx.send(ServerMessage::MatchCancelled {
                                    match_id: match_id.clone(),
                                    reason: "a player declined".to_string(),
                                });
                            }
                        }
                        // Notify the declining player too.
                        let _ = push_tx.send(ServerMessage::MatchCancelled {
                            match_id,
                            reason: "a player declined".to_string(),
                        });
                    }
                    MatchDeclineResult::NotFound => {
                        let _ = push_tx.send(ServerMessage::Error {
                            message: "match not found".to_string(),
                        });
                    }
                }
            }
        }
    }

    // Clean up on disconnect.
    if let Some(pid) = current_player_id {
        let mut st = state.lock().await;
        st.matchmaker.remove_player(&pid);
        st.senders.remove(&pid);
    }

    push_task.abort();
}

// ---------------------------------------------------------------------------
// Tick loop
// ---------------------------------------------------------------------------

/// Background task that periodically runs the matchmaker tick and pushes
/// `MatchFound` / `MatchCancelled` to connected players.
async fn tick_loop(state: SharedState) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    loop {
        interval.tick().await;

        let mut st = state.lock().await;
        let (formed, expired) = st.matchmaker.tick();

        // Notify players of newly formed matches.
        for m in &formed {
            for team in &m.teams {
                for player in &team.players {
                    if let Some(tx) = st.senders.get(&player.player_id) {
                        let _ = tx.send(ServerMessage::MatchFound {
                            match_id: m.match_id.clone(),
                            teams: m.teams.clone(),
                        });
                    }
                }
            }
        }

        // Expired matches: players will receive `Expired` if they try to
        // accept. The tick already cleaned up the pending match state.
        let _ = &expired;
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let mut matchmaker = Matchmaker::new();

    // Register some sensible default game modes.
    matchmaker.configure_mode(
        "1v1",
        MatchConfig {
            team_size: 1,
            team_count: 2,
            max_mmr_gap: 150,
            mmr_gap_growth_per_sec: 10,
            accept_timeout_secs: 30,
        },
    );
    matchmaker.configure_mode(
        "5v5",
        MatchConfig {
            team_size: 5,
            team_count: 2,
            max_mmr_gap: 200,
            mmr_gap_growth_per_sec: 5,
            accept_timeout_secs: 45,
        },
    );
    matchmaker.configure_mode("casual", MatchConfig::default());

    let state: SharedState = Arc::new(Mutex::new(AppState {
        matchmaker,
        senders: HashMap::new(),
    }));

    // Spawn the background tick loop.
    tokio::spawn(tick_loop(state.clone()));

    let app = Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/ws", get(ws_handler))
        .with_state(state);

    let addr = "0.0.0.0:3030";
    tracing::info!("Euca Matchmaking Server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind address");

    axum::serve(listener, app).await.expect("server error");
}
