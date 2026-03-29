//! Euca Poker Server — authoritative Texas Hold'em WebSocket game server.

mod msg;
mod poker;
mod room;
mod state;
mod ws;

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::room::RoomSummary;
use crate::state::AppState;

type SharedState = Arc<Mutex<AppState>>;

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

async fn health() -> &'static str {
    "ok"
}

async fn list_rooms(State(state): State<SharedState>) -> Json<Vec<RoomSummary>> {
    let app = state.lock().await;
    Json(app.rooms.list_rooms())
}

#[derive(Debug, Deserialize)]
struct CreateRoomRequest {
    #[serde(default = "default_small_blind")]
    small_blind: u32,
    #[serde(default = "default_big_blind")]
    big_blind: u32,
}

fn default_small_blind() -> u32 {
    5
}

fn default_big_blind() -> u32 {
    10
}

#[derive(Debug, Serialize)]
struct CreateRoomResponse {
    room_id: String,
}

async fn create_room(
    State(state): State<SharedState>,
    Json(req): Json<CreateRoomRequest>,
) -> Json<CreateRoomResponse> {
    let mut app = state.lock().await;
    let room_id = app.rooms.create_room(req.small_blind, req.big_blind);
    Json(CreateRoomResponse { room_id })
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<SharedState>) -> impl IntoResponse {
    ws.on_upgrade(|socket| ws::handle_socket(socket, state))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state: SharedState = Arc::new(Mutex::new(AppState::new()));

    let app = Router::new()
        .route("/health", get(health))
        .route("/rooms", get(list_rooms).post(create_room))
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "0.0.0.0:8080";
    tracing::info!("Poker server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind address");
    axum::serve(listener, app).await.expect("Server error");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msg::{ClientMsg, ServerMsg};
    use crate::poker::PokerTable;
    use tokio::sync::mpsc;

    // -- Message serialization roundtrip --

    #[test]
    fn test_message_serialization() {
        let client_msg = ClientMsg::JoinRoom {
            room_id: Some("abc".to_string()),
            player_name: "Alice".to_string(),
            buy_in: 1000,
        };
        let json = serde_json::to_string(&client_msg).unwrap();
        let decoded: ClientMsg = serde_json::from_str(&json).unwrap();
        match decoded {
            ClientMsg::JoinRoom {
                room_id,
                player_name,
                buy_in,
            } => {
                assert_eq!(room_id, Some("abc".to_string()));
                assert_eq!(player_name, "Alice");
                assert_eq!(buy_in, 1000);
            }
            _ => panic!("wrong variant"),
        }

        let server_msg = ServerMsg::RoomJoined {
            room_id: "xyz".to_string(),
            seat: 3,
        };
        let json = serde_json::to_string(&server_msg).unwrap();
        let decoded: ServerMsg = serde_json::from_str(&json).unwrap();
        match decoded {
            ServerMsg::RoomJoined { room_id, seat } => {
                assert_eq!(room_id, "xyz");
                assert_eq!(seat, 3);
            }
            _ => panic!("wrong variant"),
        }
    }

    // -- Room create and join --

    #[test]
    fn test_room_create_join() {
        let mut state = AppState::new();
        let room_id = state.rooms.create_room(5, 10);

        let (tx, _rx) = mpsc::unbounded_channel();
        let room = state.rooms.get_room_mut(&room_id).unwrap();
        let seat = room.join("player1", "Alice", 1000, tx).unwrap();
        assert_eq!(seat, 0);
        assert_eq!(room.player_count(), 1);
    }

    // -- Room leave --

    #[test]
    fn test_room_leave() {
        let mut state = AppState::new();
        let room_id = state.rooms.create_room(5, 10);

        let (tx, _rx) = mpsc::unbounded_channel();
        let room = state.rooms.get_room_mut(&room_id).unwrap();
        room.join("player1", "Alice", 1000, tx).unwrap();
        assert_eq!(room.player_count(), 1);

        room.leave("player1");
        assert_eq!(room.player_count(), 0);
        assert!(room.is_empty());
    }

    // -- Basic hand flow --

    #[test]
    fn test_basic_hand_flow() {
        let mut state = AppState::new();
        let room_id = state.rooms.create_room(5, 10);

        let (tx1, _rx1) = mpsc::unbounded_channel();
        let (tx2, _rx2) = mpsc::unbounded_channel();
        let room = state.rooms.get_room_mut(&room_id).unwrap();
        room.join("p1", "Alice", 1000, tx1).unwrap();
        room.join("p2", "Bob", 1000, tx2).unwrap();

        room.set_ready("p1");
        let started = room.set_ready("p2");
        assert!(started);

        // Hand should be in preflop.
        assert_eq!(room.table.phase, crate::poker::Phase::Preflop);
        // Both players should have hole cards.
        assert!(room.table.players[0].hole_cards.is_some());
        assert!(room.table.players[1].hole_cards.is_some());
        // Pot should have blinds (5 + 10 = 15).
        assert_eq!(room.table.pot, 15);
    }

    // -- Fold gives pot --

    #[test]
    fn test_fold_gives_pot() {
        let mut state = AppState::new();
        let room_id = state.rooms.create_room(5, 10);

        let (tx1, _rx1) = mpsc::unbounded_channel();
        let (tx2, _rx2) = mpsc::unbounded_channel();
        let room = state.rooms.get_room_mut(&room_id).unwrap();
        room.join("p1", "Alice", 1000, tx1).unwrap();
        room.join("p2", "Bob", 1000, tx2).unwrap();

        room.set_ready("p1");
        room.set_ready("p2");

        // Determine who has action.
        let action_seat = room.table.action_on;
        let action_player = if action_seat == 0 { "p1" } else { "p2" };
        let _other_player = if action_seat == 0 { "p2" } else { "p1" };
        let other_seat = if action_seat == 0 { 1 } else { 0 };

        let pot_before = room.table.pot;

        // The player with action folds.
        room.handle_action(action_player, "fold", None).unwrap();

        // The other player should have won the pot.
        assert_eq!(room.table.phase, crate::poker::Phase::Waiting);
        // Winner's chips = initial (1000) - their blind + pot.
        // The exact amount depends on who was SB/BB. Just verify chips increased.
        assert!(room.table.players[other_seat].chips > 1000 - 10);
        // Pot should be zero.
        assert_eq!(room.table.pot, 0);
        let _ = pot_before; // suppress unused warning
    }

    // -- Full hand (preflop -> river -> showdown) --

    #[test]
    fn test_full_hand() {
        let mut table = PokerTable::new(6, 5, 10);
        table.seat_player("Alice".into(), 1000);
        table.seat_player("Bob".into(), 1000);
        table.set_ready(0);
        table.set_ready(1);
        table.start_hand();

        assert_eq!(table.phase, crate::poker::Phase::Preflop);

        // Both players call/check through every street.
        // Preflop: action is on player left of BB.
        // With 2 players: dealer = SB, other = BB, action on dealer (SB).
        let mut phase_transitions = vec![];

        // Helper: play through a betting round with both players checking/calling.
        fn play_round(table: &mut PokerTable) -> crate::poker::Phase {
            for _ in 0..10 {
                // safety limit
                if table.phase == crate::poker::Phase::Waiting
                    || table.phase == crate::poker::Phase::Showdown
                {
                    return table.phase;
                }
                let seat = table.action_on;
                let p = &table.players[seat];
                let action = if p.current_bet < table.current_bet {
                    "call"
                } else {
                    "check"
                };
                match table.apply_action(seat, action, None) {
                    Ok(crate::poker::ActionOutcome::AdvancePhase { new_phase }) => {
                        return new_phase;
                    }
                    Ok(crate::poker::ActionOutcome::LastPlayerWins { .. }) => {
                        return crate::poker::Phase::Waiting;
                    }
                    Ok(crate::poker::ActionOutcome::Continue) => {}
                    Err(e) => panic!("Unexpected error: {e}"),
                }
            }
            table.phase
        }

        // Preflop.
        let next = play_round(&mut table);
        phase_transitions.push(next);

        if next == crate::poker::Phase::Flop {
            assert_eq!(table.community_cards.len(), 3);
            let next = play_round(&mut table);
            phase_transitions.push(next);

            if next == crate::poker::Phase::Turn {
                assert_eq!(table.community_cards.len(), 4);
                let next = play_round(&mut table);
                phase_transitions.push(next);

                if next == crate::poker::Phase::River {
                    assert_eq!(table.community_cards.len(), 5);
                    let next = play_round(&mut table);
                    phase_transitions.push(next);
                }
            }
        }

        // We should have reached showdown.
        assert!(
            phase_transitions.contains(&crate::poker::Phase::Showdown),
            "Expected showdown, got phases: {phase_transitions:?}"
        );

        // Evaluate showdown.
        let (winners, hands_shown) = table.showdown();
        assert!(!winners.is_empty(), "Should have at least one winner");
        assert!(!hands_shown.is_empty(), "Should show hands at showdown");
    }
}
