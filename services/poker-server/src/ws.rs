//! WebSocket connection handler.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{Mutex, mpsc};

use crate::msg::{ClientMsg, ServerMsg};
use crate::state::AppState;

/// Handle a single WebSocket connection.
pub async fn handle_socket(socket: WebSocket, state: Arc<Mutex<AppState>>) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Channel for outgoing messages (room broadcasts -> this socket).
    let (msg_tx, mut msg_rx) = mpsc::unbounded_channel::<ServerMsg>();

    // Assign a unique player ID for this connection.
    let player_id = uuid::Uuid::new_v4().to_string();

    // Spawn a task that forwards queued ServerMsgs to the WebSocket.
    let forward_handle = tokio::spawn(async move {
        while let Some(msg) = msg_rx.recv().await {
            match serde_json::to_string(&msg) {
                Ok(text) => {
                    if ws_tx.send(Message::Text(text.into())).await.is_err() {
                        break; // client disconnected
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to serialize ServerMsg: {e}");
                }
            }
        }
    });

    // Track which room this player is currently in.
    let mut current_room: Option<String> = None;

    // Process incoming messages.
    while let Some(Ok(frame)) = ws_rx.next().await {
        let text = match frame {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };

        let client_msg = match serde_json::from_str::<ClientMsg>(&text) {
            Ok(m) => m,
            Err(e) => {
                let _ = msg_tx.send(ServerMsg::Error {
                    message: format!("Invalid message: {e}"),
                });
                continue;
            }
        };

        handle_client_message(client_msg, &state, &msg_tx, &player_id, &mut current_room).await;
    }

    // Player disconnected -- clean up.
    if let Some(room_id) = &current_room {
        let mut app = state.lock().await;
        if let Some(room) = app.rooms.get_room_mut(room_id) {
            room.leave(&player_id);
        }
        app.rooms.cleanup_empty_rooms();
    }

    // Cancel the forward task.
    forward_handle.abort();
}

/// Dispatch a single client message.
async fn handle_client_message(
    msg: ClientMsg,
    state: &Arc<Mutex<AppState>>,
    tx: &mpsc::UnboundedSender<ServerMsg>,
    player_id: &str,
    current_room: &mut Option<String>,
) {
    match msg {
        ClientMsg::JoinRoom {
            room_id,
            player_name,
            buy_in,
        } => {
            let mut app = state.lock().await;

            // If already in a room, leave it first.
            if let Some(old_room) = current_room.take()
                && let Some(room) = app.rooms.get_room_mut(&old_room)
            {
                room.leave(player_id);
            }

            // Create or join room.
            let rid = match room_id {
                Some(id) if app.rooms.get_room(&id).is_some() => id,
                _ => app.rooms.create_room(5, 10), // default blinds
            };

            match app
                .rooms
                .get_room_mut(&rid)
                .expect("room was just created or verified")
                .join(player_id, &player_name, buy_in, tx.clone())
            {
                Ok(seat) => {
                    *current_room = Some(rid.clone());
                    let _ = tx.send(ServerMsg::RoomJoined {
                        room_id: rid.clone(),
                        seat,
                    });
                    // Send current table state.
                    if let Some(room) = app.rooms.get_room(&rid) {
                        room.broadcast_table_state();
                    }
                }
                Err(e) => {
                    let _ = tx.send(ServerMsg::Error { message: e });
                }
            }
        }

        ClientMsg::LeaveRoom => {
            if let Some(room_id) = current_room.take() {
                let mut app = state.lock().await;
                if let Some(room) = app.rooms.get_room_mut(&room_id) {
                    room.leave(player_id);
                    room.broadcast_table_state();
                }
                app.rooms.cleanup_empty_rooms();
            }
        }

        ClientMsg::Action { action, amount } => {
            let Some(room_id) = current_room.as_ref() else {
                let _ = tx.send(ServerMsg::Error {
                    message: "Not in a room".to_string(),
                });
                return;
            };
            let mut app = state.lock().await;
            if let Some(room) = app.rooms.get_room_mut(room_id)
                && let Err(e) = room.handle_action(player_id, &action, amount)
            {
                let _ = tx.send(ServerMsg::Error { message: e });
            }
        }

        ClientMsg::Chat { text } => {
            let Some(room_id) = current_room.as_ref() else {
                return;
            };
            let app = state.lock().await;
            if let Some(room) = app.rooms.get_room(room_id) {
                room.chat(player_id, &text);
            }
        }

        ClientMsg::Ready => {
            let Some(room_id) = current_room.as_ref() else {
                let _ = tx.send(ServerMsg::Error {
                    message: "Not in a room".to_string(),
                });
                return;
            };
            let mut app = state.lock().await;
            if let Some(room) = app.rooms.get_room_mut(room_id) {
                room.set_ready(player_id);
            }
        }
    }
}
