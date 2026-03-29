use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Client -> Server
// ---------------------------------------------------------------------------

/// Messages sent from a connected client to the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMsg {
    /// Join or create a room.
    JoinRoom {
        room_id: Option<String>,
        player_name: String,
        buy_in: u32,
    },
    /// Leave current room.
    LeaveRoom,
    /// Poker action (fold, check, call, raise).
    Action { action: String, amount: Option<u32> },
    /// Chat message.
    Chat { text: String },
    /// Ready to start next hand.
    Ready,
}

// ---------------------------------------------------------------------------
// Server -> Client
// ---------------------------------------------------------------------------

/// Messages sent from the server to connected clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMsg {
    /// Room joined successfully.
    RoomJoined { room_id: String, seat: usize },
    /// Full table state update (sent on join and each phase change).
    TableState {
        phase: String,
        community_cards: Vec<String>,
        pot: u32,
        players: Vec<PlayerView>,
        your_cards: Option<[String; 2]>,
        action_on: Option<usize>,
        current_bet: u32,
    },
    /// A player performed an action.
    PlayerAction {
        seat: usize,
        action: String,
        amount: u32,
    },
    /// Hand result.
    HandResult {
        winners: Vec<WinnerInfo>,
        hands_shown: Vec<HandShown>,
    },
    /// Chat from another player.
    ChatMsg { sender: String, text: String },
    /// Error.
    Error { message: String },
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerView {
    pub name: String,
    pub seat: usize,
    pub chips: u32,
    pub folded: bool,
    pub current_bet: u32,
    pub is_dealer: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WinnerInfo {
    pub seat: usize,
    pub amount: u32,
    pub hand_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandShown {
    pub seat: usize,
    pub cards: [String; 2],
}
