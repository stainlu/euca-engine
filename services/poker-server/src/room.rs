//! Room management: tracks poker rooms, player connections, and coordinates
//! game actions with the poker table.

use std::collections::HashMap;

use tokio::sync::mpsc;

use crate::msg::{HandShown, PlayerView, ServerMsg, WinnerInfo};
use crate::poker::{ActionOutcome, Phase, PokerTable};

/// Maximum seats per table.
const MAX_SEATS: usize = 9;

// ---------------------------------------------------------------------------
// Room
// ---------------------------------------------------------------------------

/// A single poker room: one table plus connected players.
pub struct Room {
    pub id: String,
    /// Player ID -> unbounded sender for outgoing `ServerMsg`.
    connections: HashMap<String, mpsc::UnboundedSender<ServerMsg>>,
    /// The authoritative table state.
    pub table: PokerTable,
    /// Player ID -> seat index.
    seats: HashMap<String, usize>,
    /// Seat index -> player ID (reverse mapping).
    seat_to_player: HashMap<usize, String>,
}

impl Room {
    pub fn new(id: String, small_blind: u32, big_blind: u32) -> Self {
        Self {
            id,
            connections: HashMap::new(),
            table: PokerTable::new(MAX_SEATS, small_blind, big_blind),
            seats: HashMap::new(),
            seat_to_player: HashMap::new(),
        }
    }

    /// Join a player to this room. Returns their seat index.
    pub fn join(
        &mut self,
        player_id: &str,
        name: &str,
        buy_in: u32,
        tx: mpsc::UnboundedSender<ServerMsg>,
    ) -> Result<usize, String> {
        if self.seats.contains_key(player_id) {
            return Err("Already in this room".to_string());
        }
        let seat = self
            .table
            .seat_player(name.to_string(), buy_in)
            .ok_or_else(|| "Room is full".to_string())?;
        self.connections.insert(player_id.to_string(), tx);
        self.seats.insert(player_id.to_string(), seat);
        self.seat_to_player.insert(seat, player_id.to_string());
        Ok(seat)
    }

    /// Remove a player from this room.
    pub fn leave(&mut self, player_id: &str) {
        if let Some(seat) = self.seats.remove(player_id) {
            self.table.unseat_player(seat);
            self.seat_to_player.remove(&seat);
        }
        self.connections.remove(player_id);
    }

    /// Handle a player marking themselves ready. Returns true if a hand started.
    pub fn set_ready(&mut self, player_id: &str) -> bool {
        let Some(&seat) = self.seats.get(player_id) else {
            return false;
        };
        let should_start = self.table.set_ready(seat);
        if should_start {
            self.table.start_hand();
            self.broadcast_table_state();
            return true;
        }
        false
    }

    /// Handle a player action (fold/check/call/raise).
    pub fn handle_action(
        &mut self,
        player_id: &str,
        action: &str,
        amount: Option<u32>,
    ) -> Result<(), String> {
        let &seat = self.seats.get(player_id).ok_or("Not in this room")?;

        let outcome = self.table.apply_action(seat, action, amount)?;

        // Broadcast the action to all players.
        self.broadcast(ServerMsg::PlayerAction {
            seat,
            action: action.to_string(),
            amount: amount.unwrap_or(0),
        });

        match outcome {
            ActionOutcome::Continue => {
                self.broadcast_table_state();
            }
            ActionOutcome::LastPlayerWins { seat: winner } => {
                let winner_name = self.table.players[winner].name.clone();
                let pot = self.table.pot;
                self.table.award_pot(winner);

                self.broadcast(ServerMsg::HandResult {
                    winners: vec![WinnerInfo {
                        seat: winner,
                        amount: pot,
                        hand_name: format!("{} wins (others folded)", winner_name),
                    }],
                    hands_shown: vec![],
                });
                self.broadcast_table_state();
            }
            ActionOutcome::AdvancePhase { new_phase } => {
                if new_phase == Phase::Showdown {
                    self.resolve_showdown();
                } else {
                    self.broadcast_table_state();
                }
            }
        }

        Ok(())
    }

    /// Resolve the showdown: evaluate hands, broadcast results, award pot.
    fn resolve_showdown(&mut self) {
        let (winners, hands_shown) = self.table.showdown();

        let winner_infos: Vec<WinnerInfo> = winners
            .iter()
            .map(|(seat, amount, hand_name)| WinnerInfo {
                seat: *seat,
                amount: *amount,
                hand_name: hand_name.clone(),
            })
            .collect();

        let shown: Vec<HandShown> = hands_shown
            .iter()
            .map(|(seat, cards)| HandShown {
                seat: *seat,
                cards: [cards[0].display(), cards[1].display()],
            })
            .collect();

        self.broadcast(ServerMsg::HandResult {
            winners: winner_infos,
            hands_shown: shown,
        });

        self.table.finish_showdown();
        self.broadcast_table_state();
    }

    /// Send a chat message from one player to all.
    pub fn chat(&self, player_id: &str, text: &str) {
        let sender = self
            .seats
            .get(player_id)
            .map(|&seat| self.table.players[seat].name.clone())
            .unwrap_or_else(|| "unknown".to_string());

        self.broadcast(ServerMsg::ChatMsg {
            sender,
            text: text.to_string(),
        });
    }

    /// Broadcast a message to every connected player.
    pub fn broadcast(&self, msg: ServerMsg) {
        for tx in self.connections.values() {
            // Ignore send errors (player may have disconnected).
            let _ = tx.send(msg.clone());
        }
    }

    /// Build and send a `TableState` to each player (with per-player hole cards).
    pub fn broadcast_table_state(&self) {
        let community_cards: Vec<String> = self
            .table
            .community_cards
            .iter()
            .map(|c| c.display())
            .collect();

        let players: Vec<PlayerView> = self
            .table
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.active)
            .map(|(i, p)| PlayerView {
                name: p.name.clone(),
                seat: i,
                chips: p.chips,
                folded: p.folded,
                current_bet: p.current_bet,
                is_dealer: i == self.table.dealer,
            })
            .collect();

        let action_on = if self.table.phase != Phase::Waiting && self.table.phase != Phase::Showdown
        {
            Some(self.table.action_on)
        } else {
            None
        };

        for (pid, tx) in &self.connections {
            let your_cards = self.seats.get(pid).and_then(|&seat| {
                self.table.players[seat]
                    .hole_cards
                    .map(|cards| [cards[0].display(), cards[1].display()])
            });

            let msg = ServerMsg::TableState {
                phase: self.table.phase.name().to_string(),
                community_cards: community_cards.clone(),
                pot: self.table.pot,
                players: players.clone(),
                your_cards,
                action_on,
                current_bet: self.table.current_bet,
            };
            let _ = tx.send(msg);
        }
    }

    /// Send a message to a specific player.
    #[allow(dead_code)]
    pub fn send_to(&self, player_id: &str, msg: ServerMsg) {
        if let Some(tx) = self.connections.get(player_id) {
            let _ = tx.send(msg);
        }
    }

    /// Number of connected players.
    pub fn player_count(&self) -> usize {
        self.connections.len()
    }

    /// Whether the room is empty.
    pub fn is_empty(&self) -> bool {
        self.connections.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Room Manager
// ---------------------------------------------------------------------------

/// Central registry of all active poker rooms.
pub struct RoomManager {
    rooms: HashMap<String, Room>,
}

impl RoomManager {
    pub fn new() -> Self {
        Self {
            rooms: HashMap::new(),
        }
    }

    /// Create a new room with the given blind levels. Returns the room ID.
    pub fn create_room(&mut self, small_blind: u32, big_blind: u32) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        let room = Room::new(id.clone(), small_blind, big_blind);
        self.rooms.insert(id.clone(), room);
        id
    }

    /// Get a mutable reference to a room.
    pub fn get_room_mut(&mut self, room_id: &str) -> Option<&mut Room> {
        self.rooms.get_mut(room_id)
    }

    /// Get an immutable reference to a room.
    pub fn get_room(&self, room_id: &str) -> Option<&Room> {
        self.rooms.get(room_id)
    }

    /// List all rooms with summary info.
    pub fn list_rooms(&self) -> Vec<RoomSummary> {
        self.rooms
            .values()
            .map(|r| RoomSummary {
                id: r.id.clone(),
                players: r.player_count(),
                phase: r.table.phase.name().to_string(),
            })
            .collect()
    }

    /// Remove a room by ID.
    #[allow(dead_code)]
    pub fn remove_room(&mut self, room_id: &str) {
        self.rooms.remove(room_id);
    }

    /// Clean up empty rooms.
    pub fn cleanup_empty_rooms(&mut self) {
        self.rooms.retain(|_, r| !r.is_empty());
    }
}

impl Default for RoomManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of a room for the listing endpoint.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RoomSummary {
    pub id: String,
    pub players: usize,
    pub phase: String,
}
