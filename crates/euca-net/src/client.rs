use crate::protocol::*;
use std::collections::HashMap;

/// Client-side network state.
///
/// Tracks the last known state of all replicated entities received from the server.
/// The game client uses this to update its local ECS world.
pub struct GameClient {
    /// Our player's network ID (assigned by server on connect).
    pub player_network_id: Option<NetworkId>,
    /// Current server tick (last received).
    pub server_tick: u64,
    /// Last known state of each network entity.
    pub entities: HashMap<NetworkId, EntityState>,
    /// Entities that were despawned by the server.
    pub despawned: Vec<NetworkId>,
    /// Whether we're connected to the server.
    pub connected: bool,
    /// Pending messages to send to server.
    pub outgoing: Vec<ClientMessage>,
}

impl GameClient {
    pub fn new() -> Self {
        Self {
            player_network_id: None,
            server_tick: 0,
            entities: HashMap::new(),
            despawned: Vec::new(),
            connected: false,
            outgoing: Vec::new(),
        }
    }

    /// Queue a connect request.
    pub fn connect(&mut self, player_name: String) {
        self.outgoing.push(ClientMessage::Connect { player_name });
    }

    /// Queue a disconnect.
    pub fn disconnect(&mut self) {
        self.outgoing.push(ClientMessage::Disconnect);
        self.connected = false;
    }

    /// Queue input to send to server.
    pub fn send_input(&mut self, input: &euca_input::InputState) {
        let snapshot = euca_input::InputSnapshot::capture(input);
        self.outgoing.push(ClientMessage::Input {
            tick: snapshot.tick,
            pressed_keys: snapshot.pressed_keys,
            mouse_position: snapshot.mouse_position,
            mouse_delta: snapshot.mouse_delta,
        });
    }

    /// Handle a message from the server.
    pub fn handle_server_message(&mut self, msg: ServerMessage) {
        match msg {
            ServerMessage::Welcome {
                player_network_id,
                tick,
            } => {
                self.player_network_id = Some(player_network_id);
                self.server_tick = tick;
                self.connected = true;
                log::info!(
                    "Connected! Player NetworkId: {}, server tick: {}",
                    player_network_id.0,
                    tick
                );
            }
            ServerMessage::Rejected { reason } => {
                log::warn!("Connection rejected: {}", reason);
                self.connected = false;
            }
            ServerMessage::StateSnapshot { tick, entities } => {
                self.server_tick = tick;
                self.entities.clear();
                for state in entities {
                    self.entities.insert(state.network_id, state);
                }
            }
            ServerMessage::StateDelta {
                tick,
                changed,
                despawned,
            } => {
                self.server_tick = tick;
                for state in changed {
                    self.entities.insert(state.network_id, state);
                }
                for id in &despawned {
                    self.entities.remove(id);
                }
                self.despawned.extend(despawned);
            }
        }
    }

    /// Drain outgoing messages.
    pub fn drain_outgoing(&mut self) -> Vec<ClientMessage> {
        std::mem::take(&mut self.outgoing)
    }

    /// Drain despawned entities (game loop consumes this to remove local entities).
    pub fn drain_despawned(&mut self) -> Vec<NetworkId> {
        std::mem::take(&mut self.despawned)
    }
}

impl Default for GameClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_and_welcome() {
        let mut client = GameClient::new();
        assert!(!client.connected);

        client.connect("TestPlayer".into());
        assert_eq!(client.outgoing.len(), 1);

        // Simulate server welcome
        client.handle_server_message(ServerMessage::Welcome {
            player_network_id: NetworkId(42),
            tick: 100,
        });

        assert!(client.connected);
        assert_eq!(client.player_network_id, Some(NetworkId(42)));
        assert_eq!(client.server_tick, 100);
    }

    #[test]
    fn state_snapshot_replaces_all() {
        let mut client = GameClient::new();

        // Old state
        client.entities.insert(
            NetworkId(1),
            EntityState {
                network_id: NetworkId(1),
                position: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            },
        );

        // Snapshot with different entity
        client.handle_server_message(ServerMessage::StateSnapshot {
            tick: 50,
            entities: vec![EntityState {
                network_id: NetworkId(2),
                position: [5.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            }],
        });

        assert_eq!(client.entities.len(), 1);
        assert!(client.entities.contains_key(&NetworkId(2)));
        assert!(!client.entities.contains_key(&NetworkId(1)));
    }

    #[test]
    fn delta_updates_and_despawns() {
        let mut client = GameClient::new();

        // Initial state
        client.entities.insert(
            NetworkId(1),
            EntityState {
                network_id: NetworkId(1),
                position: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            },
        );
        client.entities.insert(
            NetworkId(2),
            EntityState {
                network_id: NetworkId(2),
                position: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            },
        );

        // Delta: update entity 1, despawn entity 2
        client.handle_server_message(ServerMessage::StateDelta {
            tick: 10,
            changed: vec![EntityState {
                network_id: NetworkId(1),
                position: [99.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            }],
            despawned: vec![NetworkId(2)],
        });

        assert_eq!(client.entities.len(), 1);
        assert_eq!(client.entities[&NetworkId(1)].position[0], 99.0);
        assert!(!client.entities.contains_key(&NetworkId(2)));
    }
}
