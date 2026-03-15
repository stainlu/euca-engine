use serde::{Deserialize, Serialize};

/// Stable network identity for an entity (persists across server ticks).
/// Different from ECS Entity (which uses generational indices internally).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct NetworkId(pub u64);

/// Marker component: entities with this are replicated to clients.
#[derive(Clone, Copy, Debug)]
pub struct Replicated;

/// Messages sent from client to server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ClientMessage {
    /// Client wants to connect.
    Connect { player_name: String },

    /// Client disconnecting gracefully.
    Disconnect,

    /// Timestamped input from the client.
    Input {
        tick: u64,
        pressed_keys: Vec<euca_input::InputKey>,
        mouse_position: [f32; 2],
        mouse_delta: [f32; 2],
    },
}

/// Messages sent from server to client.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ServerMessage {
    /// Connection accepted. Tells client their player entity's network ID.
    Welcome {
        player_network_id: NetworkId,
        tick: u64,
    },

    /// Server rejected the connection.
    Rejected { reason: String },

    /// State snapshot: positions of all replicated entities.
    /// Sent periodically or on connect.
    StateSnapshot {
        tick: u64,
        entities: Vec<EntityState>,
    },

    /// Delta update: only entities that changed since last send.
    StateDelta {
        tick: u64,
        changed: Vec<EntityState>,
        despawned: Vec<NetworkId>,
    },
}

/// Serialized state of one entity for network transmission.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityState {
    pub network_id: NetworkId,
    pub position: [f32; 3],
    pub rotation: [f32; 4], // quaternion xyzw
    pub scale: [f32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_message_serializes() {
        let msg = ClientMessage::Connect {
            player_name: "TestPlayer".into(),
        };
        let bytes = bincode::serialize(&msg).unwrap();
        let decoded: ClientMessage = bincode::deserialize(&bytes).unwrap();
        match decoded {
            ClientMessage::Connect { player_name } => assert_eq!(player_name, "TestPlayer"),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn server_message_serializes() {
        let msg = ServerMessage::StateDelta {
            tick: 42,
            changed: vec![EntityState {
                network_id: NetworkId(1),
                position: [1.0, 2.0, 3.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            }],
            despawned: vec![NetworkId(5)],
        };
        let bytes = bincode::serialize(&msg).unwrap();
        let decoded: ServerMessage = bincode::deserialize(&bytes).unwrap();
        match decoded {
            ServerMessage::StateDelta {
                tick,
                changed,
                despawned,
            } => {
                assert_eq!(tick, 42);
                assert_eq!(changed.len(), 1);
                assert_eq!(despawned.len(), 1);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn network_id_equality() {
        assert_eq!(NetworkId(1), NetworkId(1));
        assert_ne!(NetworkId(1), NetworkId(2));
    }
}
