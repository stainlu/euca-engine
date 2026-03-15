use crate::protocol::*;
use std::collections::HashMap;
use std::net::SocketAddr;

/// A connected player on the server.
#[derive(Debug)]
pub struct ConnectedPlayer {
    pub name: String,
    pub network_id: NetworkId,
    pub addr: SocketAddr,
    /// The last tick the client acknowledged receiving.
    pub last_ack_tick: u64,
}

/// Game server that manages player connections and state replication.
///
/// This is NOT the simulation — it wraps around whatever ECS World the game uses.
/// The server handles:
/// - Accepting/rejecting connections
/// - Receiving player inputs
/// - Tracking which state each client has received (for delta sync)
pub struct GameServer {
    /// Connected players indexed by socket address.
    players: HashMap<SocketAddr, ConnectedPlayer>,
    /// Next network ID to assign.
    next_network_id: u64,
    /// Pending incoming messages (filled by network layer, consumed by game loop).
    pub incoming: Vec<(SocketAddr, ClientMessage)>,
    /// Pending outgoing messages (filled by game loop, consumed by network layer).
    pub outgoing: Vec<(SocketAddr, ServerMessage)>,
}

impl GameServer {
    pub fn new() -> Self {
        Self {
            players: HashMap::new(),
            next_network_id: 1,
            incoming: Vec::new(),
            outgoing: Vec::new(),
        }
    }

    /// Allocate a new unique network ID.
    pub fn allocate_network_id(&mut self) -> NetworkId {
        let id = NetworkId(self.next_network_id);
        self.next_network_id += 1;
        id
    }

    /// Process a connect request. Returns the assigned NetworkId on success.
    pub fn handle_connect(
        &mut self,
        addr: SocketAddr,
        player_name: String,
        current_tick: u64,
    ) -> NetworkId {
        let network_id = self.allocate_network_id();

        self.players.insert(
            addr,
            ConnectedPlayer {
                name: player_name,
                network_id,
                addr,
                last_ack_tick: 0,
            },
        );

        // Send welcome message
        self.outgoing.push((
            addr,
            ServerMessage::Welcome {
                player_network_id: network_id,
                tick: current_tick,
            },
        ));

        log::info!(
            "Player connected: {} (NetworkId: {})",
            self.players[&addr].name,
            network_id.0
        );
        network_id
    }

    /// Handle a player disconnect.
    pub fn handle_disconnect(&mut self, addr: &SocketAddr) {
        if let Some(player) = self.players.remove(addr) {
            log::info!("Player disconnected: {}", player.name);
        }
    }

    /// Get the number of connected players.
    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    /// Get a connected player by address.
    pub fn get_player(&self, addr: &SocketAddr) -> Option<&ConnectedPlayer> {
        self.players.get(addr)
    }

    /// Get all connected player addresses.
    pub fn player_addrs(&self) -> Vec<SocketAddr> {
        self.players.keys().copied().collect()
    }

    /// Send a state snapshot to a specific client.
    pub fn send_snapshot(&mut self, addr: SocketAddr, tick: u64, entities: Vec<EntityState>) {
        self.outgoing
            .push((addr, ServerMessage::StateSnapshot { tick, entities }));
    }

    /// Send a state delta to a specific client.
    pub fn send_delta(
        &mut self,
        addr: SocketAddr,
        tick: u64,
        changed: Vec<EntityState>,
        despawned: Vec<NetworkId>,
    ) {
        self.outgoing.push((
            addr,
            ServerMessage::StateDelta {
                tick,
                changed,
                despawned,
            },
        ));
    }

    /// Broadcast a state delta to all connected clients.
    pub fn broadcast_delta(
        &mut self,
        tick: u64,
        changed: Vec<EntityState>,
        despawned: Vec<NetworkId>,
    ) {
        let addrs: Vec<SocketAddr> = self.players.keys().copied().collect();
        for addr in addrs {
            self.send_delta(addr, tick, changed.clone(), despawned.clone());
        }
    }

    /// Drain outgoing messages (network layer calls this to send).
    pub fn drain_outgoing(&mut self) -> Vec<(SocketAddr, ServerMessage)> {
        std::mem::take(&mut self.outgoing)
    }

    /// Push an incoming message (network layer calls this when receiving).
    pub fn push_incoming(&mut self, addr: SocketAddr, msg: ClientMessage) {
        self.incoming.push((addr, msg));
    }

    /// Drain incoming messages (game loop calls this to process).
    pub fn drain_incoming(&mut self) -> Vec<(SocketAddr, ClientMessage)> {
        std::mem::take(&mut self.incoming)
    }
}

impl Default for GameServer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_addr() -> SocketAddr {
        "127.0.0.1:12345".parse().unwrap()
    }

    #[test]
    fn connect_and_disconnect() {
        let mut server = GameServer::new();
        let addr = test_addr();

        let nid = server.handle_connect(addr, "Alice".into(), 0);
        assert_eq!(server.player_count(), 1);
        assert_eq!(nid, NetworkId(1));

        // Check welcome message was queued
        let outgoing = server.drain_outgoing();
        assert_eq!(outgoing.len(), 1);
        match &outgoing[0].1 {
            ServerMessage::Welcome {
                player_network_id, ..
            } => assert_eq!(*player_network_id, NetworkId(1)),
            _ => panic!("Expected Welcome"),
        }

        server.handle_disconnect(&addr);
        assert_eq!(server.player_count(), 0);
    }

    #[test]
    fn broadcast_delta() {
        let mut server = GameServer::new();
        let addr1: SocketAddr = "127.0.0.1:1001".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:1002".parse().unwrap();

        server.handle_connect(addr1, "P1".into(), 0);
        server.handle_connect(addr2, "P2".into(), 0);
        server.drain_outgoing(); // clear welcome messages

        server.broadcast_delta(
            10,
            vec![EntityState {
                network_id: NetworkId(1),
                position: [1.0, 2.0, 3.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            }],
            vec![],
        );

        let outgoing = server.drain_outgoing();
        assert_eq!(outgoing.len(), 2); // one delta per player
    }

    #[test]
    fn unique_network_ids() {
        let mut server = GameServer::new();
        let id1 = server.allocate_network_id();
        let id2 = server.allocate_network_id();
        let id3 = server.allocate_network_id();
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
    }
}
