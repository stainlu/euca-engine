use euca_net::{ClientMessage, EntityState, GameServer, NetworkId, ServerMessage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;

/// An agent player connected via HTTP (not UDP).
struct AgentPlayer {
    network_id: NetworkId,
    fake_addr: SocketAddr,
    /// Buffered server messages for this agent (read via /player/{id}/view).
    buffered_states: Vec<EntityState>,
    last_tick: u64,
}

/// Bridges HTTP agent requests to the GameServer's player pipeline.
///
/// Agents join, send input, and observe state through HTTP endpoints.
/// The bridge translates these into the same ClientMessage/ServerMessage
/// flow that UDP players use.
pub struct AgentBridge {
    agents: HashMap<u64, AgentPlayer>, // keyed by NetworkId.0
    next_fake_port: u16,
}

impl AgentBridge {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            next_fake_port: 50000,
        }
    }

    /// Agent joins the game. Returns (network_id, fake_addr).
    pub fn join(
        &mut self,
        server: &mut GameServer,
        name: String,
        current_tick: u64,
    ) -> (NetworkId, SocketAddr) {
        let fake_addr: SocketAddr = format!("127.0.0.1:{}", self.next_fake_port)
            .parse()
            .unwrap();
        self.next_fake_port += 1;

        let network_id = server.handle_connect(fake_addr, name, current_tick);

        self.agents.insert(
            network_id.0,
            AgentPlayer {
                network_id,
                fake_addr,
                buffered_states: Vec::new(),
                last_tick: current_tick,
            },
        );

        (network_id, fake_addr)
    }

    /// Agent sends input (same as keyboard press for human players).
    pub fn send_input(
        &self,
        server: &mut GameServer,
        network_id: u64,
        keys: Vec<String>,
        current_tick: u64,
    ) -> bool {
        let agent = match self.agents.get(&network_id) {
            Some(a) => a,
            None => return false,
        };

        let pressed_keys: Vec<euca_input::InputKey> = keys
            .iter()
            .map(|k| euca_input::InputKey::Key(k.to_uppercase()))
            .collect();

        server.push_incoming(
            agent.fake_addr,
            ClientMessage::Input {
                tick: current_tick,
                pressed_keys,
                mouse_position: [0.0, 0.0],
                mouse_delta: [0.0, 0.0],
            },
        );

        true
    }

    /// Collect outgoing messages from the server that are addressed to agent players.
    /// Call this after the game server processes a tick and broadcasts state.
    pub fn collect_server_messages(&mut self, server: &mut GameServer) {
        let outgoing = server.drain_outgoing();

        for (addr, msg) in outgoing {
            // Check if this is addressed to an agent player
            let mut is_agent = false;
            for agent in self.agents.values_mut() {
                if agent.fake_addr == addr {
                    match &msg {
                        ServerMessage::StateDelta { tick, changed, .. } => {
                            agent.buffered_states = changed.clone();
                            agent.last_tick = *tick;
                        }
                        ServerMessage::StateSnapshot { tick, entities } => {
                            agent.buffered_states = entities.clone();
                            agent.last_tick = *tick;
                        }
                        _ => {}
                    }
                    is_agent = true;
                    break;
                }
            }

            // If not an agent, re-queue for UDP sending
            if !is_agent {
                server.outgoing.push((addr, msg));
            }
        }
    }

    /// Get the current game state for an agent player.
    pub fn get_player_view(&self, network_id: u64) -> Option<PlayerView> {
        let agent = self.agents.get(&network_id)?;
        Some(PlayerView {
            player_id: agent.network_id.0,
            tick: agent.last_tick,
            entities: agent
                .buffered_states
                .iter()
                .map(|es| PlayerEntityState {
                    network_id: es.network_id.0,
                    position: es.position,
                    rotation: es.rotation,
                    scale: es.scale,
                })
                .collect(),
        })
    }

    /// Agent leaves the game.
    pub fn leave(&mut self, server: &mut GameServer, network_id: u64) -> bool {
        if let Some(agent) = self.agents.remove(&network_id) {
            server.handle_disconnect(&agent.fake_addr);
            true
        } else {
            false
        }
    }

    /// Check if a network_id belongs to an agent.
    pub fn is_agent(&self, network_id: u64) -> bool {
        self.agents.contains_key(&network_id)
    }

    /// Iterate over agent (network_id, fake_addr) pairs.
    pub fn agents_iter(&self) -> impl Iterator<Item = (u64, SocketAddr)> + '_ {
        self.agents
            .iter()
            .map(|(nid, agent)| (*nid, agent.fake_addr))
    }

    /// Get the fake_addr for an agent (used by game server to look up player).
    pub fn agent_addr(&self, network_id: u64) -> Option<SocketAddr> {
        self.agents.get(&network_id).map(|a| a.fake_addr)
    }
}

impl Default for AgentBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON-serializable player view (returned by GET /player/{id}/view).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PlayerView {
    pub player_id: u64,
    pub tick: u64,
    pub entities: Vec<PlayerEntityState>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PlayerEntityState {
    pub network_id: u64,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

/// Request to join a game as an agent player.
#[derive(Deserialize)]
pub struct JoinRequest {
    #[serde(default = "default_agent_name")]
    pub name: String,
}

fn default_agent_name() -> String {
    "Agent".to_string()
}

/// Response after joining.
#[derive(Serialize)]
pub struct JoinResponse {
    pub player_id: u64,
    pub tick: u64,
}

/// Request to send input as an agent player.
#[derive(Deserialize)]
pub struct ActionRequest {
    pub player_id: u64,
    pub keys: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_join_and_leave() {
        let mut server = GameServer::new();
        let mut bridge = AgentBridge::new();

        let (nid, _addr) = bridge.join(&mut server, "TestAgent".into(), 0);
        assert_eq!(nid, NetworkId(1));
        assert!(bridge.is_agent(nid.0));
        assert_eq!(server.player_count(), 1);

        bridge.leave(&mut server, nid.0);
        assert!(!bridge.is_agent(nid.0));
        assert_eq!(server.player_count(), 0);
    }

    #[test]
    fn agent_send_input() {
        let mut server = GameServer::new();
        let mut bridge = AgentBridge::new();

        let (nid, _) = bridge.join(&mut server, "Agent".into(), 0);
        server.drain_outgoing(); // clear welcome

        let ok = bridge.send_input(&mut server, nid.0, vec!["w".into(), "d".into()], 1);
        assert!(ok);

        let incoming = server.drain_incoming();
        assert_eq!(incoming.len(), 1);
    }

    #[test]
    fn player_view_empty_initially() {
        let mut server = GameServer::new();
        let mut bridge = AgentBridge::new();

        let (nid, _) = bridge.join(&mut server, "Agent".into(), 0);
        let view = bridge.get_player_view(nid.0).unwrap();
        assert_eq!(view.entities.len(), 0);
    }
}
