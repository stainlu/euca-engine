//! Unified game server: human players via UDP + AI agents via HTTP.
//!
//! Start: `cargo run --example agent_game`
//! Human: `cargo run --example client`
//! Agent: `cargo run --example agent_client`

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euca_agent::bridge::{ActionRequest, AgentBridge, JoinRequest, JoinResponse, PlayerView};
use euca_ecs::{Entity, Query, World};
use euca_math::{Transform, Vec3};
use euca_net::{
    ClientMessage, EntityState, GameServer, NetworkId, PacketHeader, ServerMessage, UdpTransport,
};
use euca_physics::{Collider, PhysicsBody, PhysicsConfig, Velocity, physics_step_system};
use euca_scene::{GlobalTransform, LocalTransform};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;

const TICK_RATE: u64 = 60;
const UDP_PORT: u16 = 7777;
const HTTP_PORT: u16 = 8080;
const PLAYER_SPEED: f32 = 5.0;

/// Shared game state accessible from both the game loop and HTTP handlers.
type SharedGame = Arc<Mutex<GameState>>;

struct GameState {
    world: World,
    server: GameServer,
    bridge: AgentBridge,
    players: HashMap<SocketAddr, (Entity, NetworkId)>,
    net_to_entity: HashMap<NetworkId, Entity>,
}

impl GameState {
    fn spawn_player(&mut self, network_id: NetworkId, addr: SocketAddr) -> Entity {
        let spawn_x = (self.players.len() as f32) * 3.0 - 3.0;
        let entity = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                spawn_x, 1.0, 0.0,
            ))));
        self.world.insert(entity, GlobalTransform::default());
        self.world.insert(entity, PhysicsBody::dynamic());
        self.world.insert(entity, Velocity::default());
        self.world
            .insert(entity, Collider::aabb(0.5, 0.5, 0.5).with_restitution(0.3));
        self.world.insert(entity, network_id);

        self.players.insert(addr, (entity, network_id));
        self.net_to_entity.insert(network_id, entity);
        entity
    }

    fn handle_input(&mut self, addr: SocketAddr, pressed_keys: &[euca_input::InputKey]) {
        let (entity, _) = match self.players.get(&addr) {
            Some(p) => *p,
            None => return,
        };

        let mut move_dir = Vec3::ZERO;
        for key in pressed_keys {
            if let euca_input::InputKey::Key(k) = key {
                match k.to_uppercase().as_str() {
                    "W" => move_dir.z += 1.0,
                    "S" => move_dir.z -= 1.0,
                    "A" => move_dir.x -= 1.0,
                    "D" => move_dir.x += 1.0,
                    _ => {}
                }
            }
        }

        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize() * PLAYER_SPEED;
        }

        if let Some(vel) = self.world.get_mut::<Velocity>(entity) {
            vel.linear = Vec3::new(move_dir.x, vel.linear.y, move_dir.z);
        }
    }

    fn collect_entities(&self) -> Vec<EntityState> {
        let query = Query::<(Entity, &GlobalTransform, &NetworkId)>::new(&self.world);
        query
            .iter()
            .map(|(_, gt, nid)| EntityState {
                network_id: *nid,
                position: [gt.0.translation.x, gt.0.translation.y, gt.0.translation.z],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            })
            .collect()
    }
}

// ── HTTP handlers for agent players ──

#[derive(Serialize)]
struct GameStatus {
    tick: u64,
    player_count: usize,
    entity_count: u32,
}

async fn status(State(game): State<SharedGame>) -> Json<GameStatus> {
    let g = game.lock().unwrap();
    Json(GameStatus {
        tick: g.world.current_tick(),
        player_count: g.players.len(),
        entity_count: g.world.entity_count(),
    })
}

async fn join(State(game): State<SharedGame>, Json(req): Json<JoinRequest>) -> Json<JoinResponse> {
    let mut g = game.lock().unwrap();
    let tick = g.world.current_tick();
    let (network_id, fake_addr) = g.bridge.join(&mut g.server, req.name, tick);

    // Spawn player entity for the agent
    g.spawn_player(network_id, fake_addr);

    // Drain the welcome message (agent bridge will handle it)
    g.bridge.collect_server_messages(&mut g.server);

    Json(JoinResponse {
        player_id: network_id.0,
        tick,
    })
}

async fn action(
    State(game): State<SharedGame>,
    Json(req): Json<ActionRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut g = game.lock().unwrap();
    let tick = g.world.current_tick();

    // Feed input into game server (same pipeline as UDP)
    if !g
        .bridge
        .send_input(&mut g.server, req.player_id, req.keys, tick)
    {
        return Err(StatusCode::NOT_FOUND);
    }

    // Process the input immediately (translate to entity velocity)
    if let Some(fake_addr) = g.bridge.agent_addr(req.player_id) {
        let incoming = g.server.drain_incoming();
        for (addr, msg) in incoming {
            if let ClientMessage::Input {
                pressed_keys,
                mouse_position: _,
                mouse_delta: _,
                tick: _,
            } = msg
            {
                g.handle_input(addr, &pressed_keys);
            }
            // Re-queue non-input messages
            if addr != fake_addr {
                g.server.push_incoming(addr, ClientMessage::Disconnect); // won't happen, just safety
            }
        }
    }

    Ok(Json(serde_json::json!({"ok": true, "tick": tick})))
}

async fn player_view(
    State(game): State<SharedGame>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> Result<Json<PlayerView>, StatusCode> {
    let g = game.lock().unwrap();
    g.bridge
        .get_player_view(id)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn leave(
    State(game): State<SharedGame>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let player_id = req["player_id"].as_u64().unwrap_or(0);
    let mut g = game.lock().unwrap();

    // Remove player entity
    if let Some(fake_addr) = g.bridge.agent_addr(player_id) {
        if let Some((entity, nid)) = g.players.remove(&fake_addr) {
            g.world.despawn(entity);
            g.net_to_entity.remove(&nid);
        }
    }

    g.bridge.leave(&mut g.server, player_id);
    Json(serde_json::json!({"ok": true}))
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Create shared game state
    let mut world = World::new();
    world.insert_resource(PhysicsConfig::new());

    // Ground
    let ground = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
    world.insert(ground, GlobalTransform::default());
    world.insert(ground, PhysicsBody::fixed());
    world.insert(ground, Collider::aabb(20.0, 0.1, 20.0));

    let game: SharedGame = Arc::new(Mutex::new(GameState {
        world,
        server: GameServer::new(),
        bridge: AgentBridge::new(),
        players: HashMap::new(),
        net_to_entity: HashMap::new(),
    }));

    // Start HTTP server for agents in a separate thread
    let http_game = game.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let router = Router::new()
                .route("/", get(status))
                .route("/join", post(join))
                .route("/action", post(action))
                .route("/player/{id}/view", get(player_view))
                .route("/leave", post(leave))
                .with_state(http_game);

            let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{HTTP_PORT}"))
                .await
                .unwrap();
            log::info!("Agent HTTP server on http://127.0.0.1:{HTTP_PORT}");
            axum::serve(listener, router).await.unwrap();
        });
    });

    // UDP transport for human players
    let transport = UdpTransport::bind(format!("0.0.0.0:{UDP_PORT}").parse().unwrap())
        .expect("Failed to bind UDP");
    let tick_duration = Duration::from_micros(1_000_000 / TICK_RATE);

    log::info!("Game server: UDP on :{UDP_PORT}, HTTP on :{HTTP_PORT}");
    log::info!("Human: cargo run --example client");
    log::info!("Agent: cargo run --example agent_client");

    let mut send_seq: u32 = 0;

    loop {
        let tick_start = Instant::now();
        let mut g = game.lock().unwrap();

        // Receive UDP packets from human players
        while let Some((addr, _header, payload)) = transport.recv_packet() {
            if let Ok(msg) = bincode::deserialize::<ClientMessage>(&payload) {
                match msg {
                    ClientMessage::Connect { player_name } => {
                        if !g.players.contains_key(&addr) {
                            let network_id = g.server.handle_connect(
                                addr,
                                player_name.clone(),
                                g.world.current_tick(),
                            );
                            let entity = g.spawn_player(network_id, addr);
                            log::info!("Human '{}' connected: Entity {}", player_name, entity);
                        }
                    }
                    ClientMessage::Disconnect => {
                        if let Some((entity, nid)) = g.players.remove(&addr) {
                            g.world.despawn(entity);
                            g.net_to_entity.remove(&nid);
                            g.server.handle_disconnect(&addr);
                        }
                    }
                    ClientMessage::Input {
                        pressed_keys,
                        mouse_position: _,
                        mouse_delta: _,
                        tick: _,
                    } => {
                        g.handle_input(addr, &pressed_keys);
                    }
                }
            }
        }

        // Process any pending agent inputs
        let incoming = g.server.drain_incoming();
        for (addr, msg) in incoming {
            if let ClientMessage::Input {
                pressed_keys,
                mouse_position: _,
                mouse_delta: _,
                tick: _,
            } = msg
            {
                g.handle_input(addr, &pressed_keys);
            }
        }

        // Step simulation
        physics_step_system(&mut g.world);
        euca_scene::transform_propagation_system(&mut g.world);
        g.world.tick();

        // Broadcast state
        if !g.players.is_empty() {
            let entities = g.collect_entities();
            let tick = g.world.current_tick();

            // Send to UDP human players
            let delta = ServerMessage::StateDelta {
                tick,
                changed: entities.clone(),
                despawned: vec![],
            };
            for addr in g.players.keys() {
                // Only send to real UDP addresses (not agent fake_addrs)
                if !g
                    .bridge
                    .is_agent(g.players.get(addr).map(|(_, nid)| nid.0).unwrap_or(0))
                {
                    let payload = bincode::serialize(&delta).unwrap();
                    let header = PacketHeader {
                        sequence: send_seq,
                        ack: 0,
                        ack_bits: 0,
                    };
                    send_seq += 1;
                    let _ = transport.send_packet(&header, &payload, *addr);
                }
            }

            // Send welcome/snapshot to new UDP players
            for (addr, msg) in g.server.drain_outgoing() {
                let payload = bincode::serialize(&msg).unwrap();
                let header = PacketHeader {
                    sequence: send_seq,
                    ack: 0,
                    ack_bits: 0,
                };
                send_seq += 1;
                let _ = transport.send_packet(&header, &payload, addr);
            }

            // Update agent views
            // Manually set the state for agent players
            for (nid_val, _) in g.bridge.agents_iter() {
                // We need to update the agent's view with current entities
                // This is handled by the bridge's collect mechanism
            }

            // Feed state to agent bridge as StateDelta
            let agent_nids: Vec<u64> = g
                .players
                .values()
                .filter(|(_, nid)| g.bridge.is_agent(nid.0))
                .map(|(_, nid)| nid.0)
                .collect();

            for nid in agent_nids {
                if let Some(fake_addr) = g.bridge.agent_addr(nid) {
                    g.server
                        .send_delta(fake_addr, tick, entities.clone(), vec![]);
                }
            }
            g.bridge.collect_server_messages(&mut g.server);
        }

        drop(g); // Release lock before sleeping

        let elapsed = tick_start.elapsed();
        if elapsed < tick_duration {
            std::thread::sleep(tick_duration - elapsed);
        }
    }
}
