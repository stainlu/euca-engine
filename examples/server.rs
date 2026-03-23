//! Multiplayer server example.
//!
//! Run: cargo run -p euca-game --example server
//! Then in another terminal: cargo run -p euca-game --example client
//! Multiple clients can connect simultaneously.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use euca_ecs::{Entity, Query, World};
use euca_math::{Transform, Vec3};
use euca_net::{
    ClientMessage, EntityState, GameServer, NetworkId, PacketHeader, ServerMessage, UdpTransport,
};
use euca_physics::{Collider, PhysicsBody, PhysicsConfig, Velocity, physics_step_system};
use euca_scene::{GlobalTransform, LocalTransform};

const TICK_RATE: u64 = 60;
const PORT: u16 = 7777;
const PLAYER_SPEED: f32 = 5.0;

struct ServerState {
    world: World,
    server: GameServer,
    transport: UdpTransport,
    /// Maps player address → ECS entity + NetworkId
    players: HashMap<SocketAddr, (Entity, NetworkId)>,
    /// Maps NetworkId → ECS entity
    net_to_entity: HashMap<NetworkId, Entity>,
    send_seq: u32,
    last_broadcast_tick: u32,
}

impl ServerState {
    fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(PhysicsConfig::new());

        // Ground
        let ground = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(ground, GlobalTransform::default());
        world.insert(ground, PhysicsBody::fixed());
        world.insert(ground, Collider::aabb(20.0, 0.1, 20.0));

        let transport = UdpTransport::bind(format!("0.0.0.0:{PORT}").parse().unwrap())
            .expect("Failed to bind server socket");

        log::info!("Server listening on 0.0.0.0:{PORT}");

        Self {
            world,
            server: GameServer::new(),
            transport,
            players: HashMap::new(),
            net_to_entity: HashMap::new(),
            send_seq: 0,
            last_broadcast_tick: 0,
        }
    }

    fn receive_packets(&mut self) {
        while let Some((addr, _header, payload)) = self.transport.recv_packet() {
            if let Ok(msg) = bincode::deserialize::<ClientMessage>(&payload) {
                match msg {
                    ClientMessage::Connect { player_name } => {
                        self.handle_connect(addr, player_name);
                    }
                    ClientMessage::Disconnect => {
                        self.handle_disconnect(addr);
                    }
                    ClientMessage::Input {
                        pressed_keys,
                        mouse_position: _,
                        mouse_delta: _,
                        tick: _,
                    } => {
                        self.handle_input(addr, &pressed_keys);
                    }
                }
            }
        }
    }

    fn handle_connect(&mut self, addr: SocketAddr, name: String) {
        if self.players.contains_key(&addr) {
            return; // Already connected
        }

        let network_id = self
            .server
            .handle_connect(addr, name.clone(), self.world.current_tick());

        // Spawn player entity
        let spawn_x = (self.players.len() as f32) * 3.0 - 3.0;
        let entity = self
            .world
            .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                spawn_x, 1.0, 0.0,
            ))));
        self.world.insert(entity, GlobalTransform::default());
        self.world.insert(entity, PhysicsBody::dynamic());
        self.world.insert(entity, Velocity::default());
        self.world.insert(entity, Collider::aabb(0.5, 0.5, 0.5));
        self.world.insert(entity, network_id);

        self.players.insert(addr, (entity, network_id));
        self.net_to_entity.insert(network_id, entity);

        log::info!(
            "Player '{}' connected: {:?} → Entity {}",
            name,
            network_id,
            entity
        );

        // Send welcome
        let welcome = ServerMessage::Welcome {
            player_network_id: network_id,
            tick: self.world.current_tick(),
        };
        self.send_message(addr, &welcome);

        // Send full state snapshot
        self.send_full_snapshot(addr);
    }

    fn handle_disconnect(&mut self, addr: SocketAddr) {
        if let Some((entity, nid)) = self.players.remove(&addr) {
            self.world.despawn(entity);
            self.net_to_entity.remove(&nid);
            self.server.handle_disconnect(&addr);
            log::info!("Player disconnected: {:?}", nid);
        }
    }

    fn handle_input(&mut self, addr: SocketAddr, pressed_keys: &[euca_input::InputKey]) {
        let (entity, _) = match self.players.get(&addr) {
            Some(p) => *p,
            None => return,
        };

        // Convert pressed keys to movement velocity
        let mut move_dir = Vec3::ZERO;
        for key in pressed_keys {
            match key {
                euca_input::InputKey::Key(k) if k == "w" || k == "W" => {
                    move_dir.z += 1.0;
                }
                euca_input::InputKey::Key(k) if k == "s" || k == "S" => {
                    move_dir.z -= 1.0;
                }
                euca_input::InputKey::Key(k) if k == "a" || k == "A" => {
                    move_dir.x -= 1.0;
                }
                euca_input::InputKey::Key(k) if k == "d" || k == "D" => {
                    move_dir.x += 1.0;
                }
                _ => {}
            }
        }

        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize() * PLAYER_SPEED;
        }

        if let Some(vel) = self.world.get_mut::<Velocity>(entity) {
            vel.linear = Vec3::new(move_dir.x, vel.linear.y, move_dir.z);
        }
    }

    fn tick(&mut self) {
        physics_step_system(&mut self.world);
        euca_scene::transform_propagation_system(&mut self.world);
        self.world.tick();
    }

    fn broadcast_state(&mut self) {
        let current_tick = self.world.current_tick() as u32;

        // Collect all networked entities
        let entities: Vec<EntityState> = {
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
        };

        // Send delta to all clients
        let delta = ServerMessage::StateDelta {
            tick: self.world.current_tick(),
            changed: entities,
            despawned: vec![],
        };

        let addrs: Vec<SocketAddr> = self.players.keys().copied().collect();
        for addr in addrs {
            self.send_message(addr, &delta);
        }

        self.last_broadcast_tick = current_tick;
    }

    fn send_full_snapshot(&mut self, addr: SocketAddr) {
        let entities: Vec<EntityState> = {
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
        };

        let snapshot = ServerMessage::StateSnapshot {
            tick: self.world.current_tick(),
            entities,
        };
        self.send_message(addr, &snapshot);
    }

    fn send_message(&mut self, addr: SocketAddr, msg: &ServerMessage) {
        let payload = bincode::serialize(msg).expect("serialize failed");
        let header = PacketHeader {
            sequence: self.send_seq,
            ack: 0,
            ack_bits: 0,
        };
        self.send_seq += 1;
        let _ = self.transport.send_packet(&header, &payload, addr);
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut state = ServerState::new();
    let tick_duration = Duration::from_micros(1_000_000 / TICK_RATE);

    log::info!(
        "Euca Game Server running at {} ticks/sec on port {}",
        TICK_RATE,
        PORT
    );
    log::info!("Waiting for clients to connect...");

    loop {
        let tick_start = Instant::now();

        // Receive network packets
        state.receive_packets();

        // Send outgoing messages from GameServer
        for (addr, msg) in state.server.drain_outgoing() {
            state.send_message(addr, &msg);
        }

        // Step simulation
        state.tick();

        // Broadcast state to all clients
        if !state.players.is_empty() {
            state.broadcast_state();
        }

        // Sleep to maintain tick rate
        let elapsed = tick_start.elapsed();
        if elapsed < tick_duration {
            std::thread::sleep(tick_duration - elapsed);
        }
    }
}
