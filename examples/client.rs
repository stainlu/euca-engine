//! Multiplayer client example.
//!
//! Run: cargo run -p euca-game --example server
//! Then in another terminal: cargo run -p euca-game --example client
//! Multiple clients can connect simultaneously.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use euca_ecs::{Entity, Query, World};
use euca_math::{Quat, Transform, Vec3};
use euca_net::{
    ClientMessage, ClientPrediction, GameClient, NetworkId, PacketHeader, ServerMessage,
    UdpTransport, apply_prediction_system, reconcile_entity, record_prediction_for_entity,
};
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

const SERVER_ADDR: &str = "127.0.0.1:7777";

/// Maps network entity IDs to local ECS entities.
struct NetworkEntityMap {
    network_to_ecs: HashMap<NetworkId, Entity>,
}

impl NetworkEntityMap {
    fn new() -> Self {
        Self {
            network_to_ecs: HashMap::new(),
        }
    }
}

struct ClientApp {
    transport: Option<UdpTransport>,
    client: GameClient,
    server_addr: SocketAddr,

    // ECS world holds all entities (replicated from server)
    world: World,
    net_map: NetworkEntityMap,

    // Rendering
    survey: HardwareSurvey,
    wgpu_instance: wgpu::Instance,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    cube_mesh: Option<MeshHandle>,
    player_material: Option<MaterialHandle>,
    other_material: Option<MaterialHandle>,

    // Input
    pressed_keys: Vec<euca_input::InputKey>,
    send_seq: u32,

    window_attrs: WindowAttributes,
}

impl ClientApp {
    fn new() -> Self {
        let (survey, wgpu_instance) = HardwareSurvey::detect();
        let server_addr: SocketAddr = SERVER_ADDR.parse().unwrap();

        Self {
            transport: None,
            client: GameClient::new(),
            server_addr,
            world: World::new(),
            net_map: NetworkEntityMap::new(),
            survey,
            wgpu_instance,
            window: None,
            gpu: None,
            renderer: None,
            cube_mesh: None,
            player_material: None,
            other_material: None,
            pressed_keys: Vec::new(),
            send_seq: 0,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — Multiplayer Client")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
        }
    }

    fn connect(&mut self) {
        let transport =
            UdpTransport::bind("0.0.0.0:0".parse().unwrap()).expect("Failed to bind client socket");
        log::info!("Client bound to {}", transport.local_addr().unwrap());

        // Send connect message
        let msg = ClientMessage::Connect {
            player_name: format!("Player_{}", std::process::id()),
        };
        self.send_to_server(&transport, &msg);
        self.transport = Some(transport);

        log::info!("Connecting to server at {SERVER_ADDR}...");
    }

    fn send_to_server(&mut self, transport: &UdpTransport, msg: &ClientMessage) {
        let payload = bincode::serialize(msg).expect("serialize failed");
        let header = PacketHeader {
            sequence: self.send_seq,
            ack: 0,
            ack_bits: 0,
        };
        self.send_seq += 1;
        let _ = transport.send_packet(&header, &payload, self.server_addr);
    }

    fn receive_packets(&mut self) {
        let transport = match &mut self.transport {
            Some(t) => t,
            None => return,
        };

        while let Some((_addr, _header, payload)) = transport.recv_packet() {
            if let Ok(msg) = bincode::deserialize::<ServerMessage>(&payload) {
                self.client.handle_server_message(msg);
            }
        }

        // Sync GameClient state into the ECS world.
        self.sync_world_from_client();
    }

    /// Synchronise the ECS world with the authoritative state held by `GameClient`.
    ///
    /// For each entity the server knows about:
    ///  - If it does not yet exist locally, spawn it with render components.
    ///  - Update its `LocalTransform` from the server-reported position/rotation.
    ///  - For the local player entity, reconcile predictions against the server.
    ///
    /// Entities that the server despawned are removed from the world.
    fn sync_world_from_client(&mut self) {
        let cube_mesh = match self.cube_mesh {
            Some(m) => m,
            None => return, // Renderer not ready yet
        };
        let player_mat = match self.player_material {
            Some(m) => m,
            None => return,
        };
        let other_mat = match self.other_material {
            Some(m) => m,
            None => return,
        };

        let my_nid = self.client.player_network_id;
        let server_tick = self.client.server_tick;

        // Spawn or update entities from the client's replicated state.
        let entity_states: Vec<_> = self.client.entities.values().cloned().collect();

        for state in &entity_states {
            let is_local_player = my_nid == Some(state.network_id);
            let material = if is_local_player {
                player_mat
            } else {
                other_mat
            };

            let entity = if let Some(&existing) = self.net_map.network_to_ecs.get(&state.network_id)
            {
                existing
            } else {
                // Spawn a new ECS entity for this network entity.
                let e = self
                    .world
                    .spawn(LocalTransform(Transform::from_translation(Vec3::new(
                        state.position[0],
                        state.position[1],
                        state.position[2],
                    ))));
                self.world.insert(e, GlobalTransform::default());
                self.world.insert(e, MeshRenderer { mesh: cube_mesh });
                self.world.insert(e, MaterialRef { handle: material });
                self.world.insert(e, state.network_id);

                if is_local_player {
                    self.world.insert(e, ClientPrediction::new());
                }

                self.net_map.network_to_ecs.insert(state.network_id, e);
                e
            };

            // Update transform from server state.
            if let Some(lt) = self.world.get_mut::<LocalTransform>(entity) {
                lt.0.translation =
                    Vec3::new(state.position[0], state.position[1], state.position[2]);
                lt.0.rotation = Quat::from_xyzw(
                    state.rotation[0],
                    state.rotation[1],
                    state.rotation[2],
                    state.rotation[3],
                );
                lt.0.scale = Vec3::new(state.scale[0], state.scale[1], state.scale[2]);
            }

            // Ensure the material stays correct (player vs. other).
            if let Some(mat_ref) = self.world.get_mut::<MaterialRef>(entity) {
                mat_ref.handle = material;
            }

            // Reconcile prediction for the local player.
            if is_local_player {
                reconcile_entity(&mut self.world, entity, server_tick, state.position);
            }
        }

        // Remove entities that the server despawned.
        for nid in self.client.drain_despawned() {
            if let Some(entity) = self.net_map.network_to_ecs.remove(&nid) {
                self.world.despawn(entity);
            }
        }
    }

    fn send_input(&mut self) {
        let transport = match &self.transport {
            Some(t) => t,
            None => return,
        };

        let msg = ClientMessage::Input {
            tick: 0,
            pressed_keys: self.pressed_keys.clone(),
            mouse_position: [0.0, 0.0],
            mouse_delta: [0.0, 0.0],
        };

        let payload = bincode::serialize(&msg).expect("serialize failed");
        let header = PacketHeader {
            sequence: self.send_seq,
            ack: 0,
            ack_bits: 0,
        };
        self.send_seq += 1;
        let _ = transport.send_packet(&header, &payload, self.server_addr);
    }

    /// Record a prediction for the local player entity, then apply pending
    /// prediction corrections to all entities with `ClientPrediction`.
    fn run_prediction(&mut self) {
        let my_nid = match self.client.player_network_id {
            Some(nid) => nid,
            None => return,
        };
        let entity = match self.net_map.network_to_ecs.get(&my_nid) {
            Some(&e) => e,
            None => return,
        };

        let tick = self.client.server_tick;
        let input_snapshot = bincode::serialize(&self.pressed_keys).unwrap_or_default();
        record_prediction_for_entity(&mut self.world, entity, tick, input_snapshot);

        apply_prediction_system(&mut self.world);
    }

    fn render(&mut self) {
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };
        let renderer = match &mut self.renderer {
            Some(r) => r,
            None => return,
        };

        // Propagate LocalTransform -> GlobalTransform.
        euca_scene::transform_propagation_system(&mut self.world);

        // Collect draw commands from the ECS world (same pattern as hello_cubes / editor).
        let draw_commands = collect_draw_commands(&self.world);

        // Camera looking at origin
        let camera = Camera::new(Vec3::new(0.0, 10.0, -15.0), Vec3::new(0.0, 1.0, 0.0));
        let light = DirectionalLight::default();
        let ambient = AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.3,
        };

        renderer.draw(gpu, &camera, &light, &ambient, &draw_commands);
    }
}

/// Collect draw commands from all entities with GlobalTransform + MeshRenderer + MaterialRef.
fn collect_draw_commands(world: &World) -> Vec<DrawCommand> {
    let query = Query::<(Entity, &GlobalTransform, &MeshRenderer, &MaterialRef)>::new(world);
    query
        .iter()
        .map(|(e, gt, mr, mat)| {
            let mut model_matrix = gt.0.to_matrix();
            if let Some(offset) = world.get::<GroundOffset>(e) {
                model_matrix.cols[3][1] += offset.0;
            }
            DrawCommand {
                mesh: mr.mesh,
                material: mat.handle,
                model_matrix,
                aabb: None,
                is_water: false,
            }
        })
        .collect()
}

impl ApplicationHandler for ClientApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop.create_window(self.window_attrs.clone()).unwrap();
            let gpu = GpuContext::new(window, &self.survey, &self.wgpu_instance);
            let mut renderer = Renderer::new(&gpu);

            let cube_mesh = renderer.upload_mesh(&gpu, &Mesh::cube());
            let player_mat =
                renderer.upload_material(&gpu, &Material::new([0.2, 0.8, 0.2, 1.0], 0.0, 0.5)); // green = you
            let other_mat =
                renderer.upload_material(&gpu, &Material::new([0.8, 0.2, 0.2, 1.0], 0.0, 0.5)); // red = others

            self.window = Some(gpu.window.clone());
            self.gpu = Some(gpu);
            self.renderer = Some(renderer);
            self.cube_mesh = Some(cube_mesh);
            self.player_material = Some(player_mat);
            self.other_material = Some(other_mat);

            self.connect();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref c),
                        state,
                        ..
                    },
                ..
            } => {
                let key = euca_input::InputKey::Key(c.to_string());
                if state == ElementState::Pressed {
                    if !self.pressed_keys.contains(&key) {
                        self.pressed_keys.push(key);
                    }
                } else {
                    self.pressed_keys.retain(|k| k != &key);
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                    if let Some(r) = &mut self.renderer {
                        r.resize(gpu);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                // 1. Receive authoritative state from server and sync into ECS world.
                self.receive_packets();

                // 2. Send local input to server.
                self.send_input();

                // 3. Run client-side prediction (record + apply corrections).
                self.run_prediction();

                // 4. Render from the ECS world.
                self.render();

                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("Euca Multiplayer Client — connecting to {SERVER_ADDR}");

    let event_loop = EventLoop::new().unwrap();
    let mut app = ClientApp::new();
    event_loop.run_app(&mut app).unwrap();
}
