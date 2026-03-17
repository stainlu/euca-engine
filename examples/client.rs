use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use euca_math::Vec3;
use euca_net::{ClientMessage, GameClient, NetworkId, PacketHeader, ServerMessage, UdpTransport};
use euca_render::*;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

const SERVER_ADDR: &str = "127.0.0.1:7777";

struct ClientApp {
    transport: Option<UdpTransport>,
    client: GameClient,
    server_addr: SocketAddr,

    // Rendering
    survey: HardwareSurvey,
    wgpu_instance: wgpu::Instance,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    cube_mesh: Option<MeshHandle>,
    player_material: Option<MaterialHandle>,
    other_material: Option<MaterialHandle>,

    // Local state
    /// Maps NetworkId → local entity position
    entity_positions: HashMap<NetworkId, [f32; 3]>,
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
            survey,
            wgpu_instance,
            window: None,
            gpu: None,
            renderer: None,
            cube_mesh: None,
            player_material: None,
            other_material: None,
            entity_positions: HashMap::new(),
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

        // Update local entity positions from client state
        self.entity_positions.clear();
        for (nid, state) in &self.client.entities {
            self.entity_positions.insert(*nid, state.position);
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

    fn render(&self) {
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };
        let renderer = match &self.renderer {
            Some(r) => r,
            None => return,
        };
        let cube_mesh = match self.cube_mesh {
            Some(m) => m,
            None => return,
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

        // Build draw commands from network state
        let mut draw_commands: Vec<DrawCommand> = Vec::new();

        // Ground plane (local, not networked)
        // Just render entities from network state
        for (nid, pos) in &self.entity_positions {
            let mat = if Some(*nid) == my_nid {
                player_mat
            } else {
                other_mat
            };
            draw_commands.push(DrawCommand {
                mesh: cube_mesh,
                material: mat,
                model_matrix: euca_math::Mat4::from_translation(Vec3::new(pos[0], pos[1], pos[2])),
            });
        }

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
                // Network: receive state from server
                self.receive_packets();

                // Network: send input to server
                self.send_input();

                // Render
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
