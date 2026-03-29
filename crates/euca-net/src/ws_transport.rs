//! WebSocket transport for browser-compatible networking.
//!
//! Provides the same send/receive interface as the UDP and QUIC transports,
//! but over WebSocket connections. This enables WASM/browser clients to
//! connect to game servers.
//!
//! - **Native**: Uses `tokio-tungstenite` for async WebSocket.
//! - **WASM**: Uses `web-sys` WebSocket API (not yet implemented).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

/// Errors that can occur in the WebSocket transport layer.
#[derive(Debug, thiserror::Error)]
pub enum WsError {
    /// The initial WebSocket connection handshake failed.
    #[error("connection failed: {0}")]
    Connection(String),

    /// The WebSocket connection has been closed.
    #[error("disconnected")]
    Disconnected,

    /// A send operation failed.
    #[error("send failed: {0}")]
    Send(String),

    /// The server failed to bind to the requested address.
    #[error("bind failed: {0}")]
    Bind(String),
}

// ---------------------------------------------------------------------------
// Client transport (native)
// ---------------------------------------------------------------------------

/// WebSocket client transport for connecting to a game server.
///
/// Internally spawns two Tokio tasks for reading and writing on the socket.
/// Communication between the tasks and the caller happens via unbounded
/// channels, keeping the public API non-async for easy integration into
/// game loops.
pub struct WsTransport {
    /// Sender for outgoing binary messages.
    tx: mpsc::UnboundedSender<Vec<u8>>,
    /// Receiver for incoming binary messages.
    rx: mpsc::UnboundedReceiver<Vec<u8>>,
    /// Shared flag set to `false` when the connection drops.
    connected: Arc<AtomicBool>,
}

impl WsTransport {
    /// Connect to a WebSocket server at the given URL (e.g. `ws://127.0.0.1:9001`).
    pub async fn connect(url: &str) -> Result<Self, WsError> {
        let (ws_stream, _) = tokio_tungstenite::connect_async(url)
            .await
            .map_err(|e| WsError::Connection(e.to_string()))?;

        let (mut write, mut read) = ws_stream.split();

        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (incoming_tx, incoming_rx) = mpsc::unbounded_channel::<Vec<u8>>();

        let connected = Arc::new(AtomicBool::new(true));

        // Spawn task: forward outgoing channel messages into the WebSocket sink.
        let conn_flag = Arc::clone(&connected);
        tokio::spawn(async move {
            while let Some(data) = outgoing_rx.recv().await {
                if write.send(Message::Binary(data)).await.is_err() {
                    conn_flag.store(false, Ordering::SeqCst);
                    break;
                }
            }
        });

        // Spawn task: forward incoming WebSocket messages into the channel.
        let conn_flag = Arc::clone(&connected);
        tokio::spawn(async move {
            while let Some(Ok(msg)) = read.next().await {
                match msg {
                    Message::Binary(data) => {
                        if incoming_tx.send(data.to_vec()).is_err() {
                            break;
                        }
                    }
                    Message::Close(_) => break,
                    // Ignore ping/pong/text -- tungstenite handles pong automatically.
                    _ => {}
                }
            }
            conn_flag.store(false, Ordering::SeqCst);
        });

        log::info!("WebSocket connected to {url}");

        Ok(Self {
            tx: outgoing_tx,
            rx: incoming_rx,
            connected,
        })
    }

    /// Send binary data over the WebSocket.
    pub fn send(&self, data: &[u8]) -> Result<(), WsError> {
        if !self.connected.load(Ordering::SeqCst) {
            return Err(WsError::Disconnected);
        }
        self.tx
            .send(data.to_vec())
            .map_err(|_| WsError::Disconnected)
    }

    /// Drain all pending incoming messages (non-blocking).
    pub fn recv(&mut self) -> Vec<Vec<u8>> {
        let mut messages = Vec::new();
        while let Ok(msg) = self.rx.try_recv() {
            messages.push(msg);
        }
        messages
    }

    /// Whether the underlying connection is still alive.
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// Server transport (native only -- browsers never run servers)
// ---------------------------------------------------------------------------

/// Handle to a single connected WebSocket client on the server side.
struct WsClientHandle {
    tx: mpsc::UnboundedSender<Vec<u8>>,
    rx: mpsc::UnboundedReceiver<Vec<u8>>,
}

/// Events produced by [`WsServer::poll`].
#[derive(Debug)]
pub enum WsEvent {
    /// A new client connected.
    Connected(u64),
    /// A client disconnected.
    Disconnected(u64),
    /// A binary message arrived from the given client.
    Message(u64, Vec<u8>),
}

/// WebSocket server that accepts browser client connections.
///
/// Each accepted connection is handled by a pair of Tokio tasks (read/write).
/// The server exposes a synchronous polling API so the game loop can consume
/// events without awaiting.
pub struct WsServer {
    /// Per-client channels.
    clients: HashMap<u64, WsClientHandle>,
    /// Channel through which accept tasks deliver new clients.
    new_connections: mpsc::UnboundedReceiver<(u64, WsClientHandle)>,
    /// Channel through which read tasks report disconnections.
    disconnections: mpsc::UnboundedReceiver<u64>,
    /// Retained to keep the channel open for spawned read tasks.
    _disconnect_tx: mpsc::UnboundedSender<u64>,
    /// Retained to keep the shared counter alive for the accept loop.
    _next_client_id: Arc<std::sync::atomic::AtomicU64>,
}

impl WsServer {
    /// Start listening for WebSocket connections on `addr` (e.g. `"127.0.0.1:9001"`).
    pub async fn bind(addr: &str) -> Result<Self, WsError> {
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| WsError::Bind(e.to_string()))?;

        let (conn_tx, conn_rx) = mpsc::unbounded_channel();
        let (disconnect_tx, disconnect_rx) = mpsc::unbounded_channel();
        let next_id = Arc::new(std::sync::atomic::AtomicU64::new(1));

        let next_id_clone = Arc::clone(&next_id);
        let disc_tx = disconnect_tx.clone();

        // Spawn the accept loop.
        tokio::spawn(async move {
            loop {
                let Ok((stream, _peer)) = listener.accept().await else {
                    break;
                };

                let Ok(ws_stream) = tokio_tungstenite::accept_async(stream).await else {
                    continue;
                };

                let client_id = next_id_clone.fetch_add(1, Ordering::SeqCst);

                let (mut ws_write, mut ws_read) = ws_stream.split();

                // Channel: server -> client.
                let (to_client_tx, mut to_client_rx) = mpsc::unbounded_channel::<Vec<u8>>();

                // Channel: client -> server.
                let (from_client_tx, from_client_rx) = mpsc::unbounded_channel::<Vec<u8>>();

                // Write task.
                tokio::spawn(async move {
                    while let Some(data) = to_client_rx.recv().await {
                        if ws_write.send(Message::Binary(data)).await.is_err() {
                            break;
                        }
                    }
                });

                // Read task.
                let disc = disc_tx.clone();
                tokio::spawn(async move {
                    while let Some(Ok(msg)) = ws_read.next().await {
                        match msg {
                            Message::Binary(data) => {
                                if from_client_tx.send(data.to_vec()).is_err() {
                                    break;
                                }
                            }
                            Message::Close(_) => break,
                            _ => {}
                        }
                    }
                    let _ = disc.send(client_id);
                });

                let handle = WsClientHandle {
                    tx: to_client_tx,
                    rx: from_client_rx,
                };

                if conn_tx.send((client_id, handle)).is_err() {
                    break;
                }
            }
        });

        log::info!("WebSocket server listening on {addr}");

        Ok(Self {
            clients: HashMap::new(),
            new_connections: conn_rx,
            disconnections: disconnect_rx,
            _disconnect_tx: disconnect_tx,
            _next_client_id: next_id,
        })
    }

    /// Poll for new connections, disconnections, and incoming messages.
    ///
    /// This is designed to be called once per game tick from a synchronous
    /// context. All events since the last call are returned.
    pub fn poll(&mut self) -> Vec<WsEvent> {
        let mut events = Vec::new();

        // Accept new connections.
        while let Ok((id, handle)) = self.new_connections.try_recv() {
            events.push(WsEvent::Connected(id));
            self.clients.insert(id, handle);
        }

        // Process disconnections.
        while let Ok(id) = self.disconnections.try_recv() {
            self.clients.remove(&id);
            events.push(WsEvent::Disconnected(id));
        }

        // Drain messages from each connected client.
        for (&id, handle) in &mut self.clients {
            while let Ok(msg) = handle.rx.try_recv() {
                events.push(WsEvent::Message(id, msg));
            }
        }

        events
    }

    /// Send binary data to a specific client.
    pub fn send_to(&self, client_id: u64, data: &[u8]) -> Result<(), WsError> {
        let handle = self.clients.get(&client_id).ok_or(WsError::Disconnected)?;
        handle
            .tx
            .send(data.to_vec())
            .map_err(|_| WsError::Disconnected)
    }

    /// Broadcast binary data to every connected client.
    pub fn broadcast(&self, data: &[u8]) {
        for handle in self.clients.values() {
            let _ = handle.tx.send(data.to_vec());
        }
    }

    /// Forcibly disconnect a client.
    pub fn disconnect(&mut self, client_id: u64) {
        // Dropping the handle closes the channels, which causes the
        // write task to exit, closing the WebSocket.
        self.clients.remove(&client_id);
    }

    /// Number of currently connected clients.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Iterator over connected client IDs.
    pub fn client_ids(&self) -> Vec<u64> {
        self.clients.keys().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_error_display() {
        let err = WsError::Connection("timeout".into());
        assert_eq!(err.to_string(), "connection failed: timeout");

        let err = WsError::Disconnected;
        assert_eq!(err.to_string(), "disconnected");

        let err = WsError::Send("buffer full".into());
        assert_eq!(err.to_string(), "send failed: buffer full");

        let err = WsError::Bind("address in use".into());
        assert_eq!(err.to_string(), "bind failed: address in use");
    }

    #[tokio::test]
    async fn test_ws_transport_send_recv() {
        // Start a server on an ephemeral port.
        let server_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = server_listener.local_addr().unwrap();

        // Spawn a minimal echo server for this test.
        tokio::spawn(async move {
            let (stream, _) = server_listener.accept().await.unwrap();
            let ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            let (mut write, mut read) = ws.split();
            while let Some(Ok(msg)) = read.next().await {
                if msg.is_binary() {
                    if write.send(msg).await.is_err() {
                        break;
                    }
                }
            }
        });

        // Connect a WsTransport client.
        let mut client = WsTransport::connect(&format!("ws://{addr}")).await.unwrap();
        assert!(client.is_connected());

        // Send a message and wait briefly for the echo.
        client.send(b"hello").unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let msgs = client.recv();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0], b"hello");
    }

    #[tokio::test]
    async fn test_ws_server_events() {
        // WsServer doesn't expose the bound address, so we bind a TcpListener
        // first to capture the ephemeral port, then rebind the WsServer on it.

        // Bind with a known ephemeral port that we capture.
        let tcp = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = tcp.local_addr().unwrap();
        drop(tcp); // free the port

        let mut server = WsServer::bind(&addr.to_string()).await.unwrap();

        // Connect a client.
        let (ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}"))
            .await
            .unwrap();

        let (mut write, read) = ws.split();

        // Give the server time to accept.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let events = server.poll();
        let connected: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WsEvent::Connected(_)))
            .collect();
        assert_eq!(connected.len(), 1);

        let client_id = match &connected[0] {
            WsEvent::Connected(id) => *id,
            _ => unreachable!(),
        };

        // Send a message from the client.
        write
            .send(Message::Binary(b"ping".to_vec().into()))
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let events = server.poll();
        let messages: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WsEvent::Message(_, _)))
            .collect();
        assert_eq!(messages.len(), 1);

        match &messages[0] {
            WsEvent::Message(id, data) => {
                assert_eq!(*id, client_id);
                assert_eq!(data, b"ping");
            }
            _ => unreachable!(),
        }

        // Drop both halves to fully close the connection.
        drop(write);
        drop(read);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let events = server.poll();
        let disconnected: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WsEvent::Disconnected(_)))
            .collect();
        assert_eq!(disconnected.len(), 1);
    }

    #[tokio::test]
    async fn test_ws_message_types() {
        // Verify that binary protocol messages roundtrip through the
        // WebSocket transport correctly using bincode serialization,
        // matching the pattern in protocol.rs.

        use crate::protocol::{ClientMessage, EntityState, NetworkId, ServerMessage};

        let tcp = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = tcp.local_addr().unwrap();
        drop(tcp);

        let mut server = WsServer::bind(&addr.to_string()).await.unwrap();
        let mut client = WsTransport::connect(&format!("ws://{addr}")).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        server.poll(); // accept

        // Client sends a Connect message.
        let msg = ClientMessage::Connect {
            player_name: "Alice".into(),
        };
        let encoded = bincode::serialize(&msg).unwrap();
        client.send(&encoded).unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let events = server.poll();
        let data = events.iter().find_map(|e| match e {
            WsEvent::Message(_, d) => Some(d.clone()),
            _ => None,
        });
        assert!(data.is_some());
        let decoded: ClientMessage = bincode::deserialize(&data.unwrap()).unwrap();
        match decoded {
            ClientMessage::Connect { player_name } => assert_eq!(player_name, "Alice"),
            _ => panic!("expected Connect"),
        }

        // Server sends a Welcome back.
        let client_id = server.client_ids()[0];
        let reply = ServerMessage::Welcome {
            player_network_id: NetworkId(7),
            tick: 42,
        };
        let reply_bytes = bincode::serialize(&reply).unwrap();
        server.send_to(client_id, &reply_bytes).unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let msgs = client.recv();
        assert_eq!(msgs.len(), 1);
        let decoded: ServerMessage = bincode::deserialize(&msgs[0]).unwrap();
        match decoded {
            ServerMessage::Welcome {
                player_network_id,
                tick,
            } => {
                assert_eq!(player_network_id, NetworkId(7));
                assert_eq!(tick, 42);
            }
            _ => panic!("expected Welcome"),
        }

        // Test a more complex message: StateDelta with entity state.
        let delta = ServerMessage::StateDelta {
            tick: 100,
            changed: vec![EntityState {
                network_id: NetworkId(1),
                position: [1.0, 2.0, 3.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            }],
            despawned: vec![NetworkId(5)],
        };
        let delta_bytes = bincode::serialize(&delta).unwrap();
        server.send_to(client_id, &delta_bytes).unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let msgs = client.recv();
        assert_eq!(msgs.len(), 1);
        let decoded: ServerMessage = bincode::deserialize(&msgs[0]).unwrap();
        match decoded {
            ServerMessage::StateDelta {
                tick,
                changed,
                despawned,
            } => {
                assert_eq!(tick, 100);
                assert_eq!(changed.len(), 1);
                assert_eq!(changed[0].position, [1.0, 2.0, 3.0]);
                assert_eq!(despawned, vec![NetworkId(5)]);
            }
            _ => panic!("expected StateDelta"),
        }
    }
}
