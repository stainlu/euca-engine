//! Real-time communication provider trait.
//!
//! For services that need persistent bidirectional communication
//! (matchmaking lobbies, live game state sync, chat).
//! Games implement this to connect via WebSocket, WebRTC, or custom protocols.
//!
//! This is for **service-level** real-time communication (matchmaking, chat),
//! NOT for game networking (which uses euca-net's UDP/QUIC transport).

use crate::error::ServiceError;

/// Connection state for a real-time channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
}

/// A message received from the real-time channel.
#[derive(Debug, Clone)]
pub struct RealtimeMessage {
    /// Channel or topic the message belongs to.
    pub channel: String,
    /// Raw message payload.
    pub payload: Vec<u8>,
}

/// Trait for real-time communication providers.
///
/// This is for service-level real-time communication (matchmaking, chat),
/// NOT for game networking (which uses euca-net's UDP/QUIC transport).
pub trait RealtimeProvider: Send + Sync {
    /// Current connection state.
    fn state(&self) -> ConnectionState;

    /// Connect to a real-time endpoint.
    fn connect(&mut self, url: &str) -> Result<(), ServiceError>;

    /// Disconnect from the endpoint.
    fn disconnect(&mut self);

    /// Subscribe to a channel/topic.
    fn subscribe(&mut self, channel: &str) -> Result<(), ServiceError>;

    /// Unsubscribe from a channel/topic.
    fn unsubscribe(&mut self, channel: &str) -> Result<(), ServiceError>;

    /// Send a message to a channel.
    fn send(&self, channel: &str, payload: &[u8]) -> Result<(), ServiceError>;

    /// Poll for received messages (non-blocking).
    /// Returns all messages received since the last call.
    fn poll(&mut self) -> Vec<RealtimeMessage>;
}
