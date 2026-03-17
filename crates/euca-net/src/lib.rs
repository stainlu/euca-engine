mod client;
pub mod prediction;
mod protocol;
mod server;
mod transport;

pub use client::GameClient;
pub use prediction::ClientPrediction;
pub use protocol::{ClientMessage, EntityState, NetworkId, Replicated, ServerMessage};
pub use server::GameServer;
pub use transport::{MAX_PACKET_SIZE, PacketHeader, ReliableTransport, UdpTransport};
