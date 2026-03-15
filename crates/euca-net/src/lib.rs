mod client;
mod protocol;
mod server;

pub use client::GameClient;
pub use protocol::{ClientMessage, NetworkId, Replicated, ServerMessage};
pub use server::GameServer;
