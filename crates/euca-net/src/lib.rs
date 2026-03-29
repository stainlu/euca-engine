//! Multiplayer networking: client/server, replication, prediction, and transport.
//!
//! Provides [`GameClient`] and [`GameServer`] for connection management,
//! [`ClientPrediction`] for lag-hiding, and a [`replication`] layer for
//! automatic ECS state synchronisation over the network.

pub mod bandwidth;
mod client;
pub mod interest;
pub mod prediction;
mod protocol;
pub mod quic_transport;
pub mod replication;
mod server;
pub mod tick_rate;
mod transport;
#[cfg(feature = "websocket")]
pub mod ws_transport;

pub use bandwidth::{BandwidthBudget, PriorityCalculator, select_entities_for_replication};
pub use client::GameClient;
pub use interest::{InterestConfig, InterestManager, SpatialGrid, interest_culling_system};
pub use prediction::{
    ClientPrediction, apply_prediction_system, reconcile_entity, record_prediction_for_entity,
};
pub use protocol::{ClientMessage, EntityState, NetworkId, Replicated, ServerMessage};
pub use quic_transport::{QuicTransport, generate_self_signed_cert};
pub use replication::{
    ClientReplicationReceiver, ClientRpc, ComponentData, ComponentDeserializationRegistry,
    ComponentReplicationRegistry, EntityReplicationData, FieldId, FieldRegistry,
    PendingReplication, ReplicatedComponent, ReplicatedField, ReplicationManager,
    ReplicationPriority, ReplicationState, ReplicationUpdate, ServerRpc,
    replication_collect_system, replication_receive_system, replication_send_system,
};
pub use server::GameServer;
pub use tick_rate::{NetworkTickAccumulator, TickRateConfig};
pub use transport::{MAX_PACKET_SIZE, PacketHeader, ReliableTransport, UdpTransport};
#[cfg(feature = "websocket")]
pub use ws_transport::{WsError, WsEvent, WsServer, WsTransport};
