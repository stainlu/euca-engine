pub mod bandwidth;
mod client;
pub mod interest;
pub mod prediction;
mod protocol;
pub mod replication;
mod server;
pub mod tick_rate;
mod transport;

pub use bandwidth::{BandwidthBudget, PriorityCalculator, select_entities_for_replication};
pub use client::GameClient;
pub use interest::{InterestConfig, InterestManager, interest_culling_system};
pub use prediction::ClientPrediction;
pub use protocol::{ClientMessage, EntityState, NetworkId, Replicated, ServerMessage};
pub use replication::{
    ClientReplicationReceiver, ClientRpc, ComponentData, ComponentDeserializationRegistry,
    ComponentReplicationRegistry, EntityReplicationData, PendingReplication, ReplicatedComponent,
    ReplicatedField, ReplicationManager, ReplicationPriority, ReplicationState, ReplicationUpdate,
    ServerRpc, replication_collect_system, replication_receive_system, replication_send_system,
};
pub use server::GameServer;
pub use tick_rate::{NetworkTickAccumulator, TickRateConfig};
pub use transport::{MAX_PACKET_SIZE, PacketHeader, ReliableTransport, UdpTransport};
