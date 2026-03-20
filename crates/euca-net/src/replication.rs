//! Network property replication and RPCs.
//!
//! Provides trait-based component serialization, delta compression via change ticks,
//! priority-based bandwidth allocation, server-authoritative state management,
//! and a bidirectional RPC event system.

use euca_ecs::{Entity, Query, With, World};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::bandwidth::BandwidthBudget;
use crate::protocol::{NetworkId, Replicated};

// ── ReplicatedComponent trait ──

/// Trait for components that can be serialized over the network.
///
/// Each replicated component type provides its own serialization logic,
/// allowing custom encoding (e.g., quantized floats for positions).
pub trait ReplicatedComponent: Send + Sync + 'static {
    /// Unique type name used to identify this component across the network.
    /// Must be stable across builds (do not use `std::any::type_name`).
    fn type_name(&self) -> &'static str;

    /// Serialize the component into a byte buffer.
    fn net_serialize(&self) -> Vec<u8>;

    /// Deserialize from a byte buffer, updating self in place.
    /// Returns `true` on success, `false` if the data was malformed.
    fn net_deserialize(&mut self, data: &[u8]) -> bool;
}

// ── ReplicationState component ──

/// Per-entity component tracking the last-sent field values for delta compression.
///
/// Stores a snapshot of each field's serialized bytes alongside the world tick
/// at which the snapshot was taken. The replication collect system compares
/// current field values against this state to detect changes.
#[derive(Clone, Debug, Default)]
pub struct ReplicationState {
    /// Last-sent serialized values keyed by field name.
    pub fields: HashMap<String, Vec<u8>>,
    /// World tick at which this state was last updated.
    pub change_tick: u32,
}

impl ReplicationState {
    /// Create an empty replication state.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            change_tick: 0,
        }
    }

    /// Update a field's last-sent value and bump the change tick.
    pub fn update_field(&mut self, name: impl Into<String>, data: Vec<u8>, tick: u32) {
        self.fields.insert(name.into(), data);
        self.change_tick = tick;
    }

    /// Check if a field's current data differs from the last-sent snapshot.
    pub fn field_changed(&self, name: &str, current_data: &[u8]) -> bool {
        match self.fields.get(name) {
            Some(last_sent) => last_sent.as_slice() != current_data,
            None => true, // Never sent before — treat as changed
        }
    }
}

// ── Field-level replication data ──

/// A single replicated field: the smallest unit of replication data.
///
/// Represents one named field within a component, carrying its serialized
/// value and the tick at which it last changed. Used for fine-grained
/// delta compression — only fields that actually changed are transmitted.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicatedField {
    /// Stable field name (e.g., "health", "position.x").
    pub name: String,
    /// Serialized field value.
    pub data: Vec<u8>,
    /// World tick at which this field last changed.
    pub change_tick: u32,
}

impl ReplicatedField {
    /// Create a new replicated field snapshot.
    pub fn new(name: impl Into<String>, data: Vec<u8>, change_tick: u32) -> Self {
        Self {
            name: name.into(),
            data,
            change_tick,
        }
    }

    /// Returns true if this field changed after `since_tick`.
    pub fn changed_since(&self, since_tick: u32) -> bool {
        self.change_tick > since_tick
    }
}

// ── Replication priority ──

/// Component that controls how urgently an entity is replicated.
///
/// Lower numeric values indicate higher urgency (0 = highest priority).
/// Interacts with the bandwidth budget system to determine which entities
/// are sent each tick.
#[derive(Clone, Debug)]
pub struct ReplicationPriority {
    /// Priority level. 0 = highest priority, 255 = lowest.
    pub priority: u8,
}

impl ReplicationPriority {
    /// Create a new priority with the given level (0 = highest).
    pub fn new(priority: u8) -> Self {
        Self { priority }
    }
}

impl Default for ReplicationPriority {
    fn default() -> Self {
        Self::new(128)
    }
}

// ── RPC events ──

/// Remote procedure call from client to server.
///
/// Sent as an ECS event. The server reads these each tick to process
/// client-initiated actions (e.g., "use item", "cast spell").
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerRpc {
    /// RPC function name (e.g., "use_item", "cast_spell").
    pub name: String,
    /// The target entity for this RPC.
    pub entity: Entity,
    /// Serialized arguments.
    pub payload: Vec<u8>,
}

/// Remote procedure call from server to client.
///
/// Sent as an ECS event. The client reads these to handle server-initiated
/// notifications (e.g., "show damage number", "play effect").
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientRpc {
    /// RPC function name.
    pub name: String,
    /// The target entity for this RPC.
    pub entity: Entity,
    /// Serialized arguments.
    pub payload: Vec<u8>,
}

// ── PendingReplication resource ──

/// A single entity's pending replication update, containing the changed fields.
#[derive(Clone, Debug)]
pub struct ReplicationUpdate {
    /// The entity that has changed fields.
    pub entity: Entity,
    /// The changed field names paired with their current serialized values.
    pub fields: Vec<(String, Vec<u8>)>,
}

/// Resource holding pending replication data collected by the collect system.
///
/// Populated by `replication_collect_system` and consumed by the network
/// send system. Cleared each tick after sending.
#[derive(Clone, Debug, Default)]
pub struct PendingReplication {
    /// Updates to be sent this tick, one per entity with changes.
    pub updates: Vec<ReplicationUpdate>,
}

impl PendingReplication {
    pub fn new() -> Self {
        Self {
            updates: Vec::new(),
        }
    }

    /// Take all pending updates, leaving the resource empty.
    pub fn drain(&mut self) -> Vec<ReplicationUpdate> {
        std::mem::take(&mut self.updates)
    }
}

// ── Serialized component data ──

/// A single component's serialized data for network transmission.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentData {
    /// Stable type name from `ReplicatedComponent::type_name()`.
    pub type_name: String,
    /// Serialized component bytes.
    pub data: Vec<u8>,
}

/// Complete replication payload for one entity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityReplicationData {
    pub network_id: NetworkId,
    /// Components that changed since last send.
    pub components: Vec<ComponentData>,
}

// ── Replication state tracking (server-side) ──

/// Per-client replication state: tracks what was last sent to each client.
#[derive(Clone, Debug, Default)]
pub struct ClientReplicationState {
    /// The world tick at which each entity was last replicated to this client.
    pub last_sent_tick: HashMap<NetworkId, u32>,
}

/// Server-side replication manager resource.
///
/// Tracks per-client replication state and provides the authoritative
/// source of truth. Clients never modify game state directly --- they
/// send RPCs, and the server validates and applies them.
pub struct ReplicationManager {
    /// Per-client tracking, keyed by client ID.
    pub clients: HashMap<u32, ClientReplicationState>,
    /// Outgoing replication data per client, populated by the send system.
    pub outgoing: HashMap<u32, Vec<EntityReplicationData>>,
    /// Outgoing RPCs to send to specific clients.
    pub outgoing_rpcs: HashMap<u32, Vec<ClientRpc>>,
    /// Incoming RPCs from clients, to be validated and processed.
    pub incoming_rpcs: Vec<(u32, ServerRpc)>,
}

impl ReplicationManager {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            outgoing: HashMap::new(),
            outgoing_rpcs: HashMap::new(),
            incoming_rpcs: Vec::new(),
        }
    }

    /// Register a new client for replication tracking.
    pub fn add_client(&mut self, client_id: u32) {
        self.clients
            .insert(client_id, ClientReplicationState::default());
        self.outgoing.insert(client_id, Vec::new());
        self.outgoing_rpcs.insert(client_id, Vec::new());
    }

    /// Remove a client.
    pub fn remove_client(&mut self, client_id: u32) {
        self.clients.remove(&client_id);
        self.outgoing.remove(&client_id);
        self.outgoing_rpcs.remove(&client_id);
    }

    /// Get the tick at which an entity was last sent to a client.
    pub fn last_sent_tick(&self, client_id: u32, network_id: NetworkId) -> Option<u32> {
        self.clients
            .get(&client_id)
            .and_then(|state| state.last_sent_tick.get(&network_id).copied())
    }

    /// Record that an entity was replicated to a client at the given tick.
    pub fn mark_sent(&mut self, client_id: u32, network_id: NetworkId, tick: u32) {
        if let Some(state) = self.clients.get_mut(&client_id) {
            state.last_sent_tick.insert(network_id, tick);
        }
    }

    /// Queue an RPC to send to a client.
    pub fn send_rpc(&mut self, client_id: u32, rpc: ClientRpc) {
        if let Some(rpcs) = self.outgoing_rpcs.get_mut(&client_id) {
            rpcs.push(rpc);
        }
    }

    /// Push an incoming server RPC from a client.
    pub fn receive_rpc(&mut self, client_id: u32, rpc: ServerRpc) {
        self.incoming_rpcs.push((client_id, rpc));
    }

    /// Drain outgoing replication data for a client.
    pub fn drain_outgoing(&mut self, client_id: u32) -> Vec<EntityReplicationData> {
        self.outgoing
            .get_mut(&client_id)
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Drain outgoing RPCs for a client.
    pub fn drain_outgoing_rpcs(&mut self, client_id: u32) -> Vec<ClientRpc> {
        self.outgoing_rpcs
            .get_mut(&client_id)
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Drain incoming RPCs from all clients.
    pub fn drain_incoming_rpcs(&mut self) -> Vec<(u32, ServerRpc)> {
        std::mem::take(&mut self.incoming_rpcs)
    }
}

impl Default for ReplicationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Component registry ──

/// Type-erased serialization function for a registered replicated component type.
type SerializeFn = Box<dyn Fn(&World, Entity) -> Option<Vec<u8>> + Send + Sync>;

/// Type-erased change detection function for a registered replicated component type.
type ChangeDetectFn = Box<dyn Fn(&World, Entity, u32) -> bool + Send + Sync>;

/// Registration entry for a replicated component type.
struct ReplicatedComponentEntry {
    type_name: String,
    serialize_fn: SerializeFn,
    /// Returns true if the component changed since `since_tick`.
    change_detect_fn: ChangeDetectFn,
}

/// Registry of component types that participate in replication.
///
/// Stores type-erased serialization functions keyed by stable type names.
/// This allows the replication system to serialize any registered component
/// without knowing its concrete type at compile time.
pub struct ComponentReplicationRegistry {
    entries: Vec<ReplicatedComponentEntry>,
    /// Estimated byte size per entity for bandwidth budgeting.
    pub estimated_bytes_per_entity: u32,
}

impl ComponentReplicationRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            estimated_bytes_per_entity: 128,
        }
    }

    /// Register a component type for replication.
    ///
    /// The component must implement `ReplicatedComponent` and be stored in the ECS.
    pub fn register<T: ReplicatedComponent + euca_ecs::Component>(&mut self, type_name: &str) {
        let name = type_name.to_string();
        let serialize_fn: SerializeFn = Box::new(|world: &World, entity: Entity| {
            world.get::<T>(entity).map(|c| c.net_serialize())
        });
        let change_fn: ChangeDetectFn =
            Box::new(|world: &World, entity: Entity, since_tick: u32| {
                world
                    .get_change_tick::<T>(entity)
                    .is_some_and(|tick| tick > since_tick)
            });
        self.entries.push(ReplicatedComponentEntry {
            type_name: name,
            serialize_fn,
            change_detect_fn: change_fn,
        });
    }

    /// Serialize all changed components for an entity since `since_tick`.
    /// Returns `None` if nothing changed.
    pub fn serialize_changed(
        &self,
        world: &World,
        entity: Entity,
        since_tick: u32,
    ) -> Option<Vec<ComponentData>> {
        let mut components = Vec::new();
        for entry in &self.entries {
            if (entry.change_detect_fn)(world, entity, since_tick)
                && let Some(data) = (entry.serialize_fn)(world, entity)
            {
                components.push(ComponentData {
                    type_name: entry.type_name.clone(),
                    data,
                });
            }
        }
        if components.is_empty() {
            None
        } else {
            Some(components)
        }
    }

    /// Serialize all components for an entity (full snapshot, ignoring change ticks).
    pub fn serialize_all(&self, world: &World, entity: Entity) -> Vec<ComponentData> {
        let mut components = Vec::new();
        for entry in &self.entries {
            if let Some(data) = (entry.serialize_fn)(world, entity) {
                components.push(ComponentData {
                    type_name: entry.type_name.clone(),
                    data,
                });
            }
        }
        components
    }

    /// Number of registered component types.
    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for ComponentReplicationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Replication systems ──

/// Collect system: for entities with `Replicated`, compare current serialized
/// state against `ReplicationState`, and populate `PendingReplication` with
/// any changed fields.
///
/// This is the first stage of the replication pipeline. Downstream systems
/// consume `PendingReplication` to send data over the network.
pub fn replication_collect_system(world: &mut World) {
    let registry_exists = world.resource::<ComponentReplicationRegistry>().is_some();
    if !registry_exists {
        return;
    }

    // Gather replicated entities
    let replicated_entities: Vec<Entity> = {
        let query = Query::<Entity, With<Replicated>>::new(world);
        query.iter().collect()
    };

    let current_tick = world.current_tick() as u32;
    let mut updates: Vec<ReplicationUpdate> = Vec::new();

    for entity in replicated_entities {
        // Serialize all registered component fields for this entity
        let serialized: Vec<(String, Vec<u8>)> = {
            let registry = match world.resource::<ComponentReplicationRegistry>() {
                Some(r) => r,
                None => continue,
            };
            registry
                .serialize_all(world, entity)
                .into_iter()
                .map(|cd| (cd.type_name, cd.data))
                .collect()
        };

        if serialized.is_empty() {
            continue;
        }

        // Compare against last-sent state
        let changed_fields: Vec<(String, Vec<u8>)> = {
            let state = world.get::<ReplicationState>(entity);
            serialized
                .into_iter()
                .filter(|(name, data)| match &state {
                    Some(s) => s.field_changed(name, data),
                    None => true, // No prior state — everything is new
                })
                .collect()
        };

        if changed_fields.is_empty() {
            continue;
        }

        // Update the entity's ReplicationState with new values
        if let Some(state) = world.get_mut::<ReplicationState>(entity) {
            for (name, data) in &changed_fields {
                state.update_field(name.clone(), data.clone(), current_tick);
            }
        }

        updates.push(ReplicationUpdate {
            entity,
            fields: changed_fields,
        });
    }

    // Write to PendingReplication resource
    if let Some(pending) = world.resource_mut::<PendingReplication>() {
        pending.updates = updates;
    } else {
        world.insert_resource(PendingReplication { updates });
    }
}

/// Server-side system: gather changed components for all replicated entities
/// and populate the `ReplicationManager` outgoing buffers.
///
/// Uses delta compression: only components whose change tick exceeds the
/// last-sent tick for that client are serialized. Respects bandwidth budgets
/// and replication priority.
pub fn replication_send_system(world: &mut World) {
    // Collect replicated entities and their priorities
    let replicated_entities: Vec<(Entity, NetworkId, u8)> = {
        let query = Query::<(Entity, &NetworkId), With<Replicated>>::new(world);
        query
            .iter()
            .map(|(entity, net_id)| {
                let priority = world
                    .get::<ReplicationPriority>(entity)
                    .map(|p| p.priority)
                    .unwrap_or(128);
                (entity, *net_id, priority)
            })
            .collect()
    };

    // Sort by priority (lower value = higher urgency, so sort ascending)
    let mut sorted_entities = replicated_entities;
    sorted_entities.sort_by_key(|&(_, _, priority)| priority);

    // Get resources we need
    let current_tick = world.current_tick() as u32;

    let registry_exists = world.resource::<ComponentReplicationRegistry>().is_some();
    let manager_exists = world.resource::<ReplicationManager>().is_some();

    if !registry_exists || !manager_exists {
        return;
    }

    // Collect client IDs
    let client_ids: Vec<u32> = world
        .resource::<ReplicationManager>()
        .map(|m| m.clients.keys().copied().collect())
        .unwrap_or_default();

    // For each client, determine what to send
    for client_id in &client_ids {
        let mut budget = world
            .resource::<BandwidthBudget>()
            .cloned()
            .unwrap_or_default();
        let estimated_bytes = world
            .resource::<ComponentReplicationRegistry>()
            .map(|r| r.estimated_bytes_per_entity)
            .unwrap_or(128);

        let mut replication_data: Vec<EntityReplicationData> = Vec::new();
        let mut sent_entities: Vec<(NetworkId, u32)> = Vec::new();

        for &(entity, network_id, _priority) in &sorted_entities {
            if !budget.try_allocate(estimated_bytes) {
                break;
            }

            let since_tick = world
                .resource::<ReplicationManager>()
                .and_then(|m| m.last_sent_tick(*client_id, network_id))
                .unwrap_or(0);

            let changed_components = world
                .resource::<ComponentReplicationRegistry>()
                .and_then(|r| r.serialize_changed(world, entity, since_tick));

            if let Some(components) = changed_components {
                replication_data.push(EntityReplicationData {
                    network_id,
                    components,
                });
                sent_entities.push((network_id, current_tick));
            }
        }

        // Update the manager with results
        if let Some(manager) = world.resource_mut::<ReplicationManager>() {
            for (network_id, tick) in sent_entities {
                manager.mark_sent(*client_id, network_id, tick);
            }
            if let Some(outgoing) = manager.outgoing.get_mut(client_id) {
                outgoing.extend(replication_data);
            }
        }
    }
}

/// Client-side system: process incoming replication data.
///
/// Reads `EntityReplicationData` from a `ClientReplicationReceiver` resource
/// and applies component updates to the local world. The server is authoritative:
/// client state is overwritten with whatever the server sends.
pub fn replication_receive_system(world: &mut World) {
    let incoming = {
        let receiver = match world.resource_mut::<ClientReplicationReceiver>() {
            Some(r) => r,
            None => return,
        };
        std::mem::take(&mut receiver.incoming)
    };

    let rpc_incoming = {
        let receiver = match world.resource_mut::<ClientReplicationReceiver>() {
            Some(r) => r,
            None => return,
        };
        std::mem::take(&mut receiver.incoming_rpcs)
    };

    // Apply component data to entities
    if let Some(applier) = world.resource::<ComponentDeserializationRegistry>() {
        // Need to clone the applier's function pointers since we can't hold
        // an immutable borrow on world while mutating entities.
        let apply_fns: Vec<(String, _)> = applier
            .entries
            .iter()
            .map(|e| {
                (
                    e.type_name.clone(),
                    &e.deserialize_fn as *const DeserializeFn,
                )
            })
            .collect();

        // Build a NetworkId -> Entity lookup from the world
        let entity_map: HashMap<NetworkId, Entity> = {
            let query = Query::<(Entity, &NetworkId)>::new(world);
            query.iter().map(|(e, nid)| (*nid, e)).collect()
        };

        for repl_data in &incoming {
            if let Some(&entity) = entity_map.get(&repl_data.network_id) {
                for comp_data in &repl_data.components {
                    for (type_name, fn_ptr) in &apply_fns {
                        if type_name == &comp_data.type_name {
                            // SAFETY: The function pointer is valid for the lifetime of the
                            // ComponentDeserializationRegistry resource, which we verified exists.
                            let apply_fn = unsafe { &**fn_ptr };
                            apply_fn(world, entity, &comp_data.data);
                            break;
                        }
                    }
                }
            }
        }
    }

    // Emit incoming RPCs as events
    for rpc in rpc_incoming {
        world.send_event(rpc);
    }
}

/// Type-erased deserialization function.
type DeserializeFn = Box<dyn Fn(&mut World, Entity, &[u8]) + Send + Sync>;

/// Registration entry for deserializing a replicated component.
struct DeserializationEntry {
    type_name: String,
    deserialize_fn: DeserializeFn,
}

/// Client-side registry for deserializing incoming component data.
pub struct ComponentDeserializationRegistry {
    entries: Vec<DeserializationEntry>,
}

impl ComponentDeserializationRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a component type for deserialization on the client.
    pub fn register<T: ReplicatedComponent + euca_ecs::Component + Default>(
        &mut self,
        type_name: &str,
    ) {
        let name = type_name.to_string();
        let deserialize_fn: DeserializeFn =
            Box::new(move |world: &mut World, entity: Entity, data: &[u8]| {
                if let Some(component) = world.get_mut::<T>(entity) {
                    component.net_deserialize(data);
                }
            });
        self.entries.push(DeserializationEntry {
            type_name: name,
            deserialize_fn,
        });
    }
}

impl Default for ComponentDeserializationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Client-side resource: holds incoming replication data to be applied.
pub struct ClientReplicationReceiver {
    /// Incoming entity replication data from the server.
    pub incoming: Vec<EntityReplicationData>,
    /// Incoming RPCs from the server.
    pub incoming_rpcs: Vec<ClientRpc>,
}

impl ClientReplicationReceiver {
    pub fn new() -> Self {
        Self {
            incoming: Vec::new(),
            incoming_rpcs: Vec::new(),
        }
    }

    /// Push incoming replication data (called by the transport layer).
    pub fn push_replication_data(&mut self, data: EntityReplicationData) {
        self.incoming.push(data);
    }

    /// Push an incoming RPC (called by the transport layer).
    pub fn push_rpc(&mut self, rpc: ClientRpc) {
        self.incoming_rpcs.push(rpc);
    }
}

impl Default for ClientReplicationReceiver {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use euca_ecs::World;

    #[derive(Clone, Debug, Default, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
        z: f32,
    }

    impl ReplicatedComponent for Position {
        fn type_name(&self) -> &'static str {
            "Position"
        }
        fn net_serialize(&self) -> Vec<u8> {
            let mut b = Vec::with_capacity(12);
            b.extend_from_slice(&self.x.to_le_bytes());
            b.extend_from_slice(&self.y.to_le_bytes());
            b.extend_from_slice(&self.z.to_le_bytes());
            b
        }
        fn net_deserialize(&mut self, d: &[u8]) -> bool {
            if d.len() < 12 {
                return false;
            }
            self.x = f32::from_le_bytes([d[0], d[1], d[2], d[3]]);
            self.y = f32::from_le_bytes([d[4], d[5], d[6], d[7]]);
            self.z = f32::from_le_bytes([d[8], d[9], d[10], d[11]]);
            true
        }
    }

    #[derive(Clone, Debug, Default, PartialEq)]
    struct Health {
        current: f32,
        max: f32,
    }

    impl ReplicatedComponent for Health {
        fn type_name(&self) -> &'static str {
            "Health"
        }
        fn net_serialize(&self) -> Vec<u8> {
            let mut b = Vec::with_capacity(8);
            b.extend_from_slice(&self.current.to_le_bytes());
            b.extend_from_slice(&self.max.to_le_bytes());
            b
        }
        fn net_deserialize(&mut self, d: &[u8]) -> bool {
            if d.len() < 8 {
                return false;
            }
            self.current = f32::from_le_bytes([d[0], d[1], d[2], d[3]]);
            self.max = f32::from_le_bytes([d[4], d[5], d[6], d[7]]);
            true
        }
    }

    #[test]
    fn replicated_component_serialize_roundtrip() {
        let pos = Position {
            x: 1.5,
            y: -3.0,
            z: 42.0,
        };
        let bytes = pos.net_serialize();
        assert_eq!(bytes.len(), 12);
        let mut decoded = Position::default();
        assert!(decoded.net_deserialize(&bytes));
        assert_eq!(decoded, pos);
    }

    #[test]
    fn replicated_component_deserialize_rejects_short_data() {
        let mut pos = Position::default();
        assert!(!pos.net_deserialize(&[0, 1, 2]));
    }

    #[test]
    fn replication_state_tracks_fields() {
        let mut state = ReplicationState::new();
        assert!(state.fields.is_empty());
        assert!(state.field_changed("health", &[0, 0, 200, 66]));
        state.update_field("health", vec![0, 0, 200, 66], 5);
        assert!(!state.field_changed("health", &[0, 0, 200, 66]));
        assert!(state.field_changed("health", &[0, 0, 0, 67]));
        assert!(state.field_changed("mana", &[1, 2, 3, 4]));
    }

    #[test]
    fn replication_priority_ordering() {
        let high = ReplicationPriority::new(0);
        let low = ReplicationPriority::new(255);
        assert!(high.priority < low.priority);
    }

    #[test]
    fn rpc_events_carry_entity() {
        let entity = Entity::from_raw(42, 1);
        let rpc = ServerRpc {
            name: "cast".to_string(),
            entity,
            payload: vec![1, 2],
        };
        assert_eq!(rpc.entity, entity);
        let crpc = ClientRpc {
            name: "fx".to_string(),
            entity,
            payload: vec![3],
        };
        assert_eq!(crpc.entity, entity);
    }

    #[test]
    fn rpc_serialize_roundtrip() {
        let entity = Entity::from_raw(7, 0);
        let rpc = ServerRpc {
            name: "use_item".to_string(),
            entity,
            payload: vec![1, 2, 3],
        };
        let bytes = bincode::serialize(&rpc).unwrap();
        let decoded: ServerRpc = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.name, "use_item");
        assert_eq!(decoded.entity, entity);
    }

    #[test]
    fn pending_replication_drain() {
        let entity = Entity::from_raw(1, 0);
        let mut pending = PendingReplication::new();
        pending.updates.push(ReplicationUpdate {
            entity,
            fields: vec![("Pos".into(), vec![0; 12])],
        });
        let drained = pending.drain();
        assert_eq!(drained.len(), 1);
        assert!(pending.updates.is_empty());
    }

    #[test]
    fn replication_collect_system_detects_changes() {
        let mut world = World::new();
        let mut registry = ComponentReplicationRegistry::new();
        registry.register::<Position>("Position");
        world.insert_resource(registry);
        world.insert_resource(PendingReplication::new());
        let entity = world.spawn(Position {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        });
        world.insert(entity, Replicated);
        world.insert(entity, ReplicationState::new());
        replication_collect_system(&mut world);
        assert_eq!(
            world
                .resource::<PendingReplication>()
                .unwrap()
                .updates
                .len(),
            1
        );
        replication_collect_system(&mut world);
        assert!(
            world
                .resource::<PendingReplication>()
                .unwrap()
                .updates
                .is_empty()
        );
        world.get_mut::<Position>(entity).unwrap().x = 99.0;
        replication_collect_system(&mut world);
        assert_eq!(
            world
                .resource::<PendingReplication>()
                .unwrap()
                .updates
                .len(),
            1
        );
    }

    #[test]
    fn replication_collect_system_multiple_entities() {
        let mut world = World::new();
        let mut registry = ComponentReplicationRegistry::new();
        registry.register::<Position>("Position");
        registry.register::<Health>("Health");
        world.insert_resource(registry);
        world.insert_resource(PendingReplication::new());
        let e1 = world.spawn(Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        });
        world.insert(e1, Replicated);
        world.insert(e1, ReplicationState::new());
        let e2 = world.spawn(Position {
            x: 2.0,
            y: 0.0,
            z: 0.0,
        });
        world.insert(e2, Replicated);
        world.insert(
            e2,
            Health {
                current: 100.0,
                max: 100.0,
            },
        );
        world.insert(e2, ReplicationState::new());
        replication_collect_system(&mut world);
        let pending = world.resource::<PendingReplication>().unwrap();
        assert_eq!(pending.updates.len(), 2);
        assert_eq!(
            pending
                .updates
                .iter()
                .find(|u| u.entity == e1)
                .unwrap()
                .fields
                .len(),
            1
        );
        assert_eq!(
            pending
                .updates
                .iter()
                .find(|u| u.entity == e2)
                .unwrap()
                .fields
                .len(),
            2
        );
    }

    #[test]
    fn replication_manager_client_lifecycle() {
        let mut manager = ReplicationManager::new();
        manager.add_client(1);
        manager.mark_sent(1, NetworkId(42), 10);
        assert_eq!(manager.last_sent_tick(1, NetworkId(42)), Some(10));
        manager.remove_client(1);
        assert_eq!(manager.last_sent_tick(1, NetworkId(42)), None);
    }

    #[test]
    fn replication_manager_rpc_flow() {
        let mut manager = ReplicationManager::new();
        let entity = Entity::from_raw(1, 0);
        manager.add_client(1);
        manager.send_rpc(
            1,
            ClientRpc {
                name: "effect".into(),
                entity,
                payload: vec![5],
            },
        );
        assert_eq!(manager.drain_outgoing_rpcs(1).len(), 1);
        assert!(manager.drain_outgoing_rpcs(1).is_empty());
        manager.receive_rpc(
            1,
            ServerRpc {
                name: "use_item".into(),
                entity,
                payload: vec![42],
            },
        );
        let incoming = manager.drain_incoming_rpcs();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].1.name, "use_item");
    }

    #[test]
    fn component_registry_serialize_all() {
        let mut registry = ComponentReplicationRegistry::new();
        registry.register::<Position>("Position");
        registry.register::<Health>("Health");
        let mut world = World::new();
        let entity = world.spawn(Position {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        });
        world.insert(
            entity,
            Health {
                current: 50.0,
                max: 100.0,
            },
        );
        let all = registry.serialize_all(&world, entity);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn replicated_field_change_detection() {
        let field = ReplicatedField::new("health", vec![0, 0, 200, 66], 5);
        assert!(field.changed_since(3));
        assert!(!field.changed_since(5));
        assert!(!field.changed_since(10));
    }

    #[test]
    fn replicated_field_serialization_roundtrip() {
        let field = ReplicatedField::new("velocity", vec![1, 2, 3, 4, 5, 6, 7, 8], 42);
        let bytes = bincode::serialize(&field).unwrap();
        let decoded: ReplicatedField = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.name, "velocity");
        assert_eq!(decoded.data, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(decoded.change_tick, 42);
    }
}
