# euca-net

Multiplayer networking: UDP/QUIC transport, property replication, client prediction, and bandwidth budgeting.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `UdpTransport` and `QuicTransport` with reliable packet delivery
- `GameServer` and `GameClient` for authoritative server architecture
- `ReplicationManager` with per-component registration, priority, and delta compression
- `ClientPrediction` with server reconciliation and input replay
- `BandwidthBudget` with priority-based entity selection for replication
- `InterestManager` for distance/area-based relevancy culling
- `NetworkId` and `Replicated` marker for network-synced entities
- Configurable tick rate via `TickRateConfig` and `NetworkTickAccumulator`
- RPC system (`ServerRpc`, `ClientRpc`) for one-off messages

## Usage

```rust
use euca_net::*;

let server = GameServer::new("0.0.0.0:7777").unwrap();
let client = GameClient::new("127.0.0.1:7777").unwrap();

replication_collect_system(&mut world);
replication_send_system(&mut world);
```

## License

MIT
