use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use euca_net::{EntityState, FieldId, NetworkId, PacketHeader, ReplicationState};

// ---------------------------------------------------------------------------
// Deterministic pseudo-random number generator (avoids adding `rand` dep)
// ---------------------------------------------------------------------------

struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        let t = (self.next_u64() & 0xFFFF_FFFF) as f32 / u32::MAX as f32;
        min + t * (max - min)
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }
}

// ---------------------------------------------------------------------------
// Data generation helpers
// ---------------------------------------------------------------------------

/// Create an EntityState with deterministic game-like data.
fn make_entity_state(rng: &mut SimpleRng, id: u64) -> EntityState {
    EntityState {
        network_id: NetworkId(id),
        position: [
            rng.range_f32(-1000.0, 1000.0),
            rng.range_f32(0.0, 500.0),
            rng.range_f32(-1000.0, 1000.0),
        ],
        rotation: {
            // Generate a plausible unit quaternion
            let x = rng.range_f32(-1.0, 1.0);
            let y = rng.range_f32(-1.0, 1.0);
            let z = rng.range_f32(-1.0, 1.0);
            let w = rng.range_f32(-1.0, 1.0);
            let len = (x * x + y * y + z * z + w * w).sqrt();
            [x / len, y / len, z / len, w / len]
        },
        scale: [
            rng.range_f32(0.5, 2.0),
            rng.range_f32(0.5, 2.0),
            rng.range_f32(0.5, 2.0),
        ],
    }
}

/// Generate a pair of snapshots (old, new) where ~10% of bytes differ.
fn make_snapshot_pair(rng: &mut SimpleRng, size: usize) -> (Vec<u8>, Vec<u8>) {
    let old: Vec<u8> = (0..size).map(|_| rng.next_u8()).collect();
    let mut new = old.clone();
    // Mutate ~10% of bytes
    let mutations = size / 10;
    for _ in 0..mutations {
        let idx = (rng.next_u64() as usize) % size;
        new[idx] = new[idx].wrapping_add(1);
    }
    (old, new)
}

// ---------------------------------------------------------------------------
// A. Serialization throughput
// ---------------------------------------------------------------------------

fn bench_serialize_entity_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("bincode_serialize_entity_state");

    for n in [10, 100, 1_000] {
        let mut rng = SimpleRng::new(42);
        let states: Vec<EntityState> = (0..n).map(|i| make_entity_state(&mut rng, i)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &states, |b, states| {
            b.iter(|| {
                black_box(bincode::serialize(states).unwrap());
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// B. Deserialization throughput
// ---------------------------------------------------------------------------

fn bench_deserialize_entity_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("bincode_deserialize_entity_state");

    for n in [10, 100, 1_000] {
        let mut rng = SimpleRng::new(42);
        let states: Vec<EntityState> = (0..n).map(|i| make_entity_state(&mut rng, i)).collect();
        let bytes = bincode::serialize(&states).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), &bytes, |b, bytes| {
            b.iter(|| {
                black_box(bincode::deserialize::<Vec<EntityState>>(bytes).unwrap());
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// C. Delta field comparison cost
// ---------------------------------------------------------------------------

fn bench_delta_field_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_field_comparison");

    for n in [100, 1_000] {
        let mut rng = SimpleRng::new(99);

        // Build a ReplicationState with N fields of 64 bytes each, then compare
        // against N "current" snapshots where ~10% of bytes differ.
        // Uses FieldId-indexed access (integer array indexing, no string hashing).
        let mut state = ReplicationState::new();
        let mut current_fields: Vec<(FieldId, Vec<u8>)> = Vec::with_capacity(n);

        for i in 0..n {
            let field_id = FieldId(i as u32);
            let (old, new) = make_snapshot_pair(&mut rng, 64);
            state.update_field(field_id, old, 1);
            current_fields.push((field_id, new));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(state, current_fields),
            |b, (state, fields)| {
                b.iter(|| {
                    let mut changed_count = 0u32;
                    for &(field_id, ref data) in fields {
                        if state.field_changed(field_id, data) {
                            changed_count += 1;
                        }
                    }
                    black_box(changed_count)
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// D. PacketHeader roundtrip (manual write + read)
// ---------------------------------------------------------------------------

fn bench_packet_header_roundtrip(c: &mut Criterion) {
    let header = PacketHeader {
        sequence: 48_271,
        ack: 48_270,
        ack_bits: 0xFFFF_FFFE,
    };

    c.bench_function("packet_header_roundtrip", |b| {
        b.iter(|| {
            let mut buf = [0u8; PacketHeader::SIZE];
            let h = black_box(&header);
            h.write(&mut buf);
            black_box(PacketHeader::read(&buf))
        });
    });
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_serialize_entity_state,
    bench_deserialize_entity_state,
    bench_delta_field_comparison,
    bench_packet_header_roundtrip,
);
criterion_main!(benches);
