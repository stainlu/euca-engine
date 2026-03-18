---
name: eucaengine
description: ECS-first, agent-native game engine in Rust
version: 0.1.0
auth: nit
protocol: cli
---

# EucaEngine — Agent Interface

EucaEngine is a game engine you control via the `euca` CLI. You interact with a live, visual editor — spawn entities, move them, run physics, take screenshots to verify your work.

## Quick Start

```bash
# 1. Start the engine (opens a window)
cargo run -p euca-editor --example editor

# 2. Check it's running
euca status

# 3. See what's in the scene
euca observe

# 4. Move an entity
euca modify 1 --transform 3,2,0

# 5. Take a screenshot to see the result
euca screenshot
# Returns a PNG path — read it to verify visually
```

## Authentication

EucaEngine uses [nit](https://github.com/newtype-ai/nit) for agent identity.

```bash
# Install nit
npm install -g @newtype-ai/nit

# Initialize identity (one-time)
nit init
nit push

# Login to the engine
euca auth login

# Check auth status
euca auth status
```

## CLI Reference

### Engine Status

```bash
euca status
# Returns: { engine, version, entity_count, archetype_count, tick }
```

### Observe World State

```bash
# All entities
euca observe

# Single entity
euca observe --entity 5

# Output is JSON with: id, generation, transform, velocity, collider, physics_body
```

### Spawn Entity

```bash
# Basic entity at position
euca spawn --position 1,2,3

# With physics
euca spawn --position 0,5,0 --physics Dynamic --collider aabb:0.5,0.5,0.5

# With scale
euca spawn --position 1,2,3 --scale 2,2,2
```

### Modify Entity

```bash
# Change position
euca modify <id> --transform x,y,z

# Change velocity
euca modify <id> --velocity 0,5,0

# Change physics body type
euca modify <id> --physics Static

# Change collider
euca modify <id> --collider sphere:1.0

# Full JSON control
euca modify <id> --json '{"transform": {"position": [1,2,3]}, "velocity": {"linear": [0,1,0], "angular": [0,0,0]}}'
```

### Despawn Entity

```bash
euca despawn <id>
euca despawn --all    # Clear everything
```

### Simulation Control

```bash
euca step 10         # Advance 10 physics ticks
euca play            # Start continuous simulation
euca pause           # Pause simulation
euca reset           # Reset to initial scene
```

### Screenshot (visual feedback)

```bash
# Capture viewport as PNG (returns temp file path)
euca screenshot

# Save to specific path
euca screenshot --output scene.png
```

Use screenshots to verify your work visually. The PNG captures the 3D viewport without UI panels.

### Schema

```bash
euca schema
# Returns available components and their fields
```

## Components Reference

| Component | Fields | Notes |
|-----------|--------|-------|
| LocalTransform | position [f32;3], rotation [f32;4], scale [f32;3] | Entity's local transform |
| GlobalTransform | (same) | Read-only. Computed from hierarchy. |
| Velocity | linear [f32;3], angular [f32;3] | Requires PhysicsBody |
| PhysicsBody | body_type: Dynamic / Static / Kinematic | |
| Collider | Aabb{hx,hy,hz}, Sphere{radius}, Capsule{radius,half_height} | |

## Common Workflows

### Build a Scene

```bash
# Spawn ground
euca spawn --position 0,0,0 --physics Static --collider aabb:10,0.01,10

# Add objects
euca spawn --position 0,2,0 --physics Dynamic --collider aabb:0.5,0.5,0.5
euca spawn --position 2,3,0 --physics Dynamic --collider sphere:0.5

# Check result
euca screenshot --output scene.png
```

### Test Physics

```bash
# Drop objects
euca play
# Wait...
euca pause

# Check where things landed
euca observe --entity 2
euca screenshot
```

### Iterate on Layout

```bash
# Move entity 3 to a new position
euca modify 3 --transform 5,1,0

# Verify
euca screenshot

# Adjust
euca modify 3 --transform 5,1.5,0
euca screenshot
```

## Server Details

- **Port**: 3917 (default)
- **Protocol**: HTTP REST (JSON)
- **Override**: `euca --server http://localhost:PORT ...`
- **All responses are JSON** (pipe through `jq` for filtering)
