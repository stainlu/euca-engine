---
name: eucaengine
description: ECS-first, agent-native game engine in Rust
version: 0.1.0
auth: nit
protocol: cli
---

# EucaEngine — Agent Interface

EucaEngine is a game engine you control via the `euca` CLI. You interact with a live, visual editor — create entities, move them, run physics, take screenshots to verify your work.

## Quick Start

```bash
# 1. Start the engine (opens a window)
cargo run -p euca-editor --example editor

# 2. Check it's running
euca status

# 3. See what's in the scene
euca entity list

# 4. Move an entity
euca entity update 1 --position 3,2,0

# 5. Take a screenshot to verify
euca screenshot
# Returns a PNG path — read it to see the scene
```

## Authentication

EucaEngine uses [nit](https://github.com/newtype-ai/nit) for agent identity.

```bash
npm install -g @newtype-ai/nit
nit init && nit push
euca auth login
euca auth status
```

## Command Reference

### Entity (CRUD)

```bash
# List all entities
euca entity list

# Get a single entity
euca entity get <id>

# Create an entity
euca entity create --position 1,2,3
euca entity create --position 0,5,0 --physics Dynamic --collider aabb:0.5,0.5,0.5
euca entity create --json '{"position": [1,2,3], "physics_body": "Dynamic"}'

# Preview without creating
euca entity create --position 1,2,3 --dry-run

# Update an entity
euca entity update <id> --position 3,0,0
euca entity update <id> --velocity 0,5,0
euca entity update <id> --json '{"transform": {"position": [1,2,3]}}'

# Delete an entity
euca entity delete <id>
euca entity delete --all
```

### Simulation

```bash
euca sim play              # Start physics
euca sim pause             # Pause
euca sim step --ticks 10   # Advance 10 ticks
euca sim reset             # Reset to initial scene
```

### Screenshot

```bash
euca screenshot                     # Save to temp file, print path
euca screenshot --output scene.png  # Save to specific path
```

### Camera Control

```bash
# View presets (orthographic — no perspective distortion)
euca camera view top          # Bird's-eye view
euca camera view front        # Front orthographic
euca camera view right        # Right side orthographic
euca camera view left         # Left side
euca camera view back         # Back
euca camera view perspective  # Reset to default 3D view

# Focus on entity (centers + frames)
euca camera focus <entity_id>

# Manual positioning
euca camera set --eye 10,5,10 --target 0,0,0
euca camera set --fov 60

# Query current state
euca camera get
```

### Scene Persistence

```bash
euca scene save my_scene.json    # Save current world state
euca scene load my_scene.json    # Load (replaces current scene)
```

### Status & Schema

```bash
euca status          # Engine info: version, entity count, tick
euca schema          # All component types and their fields
```

### Auth

```bash
euca auth login      # Login with nit identity
euca auth status     # Check authentication
```

## Available Colors

Named: `red`, `blue`, `green`, `gold`, `silver`, `gray`, `white`, `black`, `yellow`, `cyan`, `magenta`, `orange`

RGB: any `r,g,b` value (0.0-1.0) maps to nearest preset.

## Available Meshes

`cube`, `sphere`, `plane`

## Components

| Component | Fields | Notes |
|-----------|--------|-------|
| LocalTransform | position [f32;3], rotation [f32;4], scale [f32;3] | Entity's local transform |
| GlobalTransform | (same) | Read-only, computed from hierarchy |
| Velocity | linear [f32;3], angular [f32;3] | Requires PhysicsBody |
| PhysicsBody | body_type: Dynamic / Static / Kinematic | |
| Collider | Aabb{hx,hy,hz}, Sphere{radius}, Capsule{radius,half_height} | |

## Flag Reference

| Flag | Format | Used in |
|------|--------|---------|
| `--position` | `x,y,z` | create, update |
| `--scale` | `x,y,z` | create, update |
| `--velocity` | `x,y,z` | update |
| `--physics` | `Dynamic\|Static\|Kinematic` | create, update |
| `--collider` | `aabb:h,h,h` or `sphere:r` or `capsule:r,h` | create, update |
| `--json` | JSON string | create, update (overrides other flags) |
| `--dry-run` | (flag) | create, update |
| `--output` | file path | screenshot |
| `--eye` | `x,y,z` | camera set |
| `--target` | `x,y,z` | camera set |
| `--server` | URL | global (default: http://localhost:3917) |

## Workflows

### Build a Scene

```bash
euca entity create --position 0,0,0 --physics Static --collider aabb:10,0.01,10
euca entity create --position 0,2,0 --physics Dynamic --collider aabb:0.5,0.5,0.5
euca entity create --position 2,3,0 --physics Dynamic --collider sphere:0.5
euca screenshot --output scene.png
```

### Test Physics

```bash
euca sim play
# wait...
euca sim pause
euca entity get 2
euca screenshot
```

### Iterate on Layout

```bash
euca entity update 3 --position 5,1,0
euca screenshot
euca entity update 3 --position 5,1.5,0
euca screenshot
```

## Output

All commands return JSON. Pipe through `jq` for filtering:

```bash
euca entity list | jq '.entities[] | select(.physics_body == "Dynamic") | .id'
```

## Backward Compatibility

Old commands still work as hidden aliases:
- `euca spawn` → `euca entity create`
- `euca modify` → `euca entity update`
- `euca despawn` → `euca entity delete`
- `euca observe` → `euca entity list`
- `euca play/pause/step/reset` → `euca sim play/pause/step/reset`
