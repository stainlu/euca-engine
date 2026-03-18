---
name: eucaengine
description: ECS-first, agent-native game engine in Rust
version: 0.1.0
auth: nit
protocol: cli
---

# EucaEngine — Agent Interface

EucaEngine is a game engine you control via the `euca` CLI. You interact with a live visual editor — create entities, set up game rules, run physics, control AI, add HUD elements, and take screenshots to verify your work.

## Quick Start

```bash
# 1. Start the engine (opens a window)
cargo run -p euca-editor --example editor

# 2. Check it's running
euca status

# 3. Create a game
euca game create --mode deathmatch --score-limit 5

# 4. Spawn fighters
euca entity create --mesh cube --position 0,2,0 --health 100 --team 1 --color red --physics Dynamic --collider aabb:0.5,0.5,0.5
euca entity create --mesh sphere --position 3,2,0 --health 100 --team 2 --color blue --physics Dynamic --collider sphere:0.5

# 5. Add HUD
euca ui text "DEATHMATCH" --x 0.5 --y 0.02 --size 28 --color yellow

# 6. Run simulation
euca sim play

# 7. Check results
euca game state
euca screenshot
```

## Command Reference

### Entity (CRUD + Combat)

```bash
# List / Get
euca entity list                              # All entities as JSON
euca entity get <id>                          # Single entity (health, team, transform, etc.)

# Create
euca entity create --mesh cube --position 0,2,0 --color red
euca entity create --mesh sphere --position 3,2,0 --physics Dynamic --collider sphere:0.5
euca entity create --mesh cube --position 0,1,0 --health 100 --team 1 --color red --physics Dynamic --collider aabb:0.5,0.5,0.5
euca entity create --json '{"position": [1,2,3], "mesh": "cube", "color": "gold", "health": 50}'

# Update
euca entity update <id> --position 5,0,0
euca entity update <id> --color green
euca entity update <id> --velocity 0,10,0
euca entity update <id> --json '{"transform": {"position": [1,2,3]}}'

# Delete
euca entity delete <id>
euca entity delete --all

# Combat
euca entity damage <id> --amount 25           # Reduce health
euca entity heal <id> --amount 10             # Restore health

# Preview
euca entity create --position 1,2,3 --dry-run # Show what would be created
```

### Simulation

```bash
euca sim play                    # Start physics + gameplay systems
euca sim pause                   # Pause
euca sim step --ticks 10         # Advance N ticks
euca sim reset                   # Reset to initial scene
```

When simulation is playing, these systems run each tick:
physics → damage → death → projectiles → triggers → AI → game state → respawn

### Game Match

```bash
euca game create --mode deathmatch --score-limit 10
euca game state                  # Phase, scores, elapsed time
```

Game phases: lobby → playing → post_match (when score limit reached)

### Camera

```bash
# Preset views (orthographic — no perspective distortion)
euca camera view top              # Bird's-eye view
euca camera view front            # Front orthographic
euca camera view right            # Right side
euca camera view left             # Left side
euca camera view back             # Back
euca camera view perspective      # Reset to default 3D

# Focus on entity
euca camera focus <id>            # Center camera on entity

# Manual
euca camera set --eye 10,5,10 --target 0,0,0
euca camera set --fov 60
euca camera get                   # Current camera state
```

### Trigger Zones

```bash
euca trigger create --position 0,0,0 --zone 2,1,2 --action damage:10
euca trigger create --position 5,0,5 --zone 1,1,1 --action heal:5
```

Triggers fire when entities overlap the zone. Actions: `damage:N`, `heal:N`.

### Projectiles

```bash
euca projectile spawn --from 0,1,0 --direction 1,0,0 --speed 20 --damage 25
```

Projectiles move each tick, hit entities with Health (sphere collision r=0.5), apply DamageEvent, then despawn. Default lifetime: 3 seconds.

### AI Behavior

```bash
euca ai set <id> --behavior idle
euca ai set <id> --behavior chase --target <target_id> --speed 5
euca ai set <id> --behavior patrol --speed 3
euca ai set <id> --behavior flee --target <threat_id> --speed 4
```

AI sets entity Velocity each tick based on behavior. Requires entity to have a position (LocalTransform).

### Rules (Data-Driven Game Logic)

```bash
# When blue team dies, spawn a replacement
euca rule create --when death --filter team:2 --do-action "spawn sphere 3,3,0 blue"

# Every 10 seconds, spawn a health pickup
euca rule create --when timer:10 --do-action "spawn sphere 0,1,0 gold"

# When health drops below 25, auto-heal
euca rule create --when health-below:25 --do-action "heal this 50"

# Kill scoring
euca rule create --when death --filter team:2 --do-action "score source +1"

# Management
euca rule list
```

**Conditions:** `death`, `timer:N` (seconds), `health-below:N`, `score:N` (when any player reaches score), `phase:playing|post_match` (when game phase changes)
**Filters:** `any`, `entity:N`, `team:N`
**Actions:** `spawn <mesh> <x,y,z> [color]`, `damage <target> <amount>`, `heal <target> <amount>`, `score <target> <points>`, `despawn <target>`, `teleport <target> <x,y,z>`
**Targets:** `this` (trigger entity), `source` (e.g. killer), `entity:N`

Rules are ECS entities — they're saved/loaded with scenes and compose with all other systems.

### HUD (In-Game UI)

```bash
# Text
euca ui text "Score: 10" --x 0.5 --y 0.02 --size 24 --color white
euca ui text "GAME OVER" --x 0.5 --y 0.5 --size 48 --color red

# Bars (health, progress)
euca ui bar --x 0.02 --y 0.95 --width 0.2 --height 0.03 --fill 0.75 --color red

# Management
euca ui clear                     # Remove all HUD elements
euca ui list                      # Show current elements
```

Coordinates: (0,0) = top-left, (1,1) = bottom-right. HUD renders in the editor window.

### Entity Templates

```bash
# Define a template
euca template create soldier --mesh cube --health 100 --team 1 --color red --physics Dynamic --collider aabb:0.5,0.5,0.5

# Spawn instances at different positions
euca template spawn soldier --position 0,2,0
euca template spawn soldier --position 3,2,0
euca template spawn soldier --position 6,2,0

# List templates
euca template list
```

### Screenshot

```bash
euca screenshot                          # Save to temp file, print path
euca screenshot --output scene.png       # Save to specific path
```

Screenshots capture the 3D viewport (no HUD overlay). Use to verify entity placement, colors, camera angle.

### Scene

```bash
euca scene save my_scene.json            # Save world state
euca scene load my_scene.json            # Load (replaces current)
```

### Status & Schema

```bash
euca status                      # Engine version, entity count, tick
euca schema                      # All component types and fields
```

### Authentication

```bash
euca auth login                  # Login with nit identity
euca auth status                 # Check session
```

## Available Colors

Named: `red`, `blue`, `green`, `gold`, `silver`, `gray`, `white`, `black`, `yellow`, `cyan`, `magenta`, `orange`

## Available Meshes

`cube`, `sphere`, `plane`

## Components (visible in entity get)

| Component | Fields | Notes |
|-----------|--------|-------|
| transform | position, rotation, scale | All [f32; 3] or [f32; 4] |
| velocity | linear, angular | Both [f32; 3] |
| physics_body | Dynamic / Static / Kinematic | |
| collider | Aabb{hx,hy,hz}, Sphere{radius}, Capsule{radius,half_height} | |
| health | [current, max] | Only if --health was set |
| team | u8 | Only if --team was set |
| dead | true | Only if health reached 0 |

## Flag Reference

| Flag | Format | Used in |
|------|--------|---------|
| `--mesh` | cube/sphere/plane | entity create |
| `--color` | name or r,g,b | entity create, entity update |
| `--position` | x,y,z | entity create, entity update |
| `--scale` | x,y,z | entity create, entity update |
| `--velocity` | x,y,z | entity update |
| `--physics` | Dynamic/Static/Kinematic | entity create |
| `--collider` | aabb:h,h,h / sphere:r | entity create |
| `--health` | f32 | entity create |
| `--team` | u8 | entity create |
| `--json` | JSON string | entity create, entity update |
| `--dry-run` | flag | entity create, entity update |
| `--amount` | f32 | entity damage, entity heal |
| `--behavior` | idle/chase/patrol/flee | ai set |
| `--target` | entity ID | ai set, camera focus |
| `--speed` | f32 | ai set, projectile spawn |
| `--damage` | f32 | projectile spawn |
| `--action` | damage:N / heal:N | trigger create |
| `--fill` | 0.0-1.0 | ui bar |
| `--size` | pixels | ui text |
| `--when` | death/timer:N/health-below:N | rule create |
| `--filter` | any/entity:N/team:N | rule create |
| `--do-action` | action string (repeatable) | rule create |
| `--output` | file path | screenshot |
| `--server` | URL | global (default: http://localhost:3917) |

## Workflows

### Build a Deathmatch

```bash
euca game create --mode deathmatch --score-limit 5
euca entity create --mesh cube --position="-3,1,0" --health 100 --team 1 --color red --physics Dynamic --collider aabb:0.5,0.5,0.5
euca entity create --mesh sphere --position 3,1,0 --health 100 --team 2 --color blue --physics Dynamic --collider sphere:0.5
euca trigger create --position 0,0,0 --zone 1,1,1 --action damage:5
euca ai set 3 --behavior chase --target 2 --speed 3
euca rule create --when death --filter team:2 --do-action "score source +1"
euca rule create --when death --filter team:2 --do-action "spawn sphere 3,3,0 blue"
euca rule create --when timer:15 --do-action "spawn sphere 0,1,0 gold"
euca ui text "DEATHMATCH" --x 0.5 --y 0.02 --size 28 --color yellow
euca sim play
```

### Wave Survival (using rules)

```bash
euca game create --mode deathmatch --score-limit 20
euca entity create --mesh cube --position 0,1,0 --health 200 --team 1 --color red --physics Dynamic --collider aabb:0.5,0.5,0.5
# Spawn enemies every 8 seconds
euca rule create --when timer:8 --do-action "spawn sphere 5,2,0 blue"
# Enemies chase the player
euca rule create --when death --filter team:2 --do-action "score source +1"
# Heal player when enemy dies
euca rule create --when death --filter team:2 --do-action "heal entity:2 10"
euca sim play
```

### Inspect & Debug

```bash
euca entity list | jq '.entities[] | select(.health != null) | {id, health, team, dead}'
euca entity get 3
euca game state
euca camera view top && euca screenshot
```

### Physics Playground

```bash
euca entity create --mesh cube --position 0,5,0 --physics Dynamic --collider aabb:0.5,0.5,0.5 --color red
euca entity create --mesh sphere --position 0,10,0 --physics Dynamic --collider sphere:0.5 --color blue
euca sim play && sleep 2 && euca sim pause
euca screenshot
```

## Output

All commands return JSON. Pipe through `jq` for filtering:

```bash
euca entity list | jq '.entities[] | select(.team == 1) | .id'
euca game state | jq '.scores'
```

## Server Details

- **Port**: 3917 (default)
- **Protocol**: HTTP REST (JSON)
- **Override**: `euca --server http://localhost:PORT ...`
