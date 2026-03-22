---
name: eucaengine
description: ECS-first, agent-native game engine in Rust
version: 0.4.0
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

# 4. Spawn fighters (--combat enables auto-detect, chase, melee attack)
euca entity create --mesh cube --position 0,2,0 --health 100 --team 1 --color red --combat --physics Dynamic --collider aabb:0.5,0.5,0.5
euca entity create --mesh sphere --position 3,2,0 --health 100 --team 2 --color blue --combat --physics Dynamic --collider sphere:0.5

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
euca entity create --mesh cube --position 0,1,0 --health 100 --team 1 --color red --combat --physics Dynamic --collider aabb:0.5,0.5,0.5
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
physics -> damage -> death -> projectiles -> triggers -> AI -> auto_combat -> game state -> respawn

### Game Match

```bash
euca game create --mode deathmatch --score-limit 10
euca game state                  # Phase, scores, elapsed time
```

Game phases: lobby -> playing -> post_match (when score limit reached)

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

### AutoCombat (Auto-PvP)

Add `--combat` to any entity with `--health` and `--team` to enable automatic combat:

```bash
# Entities with --combat auto-detect enemies, chase, and melee attack
euca entity create --mesh cube --position -3,1,0 --health 100 --team 1 --color red --combat --physics Dynamic --collider aabb:0.5,0.5,0.5
euca entity create --mesh sphere --position 3,1,0 --health 80 --team 2 --color blue --combat --physics Dynamic --collider sphere:0.5
```

**Defaults:** damage: 10, range: 1.5, cooldown: 1.0s, detect_range: 20, chase_speed: 3.0

**Behavior per tick:**
1. Detect nearest alive entity on a different team within detect_range
2. If in attack range (1.5): deal damage (DamageEvent), wait cooldown
3. If out of range: chase (set Velocity toward target)
4. If no enemy found: march in the entity's MarchDirection (or stand still if none)

**March Direction:** Combat entities automatically receive a MarchDirection based on team. Team 1 marches in +X (right), team 2 marches in -X (left). When no enemy is in detect range, the entity advances toward the opposing side. Marching stops as soon as a target is found.

No `ai set` command needed — `--combat` handles everything automatically.

**Combat Customization:**
```bash
euca entity create --mesh cube --position 0,1,0 --health 800 --team 1 \
  --combat --combat-damage 40 --combat-range 5 --combat-cooldown 1.5 \
  --combat-speed 0 --combat-style stationary --physics Static --role tower
```

| Flag | Default | Description |
|------|---------|-------------|
| `--combat-damage` | 10 | Damage per attack |
| `--combat-range` | 1.5 | Attack range (units) |
| `--combat-speed` | 3.0 | Chase speed |
| `--combat-cooldown` | 1.0 | Seconds between attacks |
| `--combat-style` | melee | `melee` (chase) or `stationary` (tower mode) |

**Physics Body Types:**
- `--physics Dynamic`: physics-driven (gravity, collision response). For projectiles, falling objects.
- `--physics Kinematic`: gameplay-driven (no gravity, no collision blocking). For heroes, minions.
- `--physics Static`: immovable obstacle. For towers, walls, ground.

**Entity Roles:** `--role hero|minion|tower|structure` — affects targeting priority (towers attack minions before heroes).

**AI Patrol:** `--ai-patrol "-7,0,0:0,0,0:7,0,0"` — colon-separated waypoints. Entities patrol then fight when enemies appear.

**Spawn Points:** `--spawn-point <team>` — marks an entity as a respawn location for that team.

**Auto Health Bars:** Entities with a `Health` component automatically display a health bar above them in the viewport. Bars are colored by team (red = team 1, blue = team 2, green = other) and shrink as health decreases.

### Economy

```bash
# Hero with gold wallet and bounty
euca entity create --mesh sphere --position 0,1,0 --health 500 --team 1 \
  --combat --role hero --gold 0 --gold-bounty 300 --xp-bounty 200

# Check hero's gold, level, XP
euca ability list <id>
```

| Flag | Description |
|------|-------------|
| `--gold` | Starting gold amount |
| `--gold-bounty` | Gold awarded to killer on death |
| `--xp-bounty` | XP awarded to killer on death |

Heroes level up automatically when XP reaches threshold. Each level: +50 max HP, +5 attack damage.

### Abilities

```bash
euca ability use <entity_id> --slot Q    # Cast ability (Q/W/E/R)
euca ability list <entity_id>            # Show abilities, mana, gold, level
```

Abilities have cooldowns, mana costs, and effects (AreaDamage, Heal, SpeedBoost).

### Diagnostics

```bash
euca diagnose    # Scan all entities for problems
euca events      # Show pending damage/death/spawn events
```

`diagnose` checks: missing Velocity on combat entities, dead entities stuck without respawn timer, teams without spawn points, invisible entities.

### Visual Effects (Particles)

```bash
euca vfx spawn --position 0,3,0 --rate 100 --lifetime 1.5
euca vfx stop <entity_id>
euca vfx list
```

### Navigation

```bash
euca nav generate --cell-size 1.0    # Build navmesh from colliders
euca nav compute --from 0,0,0 --to 10,0,10    # A* pathfinding
euca nav set <entity_id> --target 10,0,5 --speed 5
```

### Audio

```bash
euca audio play <path> [--position x,y,z] [--volume 0.8] [--loop]
euca audio stop <entity_id>
euca audio list
```

### Animation

```bash
euca animation load <path.glb>       # Load glTF with animations
euca animation play <entity_id> --clip 0 [--speed 1.0] [--loop]
euca animation stop <entity_id>
euca animation list
```

### Terrain

```bash
# Create a heightmap terrain grid
euca terrain create --width 64 --height 64 --cell-size 1.0

# Edit terrain: raise, lower, flatten, or smooth at a point
euca terrain edit --op raise --x 10 --z 10 --radius 3 --amount 0.5
euca terrain edit --op lower --x 20 --z 20 --radius 5 --amount 1.0
euca terrain edit --op flatten --x 15 --z 15 --radius 4 --amount 0.0
euca terrain edit --op smooth --x 10 --z 10 --radius 6 --amount 0.3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--width` | 64 | Grid columns |
| `--height` | 64 | Grid rows |
| `--cell-size` | 1.0 | World-space size per cell |
| `--op` | raise | Operation: `raise`, `lower`, `flatten`, `smooth` |
| `--x` | required | X coordinate on the heightmap |
| `--z` | required | Z coordinate on the heightmap |
| `--radius` | 3 | Brush radius (cells) |
| `--amount` | 0.5 | Brush strength |

### Prefab

```bash
# Spawn a registered prefab by name
euca prefab spawn --name watchtower --position 5,0,3

# List all available prefabs
euca prefab list
```

Prefabs are pre-configured entity bundles registered with the engine. Use `prefab list` to see what is available, then `prefab spawn` to instantiate at a position.

### Material

```bash
# Set PBR material properties on an entity
euca material set --entity <id> --metallic 1.0 --roughness 0.2
euca material set --entity <id> --emissive 1.0,0.5,0.0
euca material set --entity <id> --alpha-mode blend
```

| Flag | Format | Description |
|------|--------|-------------|
| `--entity` | u32 | Target entity ID (required) |
| `--metallic` | 0.0-1.0 | Metallic factor |
| `--roughness` | 0.0-1.0 | Roughness factor |
| `--emissive` | r,g,b | Emissive color (HDR values allowed) |
| `--alpha-mode` | opaque/blend | Transparency mode |

### Post-Processing

```bash
# Get current post-processing settings
euca postprocess get

# Toggle individual effects
euca postprocess set --ssao true             # Screen-space ambient occlusion
euca postprocess set --fxaa true             # Fast approximate anti-aliasing
euca postprocess set --bloom true            # Bloom glow effect

# Adjust exposure and color grading
euca postprocess set --exposure 1.2
euca postprocess set --contrast 1.1 --saturation 0.9

# Combine multiple settings in one call
euca postprocess set --ssao true --fxaa true --bloom true --exposure 1.0 --contrast 1.0 --saturation 1.0
```

| Flag | Format | Description |
|------|--------|-------------|
| `--ssao` | bool | Enable/disable screen-space ambient occlusion |
| `--fxaa` | bool | Enable/disable fast approximate anti-aliasing |
| `--bloom` | bool | Enable/disable bloom |
| `--exposure` | f32 | Exposure multiplier |
| `--contrast` | f32 | Contrast adjustment |
| `--saturation` | f32 | Saturation adjustment |

### Input

```bash
euca input bind W move_forward
euca input unbind W
euca input list

# Input context stack
euca input context-push gameplay   # Push a context (gameplay, menu, editor)
euca input context-pop             # Pop the top context
```

Input contexts form a stack. Only bindings in the top context are active. Push `menu` to capture input for a pause screen, then pop to return to gameplay.

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
| auto_combat | damage, range, cooldown, detect_range, speed | Only if --combat was set |
| march_direction | [f32; 3] | Auto-set on combat entities (team 1: +X, team 2: -X) |

## Flag Reference

| Flag | Format | Used in |
|------|--------|---------|
| `--mesh` | cube/sphere/plane | entity create |
| `--color` | name or r,g,b | entity create, entity update |
| `--position` | x,y,z | entity create, entity update, prefab spawn |
| `--scale` | x,y,z | entity create, entity update |
| `--velocity` | x,y,z | entity update |
| `--physics` | Dynamic/Static/Kinematic | entity create, entity update |
| `--collider` | aabb:h,h,h / sphere:r / capsule:r,hh | entity create, entity update |
| `--health` | f32 | entity create |
| `--team` | u8 | entity create |
| `--combat` | flag | entity create, template create |
| `--json` | JSON string | entity create, entity update |
| `--dry-run` | flag | entity create, entity update |
| `--amount` | f32 | entity damage, entity heal, terrain edit |
| `--behavior` | idle/chase/patrol/flee | ai set |
| `--target` | entity ID | ai set, camera focus |
| `--speed` | f32 | ai set, projectile spawn, nav set |
| `--damage` | f32 | projectile spawn |
| `--action` | damage:N / heal:N | trigger create |
| `--fill` | 0.0-1.0 | ui bar |
| `--size` | pixels | ui text |
| `--when` | death/timer:N/health-below:N | rule create |
| `--filter` | any/entity:N/team:N | rule create |
| `--do-action` | action string (repeatable) | rule create |
| `--output` | file path | screenshot |
| `--server` | URL | global (default: http://localhost:3917) |
| `--name` | string | prefab spawn, template create/spawn |
| `--entity` | u32 | material set |
| `--metallic` | 0.0-1.0 | material set |
| `--roughness` | 0.0-1.0 | material set |
| `--emissive` | r,g,b | material set |
| `--alpha-mode` | opaque/blend | material set |
| `--ssao` | bool | postprocess set |
| `--fxaa` | bool | postprocess set |
| `--bloom` | bool | postprocess set |
| `--exposure` | f32 | postprocess set |
| `--contrast` | f32 | postprocess set |
| `--saturation` | f32 | postprocess set |
| `--width` | u32 | terrain create |
| `--height` | u32 | terrain create |
| `--cell-size` | f32 | terrain create, nav generate |
| `--op` | raise/lower/flatten/smooth | terrain edit |
| `--x` | f32 | terrain edit |
| `--z` | f32 | terrain edit |
| `--radius` | f32 | terrain edit |

## Workflows

### Arena Survival (full showcase)

```bash
# 1. Define templates (--combat enables auto-fight)
euca template create soldier --mesh cube --health 100 --team 1 --color red --combat --physics Dynamic --collider aabb:0.5,0.5,0.5
euca template create enemy --mesh sphere --health 60 --team 2 --color blue --combat --physics Dynamic --collider sphere:0.5

# 2. Create match
euca game create --mode deathmatch --score-limit 10

# 3. Spawn teams using templates
euca template spawn soldier --position="-4,1,0"
euca template spawn soldier --position="-4,1,3"
euca template spawn enemy --position 4,1,0
euca template spawn enemy --position 4,1,3
euca template spawn enemy --position 2,1,1

# 4. Center hazard zone
euca trigger create --position 0,0,0 --zone 1.5,1,1.5 --action damage:10

# 5. AI — enemies chase soldiers
euca ai set 4 --behavior chase --target 2 --speed 4
euca ai set 5 --behavior chase --target 3 --speed 4
euca ai set 6 --behavior chase --target 2 --speed 3

# 6. Rules — scoring, respawning, milestones
euca rule create --when death --filter team:2 --do-action "score source +1"
euca rule create --when death --filter team:2 --do-action "spawn sphere 4,3,0"
euca rule create --when timer:8 --do-action "spawn sphere 5,2,0"
euca rule create --when score:5 --do-action "text HALFWAY!"
euca rule create --when phase:post_match --do-action "text GAME OVER!"

# 7. HUD
euca ui text "ARENA SURVIVAL" --x 0.5 --y 0.01 --size 28 --color yellow
euca ui bar --x 0.02 --y 0.92 --width 0.12 --height 0.02 --fill 1.0 --color red
euca ui bar --x 0.86 --y 0.92 --width 0.12 --height 0.02 --fill 1.0 --color blue

# 8. Fire a projectile
euca projectile spawn --from="-4,1,0" --direction 1,0,0 --speed 15 --damage 30

# 9. Screenshot, simulate, check
euca screenshot --output setup.png
euca sim play
# wait...
euca sim pause
euca game state
euca camera view top && euca screenshot --output topdown.png
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

### Asset Pipeline (offline)

These commands work without the engine running — they process files directly.

```bash
# Show metadata about a glTF/glb file
euca asset info model.glb

# Run mesh optimization (dedup, tangents, cache reorder)
euca asset optimize model.glb
euca asset optimize model.glb -o stats.json   # Save stats to file

# Generate LOD chain
euca asset lod model.glb --levels 4
euca asset lod model.glb --levels 6 -o lod_stats.json
```

### Discover (offline)

Self-describing CLI for AI agents and humans. Always in sync with actual commands.

```bash
# Human-readable overview of all command groups
euca discover

# Machine-readable JSON manifest (for AI agents)
euca discover --json

# Details for a specific group
euca discover entity
euca discover asset
```

### Package (offline)

```bash
# Package game for distribution
euca package --project . --output dist/
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
