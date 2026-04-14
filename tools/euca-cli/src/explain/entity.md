# Entity — Spawning and Editing Entities

Entities are the atoms of the engine. Every visible thing — heroes,
towers, projectiles, ground planes — is an entity with components.

## CRUD

```bash
# Create
euca entity create --mesh cube --position 0,2,0 --color red
euca entity create --mesh sphere --position 3,2,0 --physics Dynamic --collider sphere:0.5
euca entity create --json '{"mesh":"cube","position":[1,2,3],"health":100,"team":1}'

# List / read
euca entity list                         # All entities as JSON
euca entity get <id>                     # Single entity (health, team, transform, etc.)

# Update
euca entity update <id> --position 5,0,0
euca entity update <id> --color green
euca entity update <id> --velocity 0,10,0

# Delete
euca entity delete <id>
euca entity delete --all

# Damage / heal
euca entity damage <id> --amount 25
euca entity heal <id> --amount 10
```

## Components you'll use most

| Field              | Meaning                                          |
|--------------------|--------------------------------------------------|
| `mesh`             | `cube`, `sphere`, or a glTF filename             |
| `position`         | `x,y,z` in world space                           |
| `color`            | `red`, `blue`, `green`, `yellow`, `cyan`, ...    |
| `health`           | Max HP (entity also gets a `Health` component)   |
| `team`             | `1` or `2` (used by combat + assertions)         |
| `combat`           | `true` to auto-detect + chase + melee attack     |
| `physics_body`     | `Static`, `Kinematic`, `Dynamic`                 |
| `collider`         | `aabb:x,y,z` or `sphere:r` or `capsule:r,h`      |
| `velocity`         | Initial linear velocity                          |
| `role`             | `hero`, `minion`, `tower`, `structure`           |

## Templates

If you're spawning many entities of the same kind, define a template
once and spawn from it:

```bash
euca template create soldier --json '{
  "mesh": "cube",
  "health": 100,
  "team": 1,
  "combat": true,
  "physics_body": "Dynamic",
  "collider": "aabb:0.5,0.5,0.5"
}'

euca template spawn soldier --position 0,1,0
euca template spawn soldier --position 2,1,0
euca template spawn soldier --position 4,1,0
```

For a full declarative setup, use `euca explain scenario` instead —
scenarios bundle templates, entities, rules, and assertions in one
atomic JSON document.
