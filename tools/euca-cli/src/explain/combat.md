# Combat — Auto-Targeting, Melee, Death

The `--combat` flag turns any entity into a fighter: it auto-detects
enemies, chases them, and deals damage in melee range. Combat works
with the physics system — fighters need a collider and a physics
body.

## Minimal fighter

```bash
euca entity create \
  --mesh cube \
  --position 0,2,0 \
  --health 100 \
  --team 1 \
  --color red \
  --combat \
  --physics Dynamic \
  --collider aabb:0.5,0.5,0.5
```

## Combat tuning fields

| Flag                | Default | Meaning                                 |
|---------------------|---------|-----------------------------------------|
| `--combat-damage`   | `10.0`  | HP dealt per hit                        |
| `--combat-range`    | `2.0`   | Detection/attack radius                 |
| `--combat-speed`    | `3.0`   | Chase speed (units/sec)                 |
| `--combat-cooldown` | `1.0`   | Seconds between attacks                 |
| `--combat-style`    | `melee` | `melee` or `stationary`                 |

## Example fight

```bash
# Red vs Blue, 1 fighter each.
euca entity create --mesh cube   --position -5,1,0 --health 100 \
  --team 1 --color red  --combat --physics Dynamic --collider aabb:0.5,0.5,0.5
euca entity create --mesh sphere --position  5,1,0 --health 100 \
  --team 2 --color blue --combat --physics Dynamic --collider sphere:0.5

euca sim play                    # Run — they'll find each other and fight
euca sim step --ticks 300        # Or step N ticks manually
euca entity list                 # See HP drop
euca game scoreboard             # See kills/deaths
```

## Stationary towers

Stationary fighters attack in range but don't chase:

```bash
euca entity create \
  --mesh cube --position -8,1,0 \
  --health 2000 --team 1 --color gold \
  --combat --combat-style stationary \
  --combat-range 4 --combat-damage 30 \
  --role tower \
  --physics Static --collider aabb:0.5,1,0.5
```

## Death + respawn

When HP reaches 0, the entity gains a `Dead` component. If your game
config has a `respawn_delay`, dead entities respawn automatically at
their team's spawn point.

```bash
euca game create --mode deathmatch --score-limit 10 --respawn-delay 3
# Fighters now respawn 3 seconds after death.
```

## Debugging

```bash
euca events list            # See damage/death events
euca entity get <id>        # Check an entity's HP and Dead state
euca diagnose               # Find broken entities (missing components, etc.)
```
