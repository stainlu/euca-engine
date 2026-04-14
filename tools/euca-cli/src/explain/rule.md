# Rule — Data-Driven Game Logic

Rules let you define "when X happens, do Y" without writing code.
Each rule is an entity with a condition component. A single system
evaluates all rules per tick.

## Commands

```bash
# Create a rule (legacy string DSL).
euca rule create \
  --when death \
  --filter team:2 \
  --do-action "score source +1"

# List / delete
euca rule list
euca rule delete <id>
```

For new code, prefer defining rules inside a scenario JSON file — the
typed schema is safer and composable. See `euca explain scenario`.

## Rule triggers (when)

| Trigger              | String form        | Typed form (scenario JSON)              |
|----------------------|--------------------|-----------------------------------------|
| Entity dies          | `death`            | `{ "kind": "death" }`                   |
| Every N seconds      | `timer:N`          | `{ "kind": "timer", "interval": N }`    |
| Health below N       | `health-below:N`   | `{ "kind": "health_below", "threshold": N }` |
| Score reaches N      | `score:N`          | `{ "kind": "score", "threshold": N }`   |
| Game phase changes   | `phase:NAME`       | `{ "kind": "phase", "phase": "NAME" }`  |

## Rule filters (which entities the rule watches)

| Filter      | Meaning                                 |
|-------------|-----------------------------------------|
| `any`       | Matches every entity                    |
| `entity:ID` | A single entity by index                |
| `team:N`    | All entities on team N                  |

## Actions

Actions are typed. In scenario JSON:

```jsonc
[
  { "action": "spawn",    "mesh": "cube", "position": [0, 1, 0], "team": 2 },
  { "action": "damage",   "target": "this",   "amount": 50 },
  { "action": "heal",     "target": "source", "amount": 100 },
  { "action": "score",    "target": "source", "points": 1 },
  { "action": "despawn",  "target": "this" },
  { "action": "teleport", "target": "this", "position": [10, 5, 0] },
  { "action": "color",    "target": "this", "color": "gold" },
  { "action": "text",     "text": "GAME OVER", "x": 0.5, "y": 0.1 },
  { "action": "endgame",  "winner_team": 1 }
]
```

`target` can be `"this"` (the triggering entity), `"source"` (the
entity that caused the event, e.g. killer), or `{ "entity": N }` (by
index).

## Example

"When a team-2 minion dies, spawn a replacement at [0, 1, 0]":

```jsonc
{
  "when":  { "kind": "death" },
  "filter": "team:2",
  "actions": [
    { "action": "spawn", "mesh": "cube", "position": [0, 1, 0],
      "team": 2, "health": 80, "combat": true }
  ]
}
```
