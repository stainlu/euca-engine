# Assert — Testable Expectations

Assertions are first-class ECS entities that encode expectations
about the world state. Use them to verify game rules, catch
regressions, and drive counterfactual probes.

Each assertion has a **name**, a **condition**, and a **severity**.
Running `evaluate` checks all assertions and reports pass/fail.

## Commands

```bash
# Create
euca assert create \
  --name hero-alive \
  --condition entity-exists \
  --filter role:hero

# Evaluate all (returns pass/fail per assertion)
euca assert evaluate

# List / inspect / delete
euca assert list
euca assert results        # Last evaluation without re-running
euca assert delete <id>
```

For automation, assertions live in scenario files. See
`euca explain scenario`.

## Conditions (typed)

| Condition                    | What it checks                               |
|------------------------------|----------------------------------------------|
| `entity_exists`              | At least one entity matching the filter      |
| `entity_count` (min/max)     | Count of matching entities in a range        |
| `field_check` (op/value)     | Numeric field on entities passes a comparison|
| `all_teams_have_spawn_points`| Every team with entities has a SpawnPoint    |
| `no_overlap` (min_distance)  | Matching entities aren't closer than N       |
| `none_are_dead`              | No matching entity has the Dead component    |
| `no_zero_health_alive`       | No living entity has HP ≤ 0                  |
| `all_renderable_have_transform` | Every MeshRenderer has a GlobalTransform  |
| `game_phase`                 | Game is in the named phase                   |
| `entity_budget` (max)        | Total entity count is below the budget       |

## Filter types

```jsonc
{ "type": "any" }
{ "type": "team", "team": 1 }
{ "type": "role", "role": "hero" }
{ "type": "tag",  "tag":  "boss" }
{ "type": "has_component", "component": "Health" }
{ "type": "and", "filters": [ ... ] }
```

## Example — team 1 must have a hero

```jsonc
{
  "name": "team1_has_hero",
  "condition": {
    "type": "entity_count",
    "filter": {
      "type": "and",
      "filters": [
        { "type": "team", "team": 1 },
        { "type": "role", "role": "hero" }
      ]
    },
    "min": 1
  },
  "severity": "error"
}
```

## Composing with probes

`euca fork probe` advances a fork AND evaluates assertions in a single
atomic call. This is the canonical agent testing loop:

```bash
euca fork create test
euca scenario apply-to-fork test /tmp/scenario.json
euca fork probe test --ticks 600 --assertions team1_wins
euca fork delete test
```

See `euca explain fork` for the full counterfactual workflow.
