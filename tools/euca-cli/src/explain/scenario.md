# Scenario — Declarative Game Setup

A **scenario** is a single JSON document that describes an entire game
state: templates, entities, rules, assertions, camera, game mode. It
replaces the imperative 28-command MOBA setup pipeline with one
atomic, round-trippable file.

## Commands

```bash
# Apply a scenario to the main world (wipes + loads fresh).
euca scenario load path/to/scenario.json

# Export the current main world as scenario JSON.
euca scenario save                     # to stdout
euca scenario save --out snapshot.json # to a file

# Apply a scenario to a fork (main world untouched).
euca scenario apply-to-fork <fork_id> path/to/scenario.json
```

## Scenario format (v2)

```jsonc
{
  "version": 2,
  "name": "single-lane-moba",

  "templates": {
    "hero":   { "mesh": "cube", "health": 500, "team": 1, "combat": true },
    "minion": { "mesh": "cube", "health": 80,  "team": 2 }
  },

  "entities": [
    { "template": "hero",   "position": [0, 1, 0] },
    { "template": "minion", "position": [5, 1, 0],
      "overrides": { "health": 200 } }
  ],

  "rules": [
    {
      "when":  { "kind": "death" },
      "filter": "team:2",
      "actions": [
        { "action": "score", "target": "source", "points": 1 }
      ]
    },
    {
      "when":  { "kind": "timer", "interval": 20.0 },
      "actions": [
        { "action": "spawn", "mesh": "cube", "position": [0, 1, 0], "team": 2 }
      ]
    }
  ],

  "assertions": [
    {
      "name": "team1_alive",
      "condition": {
        "type": "entity_count",
        "filter": { "type": "team", "team": 1 },
        "min": 1
      },
      "severity": "error"
    }
  ],

  "camera": { "eye": [0, 10, -10], "target": [0, 0, 0], "fov_y": 60.0 },
  "game":   { "mode": "deathmatch", "score_limit": 10, "auto_start": true }
}
```

## Why declarative beats imperative

Before scenarios, setup was 28 ordered commands. Any one could fail
mid-setup, leaving the world half-built; agents had to figure out where
it broke and resume from partial state.

With scenarios, the engine receives the **entire description** and
applies it atomically. No partial states. No command-ordering bugs.

## Composition with fork

Scenarios pair naturally with `euca fork`:

```bash
# Baseline.
euca scenario save --out /tmp/baseline.json

# Fork, then apply a variant.
euca fork create buff-test
euca scenario apply-to-fork buff-test /tmp/variant.json
euca fork probe buff-test --ticks 600 --assertions team1_wins
euca fork delete buff-test
```

See `euca explain fork` for the full counterfactual workflow.
