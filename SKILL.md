---
name: eucaengine
description: ECS-first, agent-native game engine in Rust
version: 1.5.0
auth: nit
protocol: cli
---

# EucaEngine — Agent Interface

EucaEngine is a game engine you control via the `euca` CLI. The engine
runs as a persistent process on `http://localhost:3917` and exposes a
stateful ECS world; every command is an HTTP call.

## Progressive disclosure

This file is intentionally short. Discover the rest on demand:

- `euca discover`                    — list all command groups
- `euca discover <group>`            — commands in one group, with args
- `euca discover --scope core`       — hide genre-specific vocabulary
- `euca discover --json`             — full machine-readable manifest
- `euca explain <topic>`             — focused worked example on a topic
- `euca explain`                     — list all available explain topics

Available `explain` topics: `quickstart`, `entity`, `combat`, `rule`,
`assert`, `fork`, `scenario`.

For a hands-on introduction, run `euca explain quickstart`.

## Bootstrap sequence

The minimum commands to verify the engine is alive and spawn your
first entity:

```bash
# 1. Start the engine (pick one).
cargo run -p euca-editor --example editor           # visual editor
cargo run -p euca-agent  --example agent_headless   # headless for agents

# 2. Connect.
euca                                                # status

# 3. Spawn.
euca entity create --mesh cube --position 0,2,0 --health 100 --team 1 --combat

# 4. Run the sim.
euca sim play
euca sim step --ticks 100
```

That's ~90% of the knowledge needed to bootstrap. Everything else
lives behind `discover` and `explain`.

## Core primitives (reference)

| Primitive   | Purpose                                    | Explain topic |
|-------------|--------------------------------------------|---------------|
| `entity`    | Spawn and edit entities                    | `entity`      |
| `sim`       | Advance time (play/pause/step/reset)       | `quickstart`  |
| `rule`      | Data-driven game logic                     | `rule`        |
| `assert`    | Testable expectations                      | `assert`      |
| `fork`      | Counterfactual simulation (clone world)    | `fork`        |
| `scenario`  | Declarative game setup (one JSON document) | `scenario`    |
| `game`      | Match mode, score, time limit              | —             |
| `scene`     | Save/load world state to disk              | —             |
| `camera`    | Move the view                              | —             |

## The counterfactual loop (agent workflow)

The engine is designed for agents that iterate: hypothesize →
intervene → simulate → observe → decide → repeat. The canonical loop:

```bash
# 1. Set up a baseline.
euca scenario load path/to/baseline.json

# 2. Create a fork to test a hypothesis.
euca fork create experiment

# 3. Apply an intervention to the fork.
euca scenario apply-to-fork experiment path/to/variant.json

# 4. Advance the fork and evaluate.
euca fork probe experiment --ticks 600 --assertions team1_wins

# 5. Compare, decide, drop.
euca fork delete experiment
```

See `euca explain fork` and `euca explain scenario` for details.

## Authentication

Agents authenticate with [nit](https://github.com/newtype-ai/nit)
Ed25519 identity. Generated entities track their owner via the `Owner`
component. If you're writing an agent: `nit init` once, then include
your identity in request headers.

## Where to go next

- **New to euca?** `euca explain quickstart`
- **Building a game loop?** `euca explain entity` → `rule` → `assert`
- **Want counterfactuals?** `euca explain fork` + `scenario`
- **Full human docs?** See `GUIDE.md` in the repo root (not preloaded)
