# Euca Engine — Quickstart

Euca is a game engine you control via the `euca` CLI. The engine runs
as a persistent process on `http://localhost:3917` and exposes a
stateful ECS world; the CLI is a thin wrapper around that HTTP API.

## 1. Start the engine

Pick one:

```bash
# Visual editor (opens a window with a 3D viewport).
cargo run -p euca-editor --example editor

# Headless server (no window — ideal for agents).
cargo run -p euca-agent --example agent_headless
```

Both options serve the same HTTP API at `http://localhost:3917`.

## 2. Confirm it's alive

```bash
euca                              # Status
euca discover                     # List all command groups
```

If you see a command groups list, you're connected.

## 3. Spawn your first entity

```bash
euca entity create \
  --mesh cube \
  --position 0,2,0 \
  --health 100 \
  --team 1 \
  --color red \
  --combat

euca entity list                  # Verify it shows up
```

## 4. Run the sim

```bash
euca sim play                     # Start physics + gameplay
euca sim step --ticks 10          # Or advance manually
```

## Discovering more

This quickstart covers ~0.5% of the engine. Use:

- `euca discover <group>` — list commands in a group (e.g. `euca discover entity`)
- `euca discover --json` — full command manifest for agents
- `euca explain <topic>` — deep dive on a single topic

Available topics: `fork`, `scenario`, `entity`, `rule`, `assert`, `combat`.
