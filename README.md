# Euca Engine

[![CI](https://github.com/stainlu/euca-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/stainlu/euca-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/stainlu/euca-engine)](https://github.com/stainlu/euca-engine/releases)
[![Rust](https://img.shields.io/badge/Rust-1.89+-orange.svg)](https://www.rust-lang.org)
[![Website](https://img.shields.io/badge/Website-eucaengine.com-green)](https://eucaengine.com)

An ECS-first, agent-native game engine in Rust. AI agents build games via CLI commands.

## Design Goals

* **ECS-First** — Everything is entities, components, and systems. No inheritance, no god objects.
* **Agent-Native** — AI agents control the engine via CLI/HTTP. The engine is the runtime, agents are the developers.
* **Composable** — Pick the systems you need. Health + Team + AutoCombat = a fighter. Add Gold + XpBounty = an RPG enemy. No framework lock-in.
* **Data-Driven** — Game logic via rules ("when death → score +1"), not code. Agents compose behavior, never write Rust.
* **Fast** — Custom ECS with archetype storage, parallel queries, 60+ FPS with 50+ entities.

## Links

* **[Website](https://eucaengine.com)** — Landing page
* **[CLI Reference (SKILL.md)](SKILL.md)** — Complete command reference for agents
* **[MOBA Demo](scripts/moba.sh)** — Full working game built entirely from CLI commands
* **[Examples](examples/)** — Editor, headless server, agent client

## 30-Second Demo

```bash
git clone https://github.com/stainlu/euca-engine.git
cd euca-engine
cargo build -p euca-editor --example editor -p euca-cli

# Terminal 1: start the editor
cargo run -p euca-editor --example editor

# Terminal 2: run the MOBA demo
./scripts/moba.sh
```

Heroes charge, fight, die, respawn. Minions spawn in waves. Towers attack. Gold and XP accumulate. All built from CLI commands — no game code.

## What It Can Do

| System | Description |
|--------|-------------|
| **ECS** | Custom archetype storage, generational entities, parallel queries, change detection |
| **Rendering** | Forward+ PBR, 3-cascade shadows, 4x MSAA, bloom, ACES tonemapping, sky dome |
| **Physics** | AABB/sphere/capsule collision, raycasting, fixed-timestep, CCD |
| **Combat** | AutoCombat (melee/stationary), targeting priority, projectiles |
| **Economy** | Gold, XP, leveling (auto stat boost), bounties |
| **Abilities** | Q/W/E/R slots, cooldowns, mana, effects (AoE damage, heal, speed boost) |
| **AI** | Patrol, chase, flee + combat hybrid (fight when enemies appear, patrol when clear) |
| **Rules** | Data-driven: "when death → score +1", "every 10s → spawn minions" |
| **Audio** | Spatial audio via kira |
| **Animation** | glTF skeletal animation |
| **Particles** | CPU particle emitters |
| **Navigation** | Grid navmesh, A* pathfinding, steering |
| **Networking** | UDP transport, interest culling, bandwidth budgeting |
| **Editor** | egui: hierarchy, inspector, play/pause, gizmos, undo/redo |
| **Diagnostics** | `euca diagnose` health check, `euca events` real-time debugging |

## Architecture

```
AI Agents (Claude Code, scripts, RL agents)
    │ CLI (euca) / HTTP REST (port 3917)
    ▼
┌─ Agent Layer (euca-agent) ─────────────────┐
│  50+ endpoints: spawn, observe, combat,    │
│  rules, economy, abilities, diagnose       │
└────────────────────────────────────────────┘
    │
┌─ Gameplay (euca-gameplay) ─────────────────┐
│  Health, Teams, Combat, Economy, Leveling,  │
│  Abilities, Rules, Triggers, AI            │
└────────────────────────────────────────────┘
    │
┌─ Engine Core ──────────────────────────────┐
│  ECS (euca-ecs)     Render (euca-render)   │
│  Scene (euca-scene) Physics (euca-physics) │
│  Audio (euca-audio) Particles (euca-particle)
│  Nav (euca-nav)     Asset (euca-asset)     │
└────────────────────────────────────────────┘
    │
┌─ Editor (euca-editor) ────────────────────┐
│  3D viewport, hierarchy, inspector,       │
│  play/pause, gizmos, undo/redo            │
└───────────────────────────────────────────┘
```

## Crates (19)

| Crate | Purpose |
|-------|---------|
| `euca-ecs` | Archetype ECS: Entity, World, Query, Schedule, Events, change detection |
| `euca-math` | Vec2/3/4, Quat, Mat4, Transform, AABB (zero external deps) |
| `euca-reflect` | Runtime reflection via `#[derive(Reflect)]` |
| `euca-scene` | Transform hierarchy: LocalTransform, GlobalTransform, Parent/Children |
| `euca-core` | App lifecycle, Plugin trait, Time resource |
| `euca-render` | wgpu PBR: shadows, MSAA, bloom, tone mapping, instancing (16K) |
| `euca-physics` | Collision (AABB/sphere/capsule), raycasting, fixed-timestep, CCD |
| `euca-gameplay` | Health, damage, teams, combat, economy, leveling, abilities, rules, AI |
| `euca-audio` | Spatial audio (kira): AudioSource, AudioListener |
| `euca-asset` | glTF loading, skeletal animation, async AssetStore, hot-reload |
| `euca-particle` | CPU particle emitters with gravity, color gradients |
| `euca-nav` | Grid navmesh, A* pathfinding, steering behaviors |
| `euca-input` | InputState, ActionMap, gamepad, input contexts |
| `euca-net` | UDP transport, interest culling, bandwidth budgeting, tick rate |
| `euca-agent` | HTTP API (axum), 50+ endpoints, nit auth, HUD canvas |
| `euca-editor` | egui editor: viewport, panels, gizmos, undo, scene save/load |
| `euca-game` | Standalone game runner |
| `euca-cli` | CLI tool: 20+ command groups |

## CLI Reference

See [SKILL.md](SKILL.md) for the complete CLI reference.

```bash
euca status                    # Engine info
euca entity create --mesh cube --position 0,2,0 --health 100 --team 1 --combat
euca sim play                  # Start simulation
euca diagnose                  # Check for broken entities
euca screenshot                # Capture viewport
```

## Requirements

- Rust 1.89+ (Edition 2024)
- macOS or Linux (wgpu for rendering)
- `libasound2-dev` on Linux (for audio)

## Contributing

Contributions welcome. The engine is early-stage — there's plenty to do:

1. Fork and create a branch
2. `cargo test --workspace` must pass
3. `cargo clippy --workspace -- -D warnings` must be clean
4. Open a PR with a clear description

## License

MIT — see [LICENSE](LICENSE)
