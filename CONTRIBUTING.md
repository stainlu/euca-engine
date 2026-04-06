# Contributing to Euca Engine

Thank you for your interest in contributing to Euca Engine. This document
covers the toolchain requirements, build and test commands, code-quality
expectations, and the contribution workflow.

## Prerequisites

| Requirement | Minimum version |
|-------------|-----------------|
| Rust        | 1.89+ (edition 2024) |
| OS          | macOS (Metal backend), Linux, or Windows (wgpu backend) |
| System libs | `libasound2-dev` on Linux (for audio) |

Install Rust via [rustup](https://rustup.rs/). The project uses the `2024`
edition and resolver v2.

## Building

Build the entire workspace:

```bash
cargo build --workspace
```

Build a specific crate:

```bash
cargo build -p euca-ecs
```

Build the editor example:

```bash
cargo build -p euca-editor --example editor
```

Build with the native Metal backend (macOS only):

```bash
cargo build -p euca-game --features metal-native
```

## Running Tests

Run the full test suite:

```bash
cargo test --workspace
```

Run tests for a single crate:

```bash
cargo test -p euca-physics
```

Run benchmarks (macOS recommended):

```bash
cargo bench --workspace
```

## Code Style

All code must pass formatting and lint checks before merge. CI enforces both.

**Format** -- uses the default `rustfmt` configuration:

```bash
cargo fmt --all -- --check   # verify
cargo fmt --all              # fix
```

**Lint** -- clippy with warnings as errors:

```bash
cargo clippy --workspace -- -D warnings
```

## CI Checks

GitHub Actions runs on every push and pull request to `main`. The full matrix:

| Job | Command | Platform |
|-----|---------|----------|
| Check | `cargo check --workspace` | Linux |
| Test | `cargo test --workspace` | Linux + macOS |
| Clippy | `cargo clippy --workspace -- -D warnings` | Linux |
| Format | `cargo fmt --all -- --check` | Linux |
| Metal | `cargo check -p euca-game --features metal-native` | macOS |
| WASM | `cargo check --target wasm32-unknown-unknown -p euca-web-demo` | Linux |
| Bench | `cargo bench --workspace` | macOS (main only) |

Before submitting, run the full local check:

```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

## Running Examples

Examples live in the `examples/` directory and are compiled under `euca-game`:

```bash
# Dota-style client
cargo run -p euca-game --example dota_client --release

# Editor with egui viewport
cargo run -p euca-editor --example editor

# Headless server (no GPU)
cargo run -p euca-game --example headless_server

# Physics demo
cargo run -p euca-game --example physics_demo

# Metal stress test (macOS only)
cargo run -p euca-game --example metal_stress --release --features metal-native
```

The MOBA demo script orchestrates the editor and CLI together:

```bash
./scripts/moba.sh
```

## Architecture

Euca Engine is a workspace of **26 crates** (under `crates/`), plus tools, games,
and services. The engine is ECS-first and agent-native -- AI agents control the
engine via the `euca` CLI backed by an HTTP REST API.

Key crates:

| Crate | Role |
|-------|------|
| `euca-ecs` | Archetype-based ECS: entities, queries, schedules, events |
| `euca-render` | Forward+ PBR renderer, GPU-driven pipeline |
| `euca-rhi` | Render Hardware Interface (wgpu + native Metal) |
| `euca-physics` | Collision, spatial queries, character controller |
| `euca-agent` | HTTP API for external AI agents |
| `euca-editor` | egui-based editor |
| `euca-scene` | Transform hierarchy, prefabs, world streaming |
| `euca-core` | App lifecycle, Plugin trait |

For the full architecture, dependency graph, and design decisions, see
[DESIGN.md](DESIGN.md).

## Contribution Workflow

This project commits directly to `main` -- no feature branches or pull requests
unless explicitly coordinated.

1. Clone the repository:
   ```bash
   git clone https://github.com/stainlu/euca-engine.git
   cd euca-engine
   ```
2. Make your changes.
3. Run the full check suite (see [CI Checks](#ci-checks)).
4. Commit with a concise message in imperative mood:
   ```
   Add collision layer filtering to physics queries
   ```
5. Push to `main`:
   ```bash
   git push origin main
   ```
6. Verify CI passes on GitHub Actions.

## License

By contributing, you agree that your contributions will be licensed under the
project's dual license: MIT OR Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE).
