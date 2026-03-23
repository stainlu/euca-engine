# euca-cli

Command-line tool for controlling the Euca engine from the terminal via the agent HTTP API.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- 25+ command groups: entity, sim, scene, camera, game, trigger, projectile, ai, rule, template, ability, audio, input, nav, vfx, animation, terrain, foliage, prefab, material, postprocess, fog, auth, ui, schema
- Entity CRUD: create with mesh/position/health/team, list, update, delete
- Simulation control: play, pause, step, reset
- Scene management: save, load
- Diagnostics: profile, diagnose, events, status, screenshot
- nit authentication support
- Connects to the agent server (default `http://localhost:3917`)

## Usage

```sh
# Create an entity
euca entity create --mesh cube --position 0,2,0 --health 100 --team 1

# Control simulation
euca sim play
euca sim pause

# Capture a screenshot
euca screenshot --output viewport.png
```

## License

MIT
