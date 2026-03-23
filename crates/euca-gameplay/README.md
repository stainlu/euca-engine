# euca-gameplay

Composable ECS game logic: health, combat, teams, economy, abilities, AI behaviors, and rules engine.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `Health`, `DamageEvent`, `Dead`, death/respawn lifecycle systems
- `Team`, `SpawnPoint`, `RespawnTimer` for team-based multiplayer
- `GameState` with phases, match config, and score tracking
- `AutoCombat` with target acquisition, projectiles, and entity roles
- `Ability` / `AbilitySet` with cooldowns, mana costs, and effects (damage, heal, speed buff)
- `Gold`, `GoldBounty`, `Level`, `XpBounty` for economy and progression
- Rules engine: `OnDeathRule`, `OnScoreRule`, `TimerRule`, `HealthBelowRule` with custom actions
- `TriggerZone` for area-based events
- `AiBehavior` / `AiGoal` for basic NPC decision-making
- `PlayerCommand` queue and viewport-based input processing

## Usage

```rust
use euca_gameplay::*;

let entity = world.spawn(Health::new(100.0));
world.insert(entity, Team(1));
world.insert(entity, AutoCombat::default());

apply_damage_system(&mut world);
death_check_system(&mut world);
```

## License

MIT
