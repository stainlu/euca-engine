# euca-ai

Behavior trees with blackboard, decorators, and composites for NPC AI.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `BtNode` enum-based recursive tree structure (no trait objects)
- Composite nodes: Sequence, Selector, Parallel (with configurable policy)
- Decorator nodes: Inverter, Repeater, Succeeder, condition guards
- `BtAction` and `BtCondition` leaf nodes with comparisons
- `Blackboard` per-entity typed key-value store (`BlackboardValue`)
- `BehaviorTreeExecutor` component holding tree + blackboard + running state
- `BtBuilder` fluent API for ergonomic tree construction
- `behavior_tree_system` ticks all executors each frame

## Usage

```rust
use euca_ai::*;

let tree = BtBuilder::selector()
    .sequence(|s| s
        .condition(BtCondition::new("health", CompareOp::Less, 20.0.into()))
        .action(BtAction::new("flee")))
    .action(BtAction::new("patrol"))
    .build();

world.insert(entity, BehaviorTreeExecutor::new(tree));
behavior_tree_system(&mut world, dt);
```

## License

MIT
