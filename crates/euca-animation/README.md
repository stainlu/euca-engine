# euca-animation

Runtime skeletal animation: state machines, blend spaces, montages, root motion, and inverse kinematics.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `AnimStateMachine` with parametric transitions and conditions
- `AnimPose` sampling and `AnimationBlender` for multi-layer crossfade blending
- `BlendSpace1D` / `BlendSpace2D` for parametric locomotion blending
- `MontagePlayer` for one-shot overlay animations (attacks, emotes)
- `RootMotionReceiver` for extracting entity-level movement from root bone
- `AnimationEvent` system with time-stamped clip callbacks
- `IkChain` (FABRIK, two-bone) and `LookAtConstraint` for inverse kinematics
- `Animator` component with `animation_evaluate_system` entry point

## Usage

```rust
use euca_animation::*;

let mut sm = AnimStateMachine::new("idle");
sm.add_state(AnimState::new("idle", clip_idle));
sm.add_state(AnimState::new("run", clip_run));
sm.add_transition(StateTransition::new("idle", "run")
    .with_condition(TransitionCondition::new("speed", CompareOp::Greater, ParamValue::Float(0.1))));
```

## License

MIT
