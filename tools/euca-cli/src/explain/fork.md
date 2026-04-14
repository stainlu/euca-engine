# Fork — Counterfactual Simulation

A **fork** is a deep copy of the main world that evolves independently.
Use forks to answer "what if?" questions: spawn a fork, apply an
intervention, advance the simulation, observe, compare to main, then
drop the fork. The main world is never touched.

Every fork shares the same Schedule as the main world, so stepping a
fork runs the same systems (physics, combat, AI, rules) that would run
on main. Only the fork's state is mutated.

## Commands

```bash
# Create a named fork (deep-clones main world).
euca fork create scenario-a

# List active forks.
euca fork list

# Advance the fork by N ticks (physics/combat/AI all run on fork only).
euca fork step scenario-a --ticks 300

# Advance the fork AND evaluate assertions in one atomic call.
euca fork probe scenario-a --ticks 300 --assertions hero-alive,team1-wins

# Read the fork's entities at the current tick.
euca fork observe scenario-a

# Drop the fork.
euca fork delete scenario-a
```

## The counterfactual loop

```bash
# 1. Baseline: set up the main world (or keep it as-is).
euca entity create --mesh cube --health 500 --team 1 --combat
euca entity create --mesh cube --health 80  --team 2 --combat

# 2. Snapshot the baseline as a scenario file (see `euca explain scenario`).
euca scenario save --out /tmp/baseline.json

# 3. Create a fork to test a variant.
euca fork create buffed-hero

# 4. Intervene on the fork (e.g. triple the hero's HP via a scenario).
euca scenario apply-to-fork buffed-hero /tmp/buffed.json

# 5. Run the fork forward and check an assertion.
euca fork probe buffed-hero --ticks 600 --assertions team1_wins

# 6. Main world is unchanged — verify:
euca entity list

# 7. Drop the fork.
euca fork delete buffed-hero
```

## When to use forks

- **Balance tuning** — run 50 sims at different parameter values in parallel forks
- **Regression detection** — compare a fork with a known-good baseline
- **Intervention testing** — "what if this entity took 100 damage at tick 500?"
- **Predicate search** — advance a fork until a condition holds

## When NOT to use forks

- You only need to step the live world — use `euca sim step` instead
- You want to persist state across sessions — use `scene save/load` or
  `scenario save/load` (scenarios survive process restarts; forks don't)
