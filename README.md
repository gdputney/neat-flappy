# Neat Flappy

A minimal Flappy Bird + NEAT-style simulation scaffold.

This repository includes:

- `bird.py` — Bird physics and neural-network inputs.
- `pipe.py` — Pipe obstacle generation and movement.
- `neat_core.py` — `Genome` representation with feedforward activation, mutation, and crossover utilities.
- `main.py` — End-to-end simulation runner, population evolution loop, and JSON export.

## Requirements

- Python 3.10+ (3.11+ recommended)
- No third-party dependencies required

## Quick Start

Run the simulation:

```bash
python main.py
```

This will execute several generations and write `simulation.json` in the repo root.

## What `simulation.json` Contains

Top-level structure:

- `config` — simulation configuration values.
- `generations` — per-generation summary and per-genome simulation results.
- `position_history` — compact timestep position snapshots for each genome run.

Per-genome data includes:

- `fitness` — final score for that simulation.
- `steps_alive` — number of timesteps survived.
- `pipes_passed` — count of successfully passed pipes.
- `crashed` — whether the bird crashed.
- `frames` — detailed frame-by-frame state (`bird`, `pipes`, `alive`, `flap`, etc.).

## Fitness Behavior

In `main.py`, fitness is calculated to:

- Reward survival time.
- Reward passing pipes.
- Penalize crashes.

This encourages agents to stay alive longer while making forward progress.

## Module Notes

### `bird.py`

- `Bird.jump()` applies an upward impulse.
- `Bird.update_physics()` applies gravity and moves the bird.
- `Bird.get_inputs(pipes)` returns normalized inputs for the controller.

### `pipe.py`

- `Pipe` randomizes a vertical gap at initialization.
- `Pipe.update()` moves the pipe left each tick.

### `neat_core.py`

- `Genome.activate(inputs)` performs feedforward sigmoid inference.
- `Genome.mutate()` includes:
  - connection weight perturbation,
  - add-node mutation,
  - add-connection mutation.
- `Genome.crossover(other)` combines parent genes into a child genome.

## Customizing the Simulation

Update `SimulationConfig` in `main.py` to change:

- population size,
- generation count,
- max steps,
- world dimensions,
- pipe spacing.

You can also extend `SimpleGenome` or replace it with your own genome/controller implementation.
