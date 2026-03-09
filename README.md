# neat-flappy

Minimal Flappy Bird simulation with a NEAT evolution loop (node/connection genes, innovation numbers, structural mutations, feedforward topological evaluation, and speciation with fitness sharing).

## Requirements

- Python 3.10+

## Fresh machine setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python main.py
```

Useful options:

- `--seed <int>` for deterministic runs.
- `--deterministic-pipes` to force the same per-generation pipe sequence across all genomes.
- `--flap-policy {probabilistic,hysteresis,deterministic}` to choose stochastic vs thresholded flap control.
- `--population-size <int>` to set the number of genomes per generation (default `100`).
- `--generations <int>` to set how many generations to evolve (default `10`).
- `--max-steps <int>` to cap episode length by simulation steps (default `5000`).
- `--max-pipes <int>` to cap episode length by pipes passed; when set, this takes precedence and step-count termination is disabled.
- `--target-species <int>` and `--compatibility-adjust-step <float>` to tune speciation threshold adaptation (`8` and `0.02` by default).
- `--csv` to additionally save `fitness.csv` in the run folder.
- `--enable-curriculum` to turn on milestone-based difficulty progression.
- `--curriculum-mode {global,species}` to choose whether milestones are driven by global champion best pipes or species champions.
- `--curriculum-milestones`, `--curriculum-gap-deltas`, `--curriculum-speed-deltas`, and `--curriculum-spacing-deltas` to define curriculum thresholds and per-level environment changes.
- `--plot` to save `fitness_over_generations.png` in the run folder (requires `matplotlib`).
- `--replay runs/run_<timestamp>/best_genome.json` to replay a single bird with a saved best genome.
- `--record-training-replay` to write `web/training_replay.json` for the web viewer.
- `--save-simulation-json` to additionally write `simulation.json` in the repository root.
- `--json-pretty` to write human-readable indented JSON (compact JSON is the default for faster/smaller output).

Example:

```bash
python main.py --seed 7 --csv --plot --deterministic-pipes --flap-policy deterministic

# Optional: also persist full simulation data at ./simulation.json
python main.py --seed 7 --save-simulation-json
```

Per run, the script now writes:

For long training runs, keep the default compact JSON mode to reduce file size and speed up disk writes. Use `--json-pretty` when you want human-readable indentation.

- `runs/run_<timestamp>/stats.json` with best/mean/median fitness per generation.
- `runs/run_<timestamp>/fitness.csv` if `--csv` is used.
- `runs/run_<timestamp>/fitness_over_generations.png` if `--plot` is used.
- `runs/run_<timestamp>/best_genome.json` with the highest-fitness genome seen across all generations.

When `--save-simulation-json` is enabled, the script also writes `simulation.json` in the repository root.


## Recommended settings to break the ~5-pipe plateau

Use these training knobs to reduce evaluation noise and stabilize species dynamics:

- `--deterministic-pipes`
- `--population-size 100`
- `--target-species 8`
- `--compatibility-adjust-step 0.02`
- `--flap-policy deterministic` (optional but useful for lower action noise)

Suggested training command:

```bash
python main.py --seed 7 --deterministic-pipes --population-size 100 --target-species 8 --compatibility-adjust-step 0.02 --flap-policy deterministic --enable-curriculum --curriculum-mode global --curriculum-milestones 10,25,50,100 --curriculum-gap-deltas 2,5,10,18 --curriculum-speed-deltas 0.05,0.10,0.20,0.35 --curriculum-spacing-deltas 0,0,5,10 --csv --plot
```

Outputs are written to:

- `runs/run_<timestamp>/stats.json` (includes per-generation `best_pipes_passed`, `mean_pipes_passed`, and curriculum telemetry fields)
- `runs/run_<timestamp>/best_genome.json`
- `runs/run_<timestamp>/fitness.csv` (if `--csv`)
- `runs/run_<timestamp>/fitness_over_generations.png` (if `--plot`)

## Input normalization + shaping knobs

The network inputs are now explicitly normalized through `normalize_inputs(...)` in `main.py` so every feature remains in bounded ranges (mostly `[-1, 1]`, with forward distance in `[0, 1]`). This keeps activation scales stable and helps structural mutations stay useful across generations.

Shaping is also now positive and active *before* the first passed pipe, with two configurable components:

- Centering reward (`--centering-reward-scale`, disable with `--disable-centering-reward`):
  - `k_center * (1 - |gap_error_norm|)` (clamped via `--abs-gap-error-clamp`)
- Progress reward (`--progress-reward-scale`, disable with `--disable-progress-reward`):
  - reward for improving normalized gap alignment from one step to the next
  - per-step delta is clamped by `--progress-reward-clamp` to prevent jitter exploitation

Mutation structure growth is tuned with:

- `--mutation-add-connection-prob`
- `--mutation-add-node-prob`
- `--mutation-toggle-connection-prob`

These defaults are set to encourage hidden structure growth without destabilizing species.

## Quick validation

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("simulation.json")
print("exists:", path.exists())
print("tip: run with --save-simulation-json to create this file")
if path.exists():
    data = json.loads(path.read_text())
    print("top-level keys:", list(data.keys()))
    print("generations:", len(data.get("generations", [])))
PY
```

Replay a saved best genome:

```bash
python main.py --replay runs/run_<timestamp>/best_genome.json --seed 7
```

Generate a training replay for the web viewer:

```bash
python main.py --record-training-replay --replay-top-k 30
```

## Web training replay viewer

Serve from the repo root, then open the viewer:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/web/`. The viewer loads exactly `./training_replay.json` (relative to `/web/`), so the replay file must exist at `web/training_replay.json`.

Viewer controls include:

- generation scrubber (by index)
- play/pause
- speed control
- sequential autoplay across generations
- Show many birds
- Show Brain (rank 1 only; output is shown as `(not recorded)` if absent in frame data)

## Visualize a saved genome

Generate a DOT + PNG network visualization from a saved best genome:

```bash
python tools/visualize_genome.py runs/run_<timestamp>/best_genome.json
```

This writes `best_genome.dot` and `best_genome.png` in the current directory by default.
