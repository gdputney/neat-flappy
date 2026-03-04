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
- `--eval-episodes <int>` to evaluate each genome over multiple episodes and use mean fitness (default `1`).
- `--deterministic-pipes` to force the same per-generation pipe sequence across all genomes.
- `--flap-policy {probabilistic,hysteresis,deterministic}` to choose stochastic vs thresholded flap control.
- `--population-size <int>` to set the number of genomes per generation (default `100`).
- `--generations <int>` to set how many generations to evolve (default `10`).
- `--target-species <int>` and `--compatibility-adjust-step <float>` to tune speciation threshold adaptation (`8` and `0.02` by default).
- `--csv` to additionally save `fitness.csv` in the run folder.
- `--plot` to save `fitness_over_generations.png` in the run folder (requires `matplotlib`).
- `--replay runs/run_<timestamp>/best_genome.json` to replay a single bird with a saved best genome.
- `--record-replay` to write `web/simulation.json` from a best-genome replay using fixed-`dt` frames.

Example:

```bash
python main.py --seed 7 --csv --plot --eval-episodes 3 --deterministic-pipes --flap-policy deterministic
```

Per run, the script now writes:

- `simulation.json` in the repository root (full simulation data).
- `runs/run_<timestamp>/stats.json` with best/mean/median fitness per generation.
- `runs/run_<timestamp>/fitness.csv` if `--csv` is used.
- `runs/run_<timestamp>/fitness_over_generations.png` if `--plot` is used.
- `runs/run_<timestamp>/best_genome.json` with the highest-fitness genome seen across all generations.


## Recommended settings to break the ~5-pipe plateau

Use these training knobs to reduce evaluation noise and stabilize species dynamics:

- `--eval-episodes 3`
- `--deterministic-pipes`
- `--population-size 100`
- `--target-species 8`
- `--compatibility-adjust-step 0.02`
- `--flap-policy deterministic` (optional but useful for lower action noise)

Suggested training command:

```bash
python main.py --seed 7 --eval-episodes 3 --deterministic-pipes --population-size 100 --target-species 8 --compatibility-adjust-step 0.02 --flap-policy deterministic --csv --plot
```

Outputs are written to:

- `runs/run_<timestamp>/stats.json` (includes per-generation `best_episode_pipes_passed_max` and `best_episode_pipes_passed_mean`)
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

Record replay frames for a web viewer:

```bash
python main.py --seed 7 --record-replay
```

Or from an existing saved best genome:

```bash
python main.py --replay runs/run_<timestamp>/best_genome.json --record-replay
```

## Web replay viewer

After generating `web/simulation.json` with `--record-replay`, open:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/web/` to view the replay on an HTML canvas with:

- Play/Pause control
- Playback speed slider
- Overlay text showing score and generation

## Web evolution viewer

Export evolution data, then serve the repo and open the web app:

```bash
python main.py --export-web-evolution --deterministic-pipes
python -m http.server 8000
```

Open `http://localhost:8000/web/` and use the overlay panel to:

- scrub generations with Prev/Next or the generation slider
- play/pause autoplay across generations
- adjust autoplay interval (`500`-`3000` ms per generation)
- toggle trails, champion-only highlight, and debug overlay labels
- toggle **Show Brain** to inspect the champion's live neural activations and wiring
- monitor generation stats (alive birds, current/all-time best pipes, and deterministic pipe seed)

When **Show Brain** is enabled, the neural panel displays the rank-1 genome (or the single replay bird):

- input vector in Python order:
  1. `y_norm` (bird y centered to `[-1, 1]`)
  2. `velocity_norm` (bird velocity normalized to `[-1, 1]`)
  3. `dx_to_next_pipe_norm` (horizontal distance to next pipe in `[0, 1]`)
  4. `gap_error_norm` (bird offset from gap center in `[-1, 1]`)
  5. `dy_to_gap_top_norm` (delta to gap top in `[-1, 1]`)
  6. `dy_to_gap_bottom_norm` (delta to gap bottom in `[-1, 1]`)
- output activation and current flap decision
- deterministic flap state details (cooldown counter + hysteresis on/off)
- node-link diagram where node brightness tracks live activation and edge color/width encodes weight sign/magnitude

## Visualize a saved genome

Generate a DOT + PNG network visualization from a saved best genome:

```bash
python tools/visualize_genome.py runs/run_<timestamp>/best_genome.json
```

This writes `best_genome.dot` and `best_genome.png` in the current directory by default.
