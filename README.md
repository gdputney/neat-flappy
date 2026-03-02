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
- `--csv` to additionally save `fitness.csv` in the run folder.
- `--plot` to save `fitness_over_generations.png` in the run folder (requires `matplotlib`).
- `--replay runs/run_<timestamp>/best_genome.json` to replay a single bird with a saved best genome.
- `--record-replay` to write `web/simulation.json` from a best-genome replay using fixed-`dt` frames.

Example:

```bash
python main.py --seed 7 --csv --plot
```

Per run, the script now writes:

- `simulation.json` in the repository root (full simulation data).
- `runs/run_<timestamp>/stats.json` with best/mean/median fitness per generation.
- `runs/run_<timestamp>/fitness.csv` if `--csv` is used.
- `runs/run_<timestamp>/fitness_over_generations.png` if `--plot` is used.
- `runs/run_<timestamp>/best_genome.json` with the highest-fitness genome seen across all generations.

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

## Visualize a saved genome

Generate a DOT + PNG network visualization from a saved best genome:

```bash
python tools/visualize_genome.py runs/run_<timestamp>/best_genome.json
```

This writes `best_genome.dot` and `best_genome.png` in the current directory by default.

