# neat-flappy

Minimal Flappy Bird simulation with a lightweight NEAT-style evolution loop.

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

This writes `simulation.json` to the repository root.

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
