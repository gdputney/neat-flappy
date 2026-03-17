# NEAT Flappy: Project & Process Report

## 1) Project Overview

**NEAT Flappy** is a Python project that evolves neural networks to play a Flappy Bird-style environment using a custom implementation of **NEAT** (NeuroEvolution of Augmenting Topologies).  
The project combines:
- a lightweight game simulation,
- an in-house NEAT evolutionary core,
- structured run artifact export (JSON/CSV/plots), and
- browser-based replay visualization.

For a portfolio context, this project demonstrates end-to-end ML/AI engineering: algorithm implementation, experiment design, reproducibility controls, metrics instrumentation, and result communication.

---

## 2) Problem Statement & Goals

### Problem
Train an autonomous agent that learns flap timing to survive and pass pipes in a continuously scrolling obstacle course.

### Goals
1. Build a **fully inspectable NEAT pipeline** (genome representation, mutation, crossover, speciation, selection).
2. Keep simulation deterministic when needed for repeatable experiments.
3. Export training outcomes in formats suitable for analysis and presentation.
4. Provide replay tooling for non-technical audiences (portfolio/demo viewers).

### Success Criteria
- Improvement in best/mean fitness across generations.
- Increasing pipes passed by champions.
- Stable species diversity and threshold adaptation.
- Reproducible results with fixed seed + deterministic pipe schedule.

---

## 3) System Architecture

### Core modules
- **`main.py`**
  - Simulation loop, evaluation, training orchestration, CLI, artifact writing.
  - Defines `SimulationConfig` and most runtime controls.
- **`neat_core.py`**
  - NEAT abstractions (`Genome`, `InnovationTracker`) with mutation, crossover, compatibility distance, and cycle-safe activation.
- **`bird.py`** / **`pipe.py`**
  - Environment entities and physics/collision primitives.
- **`web/index.html` + `web/viewer.js`**
  - Browser viewer for training replay and evolution artifacts.
- **`tools/visualize_genome.py`**
  - Utility for structural genome visualization.
- **`tests/`**
  - Unit/integration coverage for NEAT invariants, replay exports, stats integrity, schema expectations, and deterministic behavior.

### Data flow (high-level)
1. Initialize config and population.
2. Evaluate each genome in simulation.
3. Compute fitness + behavior metrics.
4. Speciate and reproduce next generation.
5. Persist run artifacts and optional web payloads.
6. Replay/visualize best genomes and generation traces.

---

## 4) Technical Implementation Details

### 4.1 Genome representation (NEAT)
Genomes contain:
- **Node genes**: input/hidden/output nodes with bias.
- **Connection genes**: directed weighted edges with innovation IDs + enabled/disabled state.

`InnovationTracker` ensures globally consistent innovation numbering for structural mutations.

### 4.2 Network execution
`Genome.activate()` performs feedforward evaluation over enabled edges using a topological ordering. The implementation includes safeguards that detect and repair enabled cycles by disabling problematic edges, preserving DAG execution for inference.

### 4.3 Mutation operators
The project supports:
- weight perturbation/reset,
- connection enable/disable toggling,
- add-connection mutation (cycle-aware),
- add-node mutation (splitting edges).

Mutation probabilities are exposed in CLI/config, allowing controlled topology growth.

### 4.4 Crossover & compatibility
- Crossover aligns genes by innovation numbers and combines parents into child genomes.
- Compatibility distance uses excess/disjoint/matching-weight components.
- This metric drives speciation and fitness sharing dynamics.

### 4.5 Simulation and control policy
A bird is evaluated in a side-scrolling world with configurable width/height, pipe speed, gap, and spacing. Policy output is interpreted via selectable flap modes:
- probabilistic,
- hysteresis,
- deterministic thresholding.

### 4.6 Input normalization and shaping
The simulation includes explicit normalized inputs and reward shaping terms to stabilize early learning. Additional configurable signals include centering and progress rewards, with clamps to avoid exploitative jitter.

### 4.7 Curriculum learning
Optional curriculum mode progressively increases difficulty based on milestone thresholds (global or species-driven), adjusting:
- pipe gap,
- pipe speed,
- pipe spacing.

This supports smoother long-horizon performance gains compared to static-difficulty training.

---

## 5) Experimentation Process

### Phase 1: Baseline implementation
- Built minimal NEAT + Flappy simulation loop.
- Verified basic evolution and artifact outputs.

### Phase 2: Stability & reproducibility
- Added seed handling and deterministic pipe schedule options.
- Added structured metrics per generation.
- Introduced safety checks around topology/cycle behavior.

### Phase 3: Learning-quality improvements
- Added input normalization and richer shaping signals.
- Exposed mutation/speciation knobs for tuning.
- Added curriculum controls for difficulty progression.

### Phase 4: Reporting & visualization
- Added run directories and summary files (`stats.json`, `best_genome.json`, optional CSV/plot).
- Added web exports (`training_replay.json`, `evolution.json`) and browser viewer.

---

## 6) Outputs & Artifacts

Per run, the pipeline can produce:
- **`runs/run_<timestamp>/stats.json`**: generation-level metrics.
- **`runs/run_<timestamp>/best_genome.json`**: strongest genome snapshot.
- **`runs/run_<timestamp>/fitness.csv`** (optional): tabular fitness history.
- **`runs/run_<timestamp>/fitness_over_generations.png`** (optional): plot output.
- **`simulation.json`** (optional): full simulation data.
- **`web/training_replay.json`** (optional): viewer playback data.
- **`web/evolution.json`** (optional): web-facing evolution summary.

These files are intentionally portfolio-friendly: they support screenshots, charts, and narrative evidence.

---

## 7) Testing & Quality Assurance

The test suite covers multiple reliability dimensions:
- cycle safety in structural mutation and activation,
- deterministic behavior for seeded runs,
- export schema consistency,
- simulation stats integrity,
- replay payload correctness for web tooling.

This emphasizes not just algorithm correctness, but also confidence in reproducible and communicable results.

---

## 8) Portfolio Report Framing (recommended)

Use this project in a portfolio under a structure like:

1. **Context**: Why neuroevolution instead of gradient-based RL for this problem.
2. **Approach**: Brief explanation of NEAT and custom implementation choices.
3. **Engineering highlights**:
   - deterministic experiment controls,
   - modular architecture,
   - artifact pipeline and viewer integration,
   - robust test coverage.
4. **Results**:
   - fitness trajectory chart,
   - best pipes passed over generations,
   - screenshot/GIF from web replay.
5. **Reflection**:
   - what worked (speciation + shaping + deterministic pipes),
   - what constrained progress,
   - what you’d improve next.

---

## 9) Key Challenges & Resolutions

- **Challenge:** Evolution noise and unstable progress.  
  **Resolution:** deterministic evaluation modes, policy control variants, and curriculum milestones.

- **Challenge:** Topological mutation can introduce invalid cyclic graphs.  
  **Resolution:** cycle-aware mutation logic + repair mechanisms during activation.

- **Challenge:** Communicating training behavior to non-technical viewers.  
  **Resolution:** replay/evolution JSON exports and a web viewer for visual storytelling.

---

## 10) Future Improvements

1. Multi-seed benchmark harness with confidence intervals.
2. Better species-level dashboards (lineage trees, innovation adoption rates).
3. Hyperparameter sweeps + automatic run comparison reports.
4. Real-time visualization during training.
5. Optional parallel evaluation backends for larger populations.

---

## 11) Reproducible Run Recipe

```bash
python main.py \
  --seed 7 \
  --deterministic-pipes \
  --population-size 100 \
  --target-species 8 \
  --compatibility-adjust-step 0.02 \
  --flap-policy deterministic \
  --enable-curriculum \
  --curriculum-mode global \
  --curriculum-milestones 10,25,50,100 \
  --curriculum-gap-deltas 2,5,10,18 \
  --curriculum-speed-deltas 0.05,0.10,0.20,0.35 \
  --curriculum-spacing-deltas 0,0,5,10 \
  --csv --plot
```

Then serve web outputs:

```bash
python -m http.server 8000
# open /web/ in a browser
```

---

## 12) One-Paragraph Portfolio Summary (copy-ready)

I built a full NEAT-based Flappy Bird training system in Python from scratch, including genome evolution, structural mutations with innovation tracking, speciation, deterministic experiment controls, curriculum learning, and exportable analytics artifacts. Beyond model training, I designed the project for communication and reproducibility: each run emits structured stats, best-genome snapshots, optional CSV/plots, and browser-ready replay data rendered by a custom web viewer. The project demonstrates my ability to combine AI algorithm design, software architecture, testing, and presentation tooling into a coherent, production-style experimentation workflow.
