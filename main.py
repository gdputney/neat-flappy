"""Run a simple NEAT-style Flappy Bird simulation and export run statistics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from bird import Bird
from neat_core import Genome
from pipe import Pipe


@dataclass
class SimpleGenome(Genome):
    """Minimal concrete genome implementation used by this simulation script."""

    weights: list[float] = field(default_factory=list)
    bias: float = 0.0

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [random.uniform(-1.0, 1.0) for _ in range(5)]

    def activate(self, inputs: list[float]) -> list[float]:
        total = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        output = 1.0 / (1.0 + (2.718281828 ** (-total)))
        return [output]

    def mutate(self) -> None:
        for index in range(len(self.weights)):
            if random.random() < 0.3:
                self.weights[index] += random.uniform(-0.4, 0.4)
        if random.random() < 0.3:
            self.bias += random.uniform(-0.4, 0.4)

    def crossover(self, other: "Genome") -> "Genome":
        if not isinstance(other, SimpleGenome):
            raise TypeError("SimpleGenome can only crossover with SimpleGenome")

        child_weights = [random.choice([a, b]) for a, b in zip(self.weights, other.weights)]
        child_bias = random.choice([self.bias, other.bias])
        return SimpleGenome(
            node_genes=list(self.node_genes),
            connection_genes=list(self.connection_genes),
            weights=child_weights,
            bias=child_bias,
        )


@dataclass
class SimulationConfig:
    population_size: int = 20
    generations: int = 5
    max_steps: int = 300
    world_width: float = 500.0
    world_height: float = 800.0
    pipe_spacing: float = 220.0
    seed: int | None = None


def create_population(size: int) -> list[SimpleGenome]:
    return [SimpleGenome() for _ in range(size)]


def bird_hits_pipe(bird: Bird, pipe: Pipe) -> bool:
    within_x = pipe.x <= bird.x <= (pipe.x + pipe.width)
    if not within_x:
        return False
    return not (pipe.top <= bird.y <= pipe.bottom)


def simulate_genome(genome: SimpleGenome, config: SimulationConfig) -> dict[str, Any]:
    bird = Bird(world_width=config.world_width, world_height=config.world_height)
    pipes = [Pipe(x=config.world_width + 100.0, world_height=config.world_height)]
    passed_ids: set[int] = set()
    frames: list[dict[str, Any]] = []
    alive = True
    steps = 0
    pipes_passed = 0
    crashed = False

    while alive and steps < config.max_steps:
        if pipes[-1].x < config.world_width - config.pipe_spacing:
            pipes.append(Pipe(x=config.world_width + 50.0, world_height=config.world_height))

        inputs = bird.get_inputs(pipes)
        flap = genome.activate(inputs)[0] > 0.5
        if flap:
            bird.jump()

        bird.update_physics()
        for pipe in pipes:
            pipe.update()

        pipes = [pipe for pipe in pipes if (pipe.x + pipe.width) > -5]

        if bird.y < 0 or bird.y > config.world_height:
            alive = False
            crashed = True

        for pipe in pipes:
            if bird_hits_pipe(bird, pipe):
                alive = False
                crashed = True
            if (pipe.x + pipe.width) < bird.x and id(pipe) not in passed_ids:
                passed_ids.add(id(pipe))
                pipes_passed += 1

        frames.append(
            {
                "step": steps,
                "bird": {"x": bird.x, "y": bird.y, "velocity": bird.velocity},
                "pipes": [
                    {
                        "x": pipe.x,
                        "width": pipe.width,
                        "top": pipe.top,
                        "bottom": pipe.bottom,
                    }
                    for pipe in pipes
                ],
                "alive": alive,
                "flap": flap,
                "pipes_passed": pipes_passed,
            }
        )
        steps += 1

    survival_reward = float(steps * 1.0)
    pipe_reward = float(pipes_passed * 150.0)
    crash_penalty = 50.0 if crashed else 0.0
    genome.fitness = max(0.0, survival_reward + pipe_reward - crash_penalty)
    return {
        "fitness": genome.fitness,
        "steps_alive": steps,
        "pipes_passed": pipes_passed,
        "frames": frames,
        "crashed": crashed,
    }


def evolve_population(population: list[SimpleGenome]) -> list[SimpleGenome]:
    population = sorted(population, key=lambda genome: genome.fitness, reverse=True)
    elite_count = max(2, len(population) // 5)
    elites = population[:elite_count]

    next_population: list[SimpleGenome] = [
        SimpleGenome(
            node_genes=list(genome.node_genes),
            connection_genes=list(genome.connection_genes),
            weights=list(genome.weights),
            bias=genome.bias,
        )
        for genome in elites
    ]

    selection_pool = population[: max(2, len(population) // 2)]
    while len(next_population) < len(population):
        parent_a = random.choice(selection_pool)
        parent_b = random.choice(selection_pool)
        child = parent_a.crossover(parent_b)
        child.mutate()
        next_population.append(child)

    return next_population[: len(population)]


def serialize_genome(genome: SimpleGenome) -> dict[str, Any]:
    return {
        "weights": list(genome.weights),
        "bias": genome.bias,
        "fitness": genome.fitness,
        "node_genes": copy.deepcopy(genome.node_genes),
        "connection_genes": copy.deepcopy(genome.connection_genes),
    }


def deserialize_genome(data: dict[str, Any]) -> SimpleGenome:
    return SimpleGenome(
        weights=[float(weight) for weight in data.get("weights", [])],
        bias=float(data.get("bias", 0.0)),
        fitness=float(data.get("fitness", 0.0)),
        node_genes=copy.deepcopy(data.get("node_genes", [])),
        connection_genes=copy.deepcopy(data.get("connection_genes", [])),
    )


def run_simulation(config: SimulationConfig) -> dict[str, Any]:
    if config.seed is not None:
        random.seed(config.seed)

    population = create_population(config.population_size)
    output: dict[str, Any] = {
        "config": asdict(config),
        "generations": [],
        "position_history": {},
    }

    best_genome: SimpleGenome | None = None
    best_generation = -1
    best_genome_index = -1

    for generation_index in range(config.generations):
        generation_results = []
        for genome_index, genome in enumerate(population):
            result = simulate_genome(genome, config)
            generation_results.append({"genome_index": genome_index, **result})
            output["position_history"][f"g{generation_index}_genome{genome_index}"] = {
                str(frame["step"]): {
                    "bird": frame["bird"],
                    "pipes": frame["pipes"],
                }
                for frame in result["frames"]
            }
            if best_genome is None or genome.fitness > best_genome.fitness:
                best_genome = copy.deepcopy(genome)
                best_generation = generation_index
                best_genome_index = genome_index

        fitnesses = [result["fitness"] for result in generation_results]
        best_fitness = max(fitnesses)
        mean_fitness = sum(fitnesses) / len(fitnesses)
        median_fitness = median(fitnesses)

        print(
            f"Generation {generation_index + 1}/{config.generations}: "
            f"best={best_fitness:.2f} mean={mean_fitness:.2f} median={median_fitness:.2f}"
        )

        output["generations"].append(
            {
                "generation": generation_index,
                "best_fitness": best_fitness,
                "mean_fitness": mean_fitness,
                "median_fitness": median_fitness,
                "genomes": generation_results,
            }
        )
        population = evolve_population(population)

    output["best_genome"] = {
        "generation": best_generation,
        "genome_index": best_genome_index,
        "genome": serialize_genome(best_genome) if best_genome is not None else None,
    }
    return output


def write_stats(simulation_data: dict[str, Any], run_dir: Path, save_csv: bool) -> None:
    generation_stats = [
        {
            "generation": generation["generation"],
            "best_fitness": generation["best_fitness"],
            "mean_fitness": generation["mean_fitness"],
            "median_fitness": generation["median_fitness"],
        }
        for generation in simulation_data["generations"]
    ]

    stats = {
        "config": simulation_data["config"],
        "generation_stats": generation_stats,
    }

    stats_path = run_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    if save_csv:
        csv_path = run_dir / "fitness.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["generation", "best_fitness", "mean_fitness", "median_fitness"],
            )
            writer.writeheader()
            writer.writerows(generation_stats)


def write_best_genome(simulation_data: dict[str, Any], run_dir: Path) -> Path | None:
    best_genome = simulation_data.get("best_genome", {}).get("genome")
    if best_genome is None:
        return None

    payload = {
        "config": simulation_data.get("config", {}),
        "best_genome": simulation_data["best_genome"],
    }
    best_genome_path = run_dir / "best_genome.json"
    with best_genome_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return best_genome_path


def write_plot(simulation_data: dict[str, Any], run_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("--plot requested, but matplotlib is not installed; skipping plot output.")
        return None

    generations = [generation["generation"] for generation in simulation_data["generations"]]
    best = [generation["best_fitness"] for generation in simulation_data["generations"]]
    mean = [generation["mean_fitness"] for generation in simulation_data["generations"]]
    med = [generation["median_fitness"] for generation in simulation_data["generations"]]

    plt.figure(figsize=(8, 4.5))
    plt.plot(generations, best, marker="o", label="Best")
    plt.plot(generations, mean, marker="o", label="Mean")
    plt.plot(generations, med, marker="o", label="Median")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over generations")
    plt.grid(alpha=0.3)
    plt.legend()

    output_path = run_dir / "fitness_over_generations.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def load_genome_payload(genome_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    with genome_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if "best_genome" in payload:
        meta = payload.get("best_genome")
        genome_data = payload.get("best_genome", {}).get("genome")
    else:
        meta = None
        genome_data = payload

    if not isinstance(genome_data, dict):
        raise ValueError(f"Could not find genome data in replay file: {genome_path}")

    return genome_data, meta


def replay_genome(genome_data: dict[str, Any], config: SimulationConfig) -> dict[str, Any]:
    genome = deserialize_genome(genome_data)
    if config.seed is not None:
        random.seed(config.seed)
    return simulate_genome(genome, config)


def write_record_replay(
    replay_result: dict[str, Any],
    generation_metadata: dict[str, Any],
    output_path: Path,
    dt: float = 1.0 / 60.0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in replay_result["frames"]:
        frames.append(
            {
                "t": frame["step"] * dt,
                "dt": dt,
                "bird": {
                    "y": frame["bird"]["y"],
                    "vel": frame["bird"]["velocity"],
                },
                "pipes": [
                    {"x": pipe["x"], "top": pipe["top"], "bottom": pipe["bottom"]}
                    for pipe in frame["pipes"]
                ],
                "score": frame["pipes_passed"],
                "generation": generation_metadata,
            }
        )

    data = {
        "meta": {
            "dt": dt,
            "total_frames": len(frames),
            "fitness": replay_result["fitness"],
            "steps_alive": replay_result["steps_alive"],
            "pipes_passed": replay_result["pipes_passed"],
            "crashed": replay_result["crashed"],
            "generation": generation_metadata,
        },
        "frames": frames,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return output_path


def replay_from_genome(genome_path: Path, config: SimulationConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    genome_data, meta = load_genome_payload(genome_path)
    result = replay_genome(genome_data, config)

    print(
        "Replay result: "
        f"fitness={result['fitness']:.2f} "
        f"steps_alive={result['steps_alive']} "
        f"pipes_passed={result['pipes_passed']} "
        f"crashed={result['crashed']}"
    )
    replay_data = {"genome_path": str(genome_path), "result": result}
    generation_metadata = {
        "generation": (meta or {}).get("generation"),
        "genome_index": (meta or {}).get("genome_index"),
        "source": str(genome_path),
    }
    return replay_data, generation_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic runs")
    parser.add_argument("--plot", action="store_true", help="Save a fitness-over-generations PNG")
    parser.add_argument("--csv", action="store_true", help="Also save fitness.csv")
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Path to best_genome.json (or raw genome json) to replay a single bird",
    )
    parser.add_argument(
        "--record-replay",
        action="store_true",
        help="Replay the best genome and write web/simulation.json with fixed-dt frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(seed=args.seed)

    if args.replay is not None:
        replay_data, generation_metadata = replay_from_genome(args.replay, config)
        replay_path = Path(__file__).resolve().parent / "replay.json"
        with replay_path.open("w", encoding="utf-8") as file:
            json.dump(replay_data, file, indent=2)
        print(f"Saved replay output: {replay_path}")
        if args.record_replay:
            web_path = Path(__file__).resolve().parent / "web" / "simulation.json"
            out_path = write_record_replay(replay_data["result"], generation_metadata, web_path)
            print(f"Saved record replay output: {out_path}")
        return

    simulation_data = run_simulation(config)

    output_path = Path(__file__).resolve().parent / "simulation.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(simulation_data, file, indent=2)

    runs_dir = Path(__file__).resolve().parent / "runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_stats(simulation_data, run_dir, save_csv=args.csv)
    best_genome_path = write_best_genome(simulation_data, run_dir)
    plot_path = write_plot(simulation_data, run_dir) if args.plot else None

    if args.record_replay and best_genome_path is not None:
        replay_data, generation_metadata = replay_from_genome(best_genome_path, config)
        web_path = Path(__file__).resolve().parent / "web" / "simulation.json"
        out_path = write_record_replay(replay_data["result"], generation_metadata, web_path)
        print(f"Saved record replay output: {out_path}")

    print(f"Saved simulation output: {output_path}")
    print(f"Saved run stats: {run_dir / 'stats.json'}")
    if best_genome_path is not None:
        print(f"Saved best genome: {best_genome_path}")
    if args.csv:
        print(f"Saved fitness CSV: {run_dir / 'fitness.csv'}")
    if plot_path is not None:
        print(f"Saved fitness plot: {plot_path}")


if __name__ == "__main__":
    main()
