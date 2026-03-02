"""Run a NEAT Flappy Bird simulation and export run statistics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from bird import Bird
from neat_core import Genome, InnovationTracker
from pipe import Pipe


NETWORK_INPUT_SIZE = 6
NETWORK_OUTPUT_SIZE = 1
FITNESS_CENTERING_PENALTY_SCALE = 0.2


@dataclass
class SimulationConfig:
    population_size: int = 40
    generations: int = 10
    max_steps: int = 300
    world_width: float = 500.0
    world_height: float = 800.0
    pipe_spacing: float = 220.0
    seed: int | None = None
    compatibility_threshold: float = 1.2
    target_species: int = 5
    compatibility_adjust_step: float = 0.05
    min_compatibility_threshold: float = 0.3


def create_initial_genome(input_size: int, output_size: int, tracker: InnovationTracker) -> Genome:
    node_genes = []
    for node_id in range(input_size):
        node_genes.append({"id": node_id, "type": "input", "bias": 0.0})

    output_start = input_size
    for node_id in range(output_start, output_start + output_size):
        node_genes.append({"id": node_id, "type": "output", "bias": random.uniform(-1.0, 1.0)})

    connection_genes = []
    for input_id in range(input_size):
        for output_id in range(output_start, output_start + output_size):
            connection_genes.append(
                {
                    "in_node": input_id,
                    "out_node": output_id,
                    "weight": random.uniform(-1.0, 1.0),
                    "enabled": True,
                    "innovation": tracker.get_connection_innovation(input_id, output_id),
                }
            )

    return Genome(node_genes=node_genes, connection_genes=connection_genes)


def create_population(size: int, tracker: InnovationTracker) -> list[Genome]:
    return [
        create_initial_genome(input_size=NETWORK_INPUT_SIZE, output_size=NETWORK_OUTPUT_SIZE, tracker=tracker)
        for _ in range(size)
    ]


def bird_hits_pipe(bird: Bird, pipe: Pipe) -> bool:
    within_x = pipe.x <= bird.x <= (pipe.x + pipe.width)
    if not within_x:
        return False
    return not (pipe.top <= bird.y <= pipe.bottom)


def normalized_gap_center_distance(bird: Bird, pipes: list[Pipe], world_height: float) -> float:
    """Return normalized absolute distance from bird y to nearest upcoming gap center."""
    if not pipes:
        return 0.0

    ahead_pipes = [pipe for pipe in pipes if (pipe.x + pipe.width) >= bird.x]
    next_pipe = min(ahead_pipes or pipes, key=lambda pipe: pipe.x)
    gap_center = (next_pipe.top + next_pipe.bottom) / 2.0
    distance = abs(bird.y - gap_center)
    height = world_height if world_height > 0 else 1.0
    return min(distance / height, 1.0)


def simulate_genome(genome: Genome, config: SimulationConfig) -> dict[str, Any]:
    bird = Bird(world_width=config.world_width, world_height=config.world_height)
    pipes = [Pipe(x=config.world_width + 100.0, world_height=config.world_height)]
    passed_ids: set[int] = set()
    frames: list[dict[str, Any]] = []
    alive = True
    steps = 0
    pipes_passed = 0
    crashed = False
    centering_penalty = 0.0

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

        centering_penalty += FITNESS_CENTERING_PENALTY_SCALE * normalized_gap_center_distance(
            bird,
            pipes,
            config.world_height,
        )

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

    steps_survived = float(steps)
    pipe_reward = float(pipes_passed * 5000.0)
    alive_bonus = 25.0 if not crashed else 0.0
    genome.fitness = pipe_reward + steps_survived + alive_bonus - centering_penalty
    return {
        "fitness": genome.fitness,
        "steps_alive": steps,
        "pipes_passed": pipes_passed,
        "frames": frames,
        "crashed": crashed,
    }


def speciate_population(population: list[Genome], threshold: float) -> list[list[Genome]]:
    species: list[list[Genome]] = []
    for genome in population:
        placed = False
        for group in species:
            representative = group[0]
            if genome.compatibility_distance(representative) <= threshold:
                group.append(genome)
                placed = True
                break
        if not placed:
            species.append([genome])
    return species


def adjust_compatibility_threshold(
    threshold: float,
    species_count: int,
    target_species: int,
    step: float,
    min_threshold: float,
) -> float:
    if species_count < target_species:
        return max(min_threshold, threshold - step)
    if species_count > target_species:
        return threshold + step
    return threshold


def evolve_population(population: list[Genome], tracker: InnovationTracker, config: SimulationConfig) -> list[Genome]:
    species = speciate_population(population, config.compatibility_threshold)

    elite_count = min(max(1, len(population) // 8), 5, len(population))
    global_elites = sorted(population, key=lambda genome: genome.fitness, reverse=True)[:elite_count]

    adjusted_fitness: dict[int, float] = {}
    for group in species:
        group_size = max(len(group), 1)
        for genome in group:
            adjusted_fitness[id(genome)] = genome.fitness / group_size

    total_adjusted = sum(adjusted_fitness.values()) or 1.0
    next_population: list[Genome] = [copy.deepcopy(genome) for genome in global_elites]

    target_non_elite_count = max(0, len(population) - len(next_population))
    generated_non_elites = 0

    for group in species:
        if generated_non_elites >= target_non_elite_count:
            break

        group_sorted = sorted(group, key=lambda genome: genome.fitness, reverse=True)
        group_adjusted = sum(adjusted_fitness[id(genome)] for genome in group)
        offspring_quota = int(round((group_adjusted / total_adjusted) * target_non_elite_count))
        offspring_quota = max(1, offspring_quota)

        selection_pool = group_sorted[: max(2, len(group_sorted) // 2)]
        for _ in range(offspring_quota):
            if generated_non_elites >= target_non_elite_count:
                break
            parent_a = random.choice(selection_pool)
            parent_b = random.choice(selection_pool)
            child = parent_a.crossover(parent_b)
            child.mutate(tracker)
            next_population.append(child)
            generated_non_elites += 1

    while len(next_population) < len(population):
        parent = random.choice(population)
        child = copy.deepcopy(parent)
        child.fitness = 0.0
        child.mutate(tracker)
        next_population.append(child)

    return next_population[: len(population)]


def serialize_genome(genome: Genome) -> dict[str, Any]:
    return {
        "fitness": genome.fitness,
        "node_genes": copy.deepcopy(genome.node_genes),
        "connection_genes": copy.deepcopy(genome.connection_genes),
    }


def deserialize_genome(data: dict[str, Any]) -> Genome:
    return Genome(
        fitness=float(data.get("fitness", 0.0)),
        node_genes=copy.deepcopy(data.get("node_genes", [])),
        connection_genes=copy.deepcopy(data.get("connection_genes", [])),
    )


def run_simulation(config: SimulationConfig) -> dict[str, Any]:
    if config.seed is not None:
        random.seed(config.seed)

    tracker = InnovationTracker()
    population = create_population(config.population_size, tracker)
    output: dict[str, Any] = {
        "config": asdict(config),
        "generations": [],
        "position_history": {},
    }

    best_genome: Genome | None = None
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
        steps_alive = [result["steps_alive"] for result in generation_results]
        pipes_passed = [result["pipes_passed"] for result in generation_results]
        best_fitness = max(fitnesses)
        mean_fitness = sum(fitnesses) / len(fitnesses)
        median_fitness = median(fitnesses)
        best_steps = max(steps_alive)
        best_pipes_passed = max(pipes_passed)
        mean_pipes_passed = sum(pipes_passed) / len(pipes_passed)
        species_count = len(speciate_population(population, config.compatibility_threshold))
        threshold_used = config.compatibility_threshold
        next_threshold = adjust_compatibility_threshold(
            threshold=threshold_used,
            species_count=species_count,
            target_species=config.target_species,
            step=config.compatibility_adjust_step,
            min_threshold=config.min_compatibility_threshold,
        )

        print(
            f"Generation {generation_index + 1}/{config.generations}: "
            f"best={best_fitness:.2f} mean={mean_fitness:.2f} "
            f"median={median_fitness:.2f} species={species_count} "
            f"threshold={threshold_used:.2f}->{next_threshold:.2f} "
            f"best_steps={best_steps} best_pipes_passed={best_pipes_passed} "
            f"mean_pipes_passed={mean_pipes_passed:.2f}"
        )

        output["generations"].append(
            {
                "generation": generation_index,
                "best_fitness": best_fitness,
                "mean_fitness": mean_fitness,
                "median_fitness": median_fitness,
                "species_count": species_count,
                "compatibility_threshold": threshold_used,
                "next_compatibility_threshold": next_threshold,
                "best_steps": best_steps,
                "best_pipes_passed": best_pipes_passed,
                "mean_pipes_passed": mean_pipes_passed,
                "genomes": generation_results,
            }
        )
        config.compatibility_threshold = next_threshold
        population = evolve_population(population, tracker, config)

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
            "species_count": generation.get("species_count", 0),
            "compatibility_threshold": generation.get("compatibility_threshold", 0.0),
            "next_compatibility_threshold": generation.get("next_compatibility_threshold", 0.0),
            "best_steps": generation.get("best_steps", 0),
            "best_pipes_passed": generation.get("best_pipes_passed", 0),
            "mean_pipes_passed": generation.get("mean_pipes_passed", 0.0),
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
                fieldnames=[
                    "generation",
                    "best_fitness",
                    "mean_fitness",
                    "median_fitness",
                    "species_count",
                    "compatibility_threshold",
                    "next_compatibility_threshold",
                    "best_steps",
                    "best_pipes_passed",
                    "mean_pipes_passed",
                ],
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
