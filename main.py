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
FITNESS_CENTERING_PENALTY_SCALE = 5.0
SHAPING_SCALE = 0.1
FIRST_PIPE_REACHED_BONUS = 200.0
FLAP_COOLDOWN_FRAMES = 8
FLAP_ON_THRESHOLD = 0.8
FLAP_OFF_THRESHOLD = 0.2
VEL_MIN = -12.0
VEL_MAX = 12.0
CEILING_TOUCH_PENALTY = 200.0


@dataclass
class SimulationConfig:
    population_size: int = 40
    generations: int = 10
    max_steps: int = 1000
    world_width: float = 500.0
    world_height: float = 800.0
    pipe_spacing: float = 220.0
    seed: int | None = None
    compatibility_threshold: float = 1.2
    target_species: int = 5
    compatibility_adjust_step: float = 0.05
    min_compatibility_threshold: float = 0.3
    flap_policy: str = "hysteresis"
    shaping_weight: float = 10.0
    shaping_scale: float = SHAPING_SCALE
    flap_cooldown_frames: int = FLAP_COOLDOWN_FRAMES
    flap_on_threshold: float = FLAP_ON_THRESHOLD
    flap_off_threshold: float = FLAP_OFF_THRESHOLD
    velocity_min: float = VEL_MIN
    velocity_max: float = VEL_MAX
    ceiling_touch_penalty: float = CEILING_TOUCH_PENALTY
    debug_no_flap: bool = False
    debug_always_flap: bool = False


def decide_flap(
    output: float,
    policy: str,
    is_flapping: bool = False,
    flap_on_threshold: float = FLAP_ON_THRESHOLD,
    flap_off_threshold: float = FLAP_OFF_THRESHOLD,
) -> bool:
    if policy == "probabilistic":
        flap_probability = max(0.0, min(1.0, output))
        return random.random() < flap_probability
    if policy == "hysteresis":
        if output >= flap_on_threshold:
            return True
        if output <= flap_off_threshold:
            return False
        return is_flapping
    raise ValueError(f"Unknown flap policy: {policy}")


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




def pipe_crossed_bird(previous_x: float, current_x: float, pipe_width: float, bird_x: float) -> bool:
    previous_right_edge = previous_x + pipe_width
    current_right_edge = current_x + pipe_width
    return previous_right_edge >= bird_x and current_right_edge < bird_x

def normalized_gap_center_distance(bird: Bird, pipes: list[Pipe], world_height: float) -> float:
    """Return normalized abs distance from bird y to the next pipe ahead gap center."""
    if not pipes:
        return 0.0

    ahead_pipes = [pipe for pipe in pipes if (pipe.x + pipe.width) >= bird.x]
    if not ahead_pipes:
        return 0.0

    next_pipe = min(ahead_pipes, key=lambda pipe: pipe.x)
    gap_center = (next_pipe.top + next_pipe.bottom) / 2.0
    distance = abs(bird.y - gap_center)
    height = world_height if world_height > 0 else 1.0
    return min(distance / height, 1.0)


def simulate_genome(genome: Genome, config: SimulationConfig) -> dict[str, Any]:
    bird = Bird(
        world_width=config.world_width,
        world_height=config.world_height,
        flap_cooldown_frames=config.flap_cooldown_frames,
        velocity_min=config.velocity_min,
        velocity_max=config.velocity_max,
    )
    first_pipe = Pipe(x=config.world_width + 120.0, world_height=config.world_height)
    pipes = [first_pipe]
    passed_ids: set[int] = set()
    frames: list[dict[str, Any]] = []
    alive = True
    steps = 0
    pipes_passed = 0
    crashed = False
    centering_penalty = 0.0
    average_centering_penalty = 0.0
    is_flapping = False
    reached_first_pipe = False
    ceiling_touches = 0
    death_reason: str | None = None
    death_bird_y: float | None = None
    death_bird_velocity: float | None = None

    while alive and steps < config.max_steps:
        if pipes[-1].x < config.world_width - config.pipe_spacing:
            pipes.append(Pipe(x=config.world_width + 50.0, world_height=config.world_height))

        inputs = bird.get_inputs(pipes)
        output = genome.activate(inputs)[0]
        flap = decide_flap(
            output,
            config.flap_policy,
            is_flapping=is_flapping,
            flap_on_threshold=config.flap_on_threshold,
            flap_off_threshold=config.flap_off_threshold,
        )
        is_flapping = flap
        if config.debug_no_flap:
            flap = False
        elif config.debug_always_flap:
            flap = True
        if flap:
            flap = bird.jump()

        previous_pipe_x = {id(pipe): pipe.x for pipe in pipes}

        bird.update_physics()
        for pipe in pipes:
            pipe.update()

        pipes = [pipe for pipe in pipes if (pipe.x + pipe.width) > -5]

        if bird.y < 0:
            bird.y = 0
            bird.velocity = 0
            ceiling_touches += 1
        elif bird.y >= config.world_height:
            alive = False
            crashed = True
            death_reason = "hit_ground"
            death_bird_y = bird.y
            death_bird_velocity = bird.velocity

        for pipe in pipes:
            if bird_hits_pipe(bird, pipe):
                alive = False
                crashed = True
                if death_reason is None:
                    death_reason = "hit_pipe"
                    death_bird_y = bird.y
                    death_bird_velocity = bird.velocity
            previous_x = previous_pipe_x.get(id(pipe), pipe.x)
            if pipe_crossed_bird(previous_x, pipe.x, pipe.width, bird.x) and id(pipe) not in passed_ids:
                passed_ids.add(id(pipe))
                pipes_passed += 1

        centering_penalty += FITNESS_CENTERING_PENALTY_SCALE * normalized_gap_center_distance(
            bird,
            pipes,
            config.world_height,
        )

        if not reached_first_pipe and first_pipe.x <= bird.x:
            reached_first_pipe = True

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

    if death_reason is None:
        death_reason = "max_steps"
        death_bird_y = bird.y
        death_bird_velocity = bird.velocity

    steps_survived = float(steps)
    pipes_reward = float(pipes_passed * 5000.0)
    reached_first_pipe_bonus = FIRST_PIPE_REACHED_BONUS if reached_first_pipe else 0.0
    shaping_penalty = centering_penalty * config.shaping_weight * config.shaping_scale
    ceiling_penalty = ceiling_touches * config.ceiling_touch_penalty
    average_centering_penalty = (centering_penalty / steps_survived) if steps_survived > 0 else 0.0
    genome.fitness = steps_survived + pipes_reward + reached_first_pipe_bonus - shaping_penalty - ceiling_penalty
    return {
        "fitness": genome.fitness,
        "steps_alive": steps,
        "pipes_passed": pipes_passed,
        "steps": steps_survived,
        "pipes_reward": pipes_reward,
        "shaping_penalty": shaping_penalty,
        "ceiling_penalty": ceiling_penalty,
        "ceiling_touches": ceiling_touches,
        "reached_first_pipe_bonus": reached_first_pipe_bonus,
        "frames": frames,
        "crashed": crashed,
        "centering_penalty": centering_penalty,
        "average_centering_penalty": average_centering_penalty,
        "death_reason": death_reason,
        "death_bird_y": death_bird_y,
        "death_bird_velocity": death_bird_velocity,
        "screen_bounds": {"y_min": 0.0, "y_max": config.world_height},
    }


def run_debug_one_episode(config: SimulationConfig, interval: int = 20) -> dict[str, Any]:
    """Run a single random genome episode and print periodic state snapshots."""
    if config.seed is not None:
        random.seed(config.seed)

    tracker = InnovationTracker()
    genome = create_initial_genome(input_size=NETWORK_INPUT_SIZE, output_size=NETWORK_OUTPUT_SIZE, tracker=tracker)
    bird = Bird(
        world_width=config.world_width,
        world_height=config.world_height,
        flap_cooldown_frames=config.flap_cooldown_frames,
        velocity_min=config.velocity_min,
        velocity_max=config.velocity_max,
    )
    first_pipe = Pipe(x=config.world_width + 120.0, world_height=config.world_height)
    pipes = [first_pipe]
    passed_ids: set[int] = set()
    reached_first_pipe = False
    death_reason: str | None = None
    death_bird_y: float | None = None
    death_bird_velocity: float | None = None
    pipes_passed = 0
    is_flapping = False

    print(f"[debug-one] initial_pipe_spawn_x={first_pipe.x:.2f} initial_pipe_speed={first_pipe.speed:.2f}")

    steps_executed = 0
    for step in range(config.max_steps):
        steps_executed = step + 1
        if pipes[-1].x < config.world_width - config.pipe_spacing:
            pipes.append(Pipe(x=config.world_width + 50.0, world_height=config.world_height))

        ahead_pipes = [pipe for pipe in pipes if (pipe.x + pipe.width) >= bird.x]
        next_pipe = min(ahead_pipes, key=lambda pipe: pipe.x) if ahead_pipes else None
        next_pipe_x = next_pipe.x if next_pipe is not None else float("nan")
        dx_to_next_pipe = next_pipe_x - bird.x if next_pipe is not None else float("nan")
        if step % interval == 0:
            print(
                "[debug-one] "
                f"t={step} "
                f"bird_x={bird.x:.2f} "
                f"bird_y={bird.y:.2f} "
                f"next_pipe_x={next_pipe_x:.2f} "
                f"dx_to_next_pipe={dx_to_next_pipe:.2f} "
                f"reached_first_pipe={reached_first_pipe} "
                f"pipes_passed={pipes_passed}"
            )

        inputs = bird.get_inputs(pipes)
        output = genome.activate(inputs)[0]
        flap = decide_flap(
            output,
            config.flap_policy,
            is_flapping=is_flapping,
            flap_on_threshold=config.flap_on_threshold,
            flap_off_threshold=config.flap_off_threshold,
        )
        is_flapping = flap
        if config.debug_no_flap:
            flap = False
        elif config.debug_always_flap:
            flap = True
        if flap:
            flap = bird.jump()

        previous_pipe_x = {id(pipe): pipe.x for pipe in pipes}
        bird.update_physics()
        for pipe in pipes:
            pipe.update()

        pipes = [pipe for pipe in pipes if (pipe.x + pipe.width) > -5]

        if bird.y < 0:
            bird.y = 0
            bird.velocity = 0
        if bird.y >= config.world_height:
            death_reason = "hit_ground"
            death_bird_y = bird.y
            death_bird_velocity = bird.velocity
            break

        crashed_into_pipe = False
        for pipe in pipes:
            if bird_hits_pipe(bird, pipe):
                crashed_into_pipe = True
                death_reason = "hit_pipe"
                death_bird_y = bird.y
                death_bird_velocity = bird.velocity
            previous_x = previous_pipe_x.get(id(pipe), pipe.x)
            if pipe_crossed_bird(previous_x, pipe.x, pipe.width, bird.x) and id(pipe) not in passed_ids:
                passed_ids.add(id(pipe))
                pipes_passed += 1

        if not reached_first_pipe and first_pipe.x <= bird.x:
            reached_first_pipe = True

        if crashed_into_pipe:
            break

    if death_reason is None:
        death_reason = "max_steps"
        death_bird_y = bird.y
        death_bird_velocity = bird.velocity

    print(
        "[debug-one] final "
        f"steps_executed={steps_executed} "
        f"reached_first_pipe={reached_first_pipe} "
        f"pipes_passed={pipes_passed} "
        f"death_reason={death_reason} "
        f"death_bird_y={death_bird_y:.2f} "
        f"death_bird_vel={death_bird_velocity:.2f} "
        f"screen_bounds_y=[0.00,{config.world_height:.2f}]"
    )

    return {
        "steps_executed": steps_executed,
        "reached_first_pipe": reached_first_pipe,
        "pipes_passed": pipes_passed,
        "death_reason": death_reason,
        "death_bird_y": death_bird_y,
        "death_bird_velocity": death_bird_velocity,
        "screen_bounds": {"y_min": 0.0, "y_max": config.world_height},
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
        best_result = max(generation_results, key=lambda result: result["fitness"])
        best_avg_shaping_penalty = best_result.get("average_centering_penalty", 0.0)
        best_steps_component = best_result.get("steps", 0.0)
        best_pipes_reward = best_result.get("pipes_reward", 0.0)
        best_shaping_penalty = best_result.get("shaping_penalty", 0.0)
        best_first_pipe_bonus = best_result.get("reached_first_pipe_bonus", 0.0)
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
            f"max_steps={config.max_steps} "
            f"best_steps={best_steps} best_pipes_passed={best_pipes_passed} "
            f"mean_pipes_passed={mean_pipes_passed:.2f} "
            f"best_steps_component={best_steps_component:.2f} "
            f"best_pipes_reward={best_pipes_reward:.2f} "
            f"best_shaping_penalty={best_shaping_penalty:.2f} "
            f"best_reached_first_pipe_bonus={best_first_pipe_bonus:.2f} "
            f"best_avg_shaping_penalty={best_avg_shaping_penalty:.3f}"
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
                "best_steps_component": best_steps_component,
                "best_pipes_reward": best_pipes_reward,
                "best_shaping_penalty": best_shaping_penalty,
                "best_reached_first_pipe_bonus": best_first_pipe_bonus,
                "best_avg_shaping_penalty": best_avg_shaping_penalty,
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
            "best_steps_component": generation.get("best_steps_component", 0.0),
            "best_pipes_reward": generation.get("best_pipes_reward", 0.0),
            "best_shaping_penalty": generation.get("best_shaping_penalty", 0.0),
            "best_reached_first_pipe_bonus": generation.get("best_reached_first_pipe_bonus", 0.0),
            "best_avg_shaping_penalty": generation.get("best_avg_shaping_penalty", 0.0),
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
                    "best_steps_component",
                    "best_pipes_reward",
                    "best_shaping_penalty",
                    "best_reached_first_pipe_bonus",
                    "best_avg_shaping_penalty",
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=SimulationConfig.max_steps,
        help="Maximum simulation steps per episode",
    )
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
    parser.add_argument(
        "--flap-policy",
        choices=["probabilistic", "hysteresis"],
        default="probabilistic",
        help="Policy for converting network output to flap action",
    )
    parser.add_argument(
        "--flap-cooldown-frames",
        type=int,
        default=FLAP_COOLDOWN_FRAMES,
        help="Frames to wait after a flap before another flap is allowed",
    )
    parser.add_argument("--flap-on-threshold", type=float, default=FLAP_ON_THRESHOLD)
    parser.add_argument("--flap-off-threshold", type=float, default=FLAP_OFF_THRESHOLD)
    parser.add_argument("--vel-min", type=float, default=VEL_MIN)
    parser.add_argument("--vel-max", type=float, default=VEL_MAX)
    parser.add_argument("--ceiling-touch-penalty", type=float, default=CEILING_TOUCH_PENALTY)
    parser.add_argument(
        "--debug-one",
        action="store_true",
        help="Run exactly one episode with periodic debug logging",
    )
    parser.add_argument("--debug-no-flap", action="store_true", help="Force flap=False")
    parser.add_argument("--debug-always-flap", action="store_true", help="Force flap=True")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        seed=args.seed,
        max_steps=max(1, args.max_steps),
        flap_policy=args.flap_policy,
        flap_cooldown_frames=max(0, args.flap_cooldown_frames),
        flap_on_threshold=args.flap_on_threshold,
        flap_off_threshold=args.flap_off_threshold,
        velocity_min=min(args.vel_min, args.vel_max),
        velocity_max=max(args.vel_min, args.vel_max),
        ceiling_touch_penalty=max(0.0, args.ceiling_touch_penalty),
        debug_no_flap=args.debug_no_flap,
        debug_always_flap=args.debug_always_flap,
    )

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

    if args.debug_one:
        run_debug_one_episode(config)
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
