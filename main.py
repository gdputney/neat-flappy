"""Run a simple NEAT-style Flappy Bird simulation and export frames."""

from __future__ import annotations

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
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


def run_simulation(config: SimulationConfig) -> dict[str, Any]:
    population = create_population(config.population_size)
    output: dict[str, Any] = {
        "config": config.__dict__,
        "generations": [],
        "position_history": {},
    }

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

        best_fitness = max(result["fitness"] for result in generation_results)
        avg_fitness = sum(result["fitness"] for result in generation_results) / len(generation_results)

        output["generations"].append(
            {
                "generation": generation_index,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "genomes": generation_results,
            }
        )
        population = evolve_population(population)

    return output


def main() -> None:
    config = SimulationConfig()
    simulation_data = run_simulation(config)
    output_path = Path(__file__).resolve().parent / "simulation.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(simulation_data, file, indent=2)


if __name__ == "__main__":
    main()
