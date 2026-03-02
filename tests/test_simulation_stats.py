import unittest
from unittest.mock import patch

from main import (
    FITNESS_CENTERING_PENALTY_SCALE,
    InnovationTracker,
    NETWORK_INPUT_SIZE,
    SimulationConfig,
    adjust_compatibility_threshold,
    create_initial_genome,
    decide_flap,
    evolve_population,
    pipe_crossed_bird,
    run_simulation,
    simulate_genome,
)


class SimulationStatsTests(unittest.TestCase):
    @staticmethod
    def _expected_centering_penalty(result: dict, world_height: float) -> float:
        penalty = 0.0
        height = world_height if world_height > 0 else 1.0
        for frame in result["frames"]:
            pipes = frame["pipes"]
            if not pipes:
                continue

            bird_x = frame["bird"]["x"]
            ahead_pipes = [pipe for pipe in pipes if (pipe["x"] + pipe["width"]) >= bird_x]
            next_pipe = min(ahead_pipes or pipes, key=lambda pipe: pipe["x"])
            gap_center = (next_pipe["top"] + next_pipe["bottom"]) / 2.0
            distance = abs(frame["bird"]["y"] - gap_center)
            penalty += FITNESS_CENTERING_PENALTY_SCALE * min(distance / height, 1.0)
        return penalty

    def test_generation_stats_include_pipe_and_step_metrics(self) -> None:
        config = SimulationConfig(population_size=6, generations=2, max_steps=20, seed=7)

        simulation_data = run_simulation(config)

        self.assertEqual(len(simulation_data["generations"]), 2)
        for generation in simulation_data["generations"]:
            self.assertIn("best_steps", generation)
            self.assertIn("best_pipes_passed", generation)
            self.assertIn("mean_pipes_passed", generation)
            self.assertIn("best_avg_shaping_penalty", generation)

            genomes = generation["genomes"]
            self.assertEqual(generation["best_steps"], max(genome["steps_alive"] for genome in genomes))
            self.assertEqual(
                generation["best_pipes_passed"],
                max(genome["pipes_passed"] for genome in genomes),
            )
            self.assertAlmostEqual(
                generation["mean_pipes_passed"],
                sum(genome["pipes_passed"] for genome in genomes) / len(genomes),
            )

            best_genome_result = max(genomes, key=lambda genome: genome["fitness"])
            self.assertAlmostEqual(
                generation["best_avg_shaping_penalty"],
                best_genome_result.get("average_centering_penalty", 0.0),
            )

    def test_fitness_prioritizes_pipes_passed(self) -> None:
        config = SimulationConfig(max_steps=15, seed=3)
        tracker = InnovationTracker()
        genome = create_initial_genome(input_size=NETWORK_INPUT_SIZE, output_size=1, tracker=tracker)

        result = simulate_genome(genome, config)

        expected = (
            (result["pipes_passed"] * 5000.0)
            + result["steps_alive"]
            + (0.0 if result["crashed"] else 25.0)
            - self._expected_centering_penalty(result, config.world_height)
        )
        self.assertAlmostEqual(result["fitness"], expected)


class FlapPolicyTests(unittest.TestCase):
    def test_probabilistic_flap_policy_uses_output_as_probability(self) -> None:
        with patch("main.random.random", return_value=0.2):
            self.assertTrue(decide_flap(output=0.3, policy="probabilistic"))
        with patch("main.random.random", return_value=0.4):
            self.assertFalse(decide_flap(output=0.3, policy="probabilistic"))

    def test_hysteresis_flap_policy_turns_on_off_with_thresholds(self) -> None:
        self.assertTrue(decide_flap(output=0.7, policy="hysteresis", is_flapping=False))
        self.assertFalse(decide_flap(output=0.3, policy="hysteresis", is_flapping=True))

    def test_hysteresis_flap_policy_holds_state_in_middle_band(self) -> None:
        self.assertTrue(decide_flap(output=0.5, policy="hysteresis", is_flapping=True))
        self.assertFalse(decide_flap(output=0.5, policy="hysteresis", is_flapping=False))


class PipePassingTests(unittest.TestCase):
    def test_pipe_crossing_counts_each_pipe_once(self) -> None:
        bird_x = 100.0
        pipe_width = 80.0
        pipe_a_positions = [220.0, 160.0, 40.0, -20.0]
        pipe_b_positions = [320.0, 260.0, 180.0, 60.0, 10.0]

        crossings = 0
        for previous_x, current_x in zip(pipe_a_positions, pipe_a_positions[1:]):
            crossings += int(pipe_crossed_bird(previous_x, current_x, pipe_width, bird_x))
        for previous_x, current_x in zip(pipe_b_positions, pipe_b_positions[1:]):
            crossings += int(pipe_crossed_bird(previous_x, current_x, pipe_width, bird_x))

        self.assertEqual(crossings, 2)




class ElitismTests(unittest.TestCase):
    def test_global_top_genomes_are_carried_unchanged(self) -> None:
        tracker = InnovationTracker()
        config = SimulationConfig(population_size=16)
        population = [create_initial_genome(input_size=NETWORK_INPUT_SIZE, output_size=1, tracker=tracker) for _ in range(16)]

        for idx, genome in enumerate(population):
            genome.fitness = float(idx)

        expected_elites = sorted(population, key=lambda genome: genome.fitness, reverse=True)[:2]
        expected_signatures = {
            (
                elite.fitness,
                tuple(sorted((conn["innovation"], conn["in_node"], conn["out_node"], conn["weight"], conn["enabled"]) for conn in elite.connection_genes)),
                tuple(sorted((node["id"], node["type"], node["bias"]) for node in elite.node_genes)),
            )
            for elite in expected_elites
        }

        next_population = evolve_population(population, tracker, config)

        next_signatures = {
            (
                genome.fitness,
                tuple(sorted((conn["innovation"], conn["in_node"], conn["out_node"], conn["weight"], conn["enabled"]) for conn in genome.connection_genes)),
                tuple(sorted((node["id"], node["type"], node["bias"]) for node in genome.node_genes)),
            )
            for genome in next_population
        }

        self.assertTrue(expected_signatures.issubset(next_signatures))


class CompatibilityThresholdTests(unittest.TestCase):
    def test_threshold_decreases_when_species_below_target(self) -> None:
        self.assertEqual(
            adjust_compatibility_threshold(1.2, species_count=3, target_species=5, step=0.05, min_threshold=0.3),
            1.15,
        )

    def test_threshold_increases_when_species_above_target(self) -> None:
        self.assertEqual(
            adjust_compatibility_threshold(1.2, species_count=7, target_species=5, step=0.05, min_threshold=0.3),
            1.25,
        )

    def test_threshold_respects_minimum_floor(self) -> None:
        self.assertEqual(
            adjust_compatibility_threshold(0.32, species_count=1, target_species=5, step=0.05, min_threshold=0.3),
            0.3,
        )


if __name__ == "__main__":
    unittest.main()
