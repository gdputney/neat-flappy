import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from bird import Bird
from pipe import Pipe

from main import (
    FIRST_PIPE_REACHED_BONUS,
    SHAPING_SCALE,
    ALIVE_BONUS_AFTER_FIRST_PIPE,
    InnovationTracker,
    NETWORK_INPUT_SIZE,
    SimulationConfig,
    adjust_compatibility_threshold,
    create_initial_genome,
    decide_flap,
    evolve_population,
    parse_args,
    pipe_crossed_bird,
    run_debug_one_episode,
    derive_seed,
    evaluate_genome,
    run_simulation,
    simulate_genome,
    clamp,
    proximity_weight,
    normalize_inputs,
    compute_curriculum_params,
)


class SimulationStatsTests(unittest.TestCase):
    @staticmethod
    def _expected_shaping_reward_total(result: dict, config: SimulationConfig) -> float:
        shaping_reward_total = 0.0
        previous_abs_gap_error_norm = None

        for frame in result["frames"]:
            pipes = frame["pipes"]
            if not pipes:
                continue

            bird_x = frame["bird"]["x"]
            ahead_pipes = [pipe for pipe in pipes if (pipe["x"] + pipe["width"]) >= bird_x]
            if not ahead_pipes:
                continue

            next_pipe = min(ahead_pipes, key=lambda pipe: pipe["x"])
            gap_center = (next_pipe["top"] + next_pipe["bottom"]) / 2.0
            half_gap_height = max((next_pipe["bottom"] - next_pipe["top"]) / 2.0, 1e-6)
            abs_gap_error = abs(frame["bird"]["y"] - gap_center) / half_gap_height
            dx_to_next_pipe = next_pipe["x"] - bird_x
            proximity = proximity_weight(dx_to_next_pipe=dx_to_next_pipe, ramp_distance=config.pipe_spacing)
            clamped_abs_gap_error = clamp(abs_gap_error, 0.0, config.abs_gap_error_clamp)
            normalized_abs_gap_error = clamp(
                clamped_abs_gap_error / max(config.abs_gap_error_clamp, 1e-6),
                0.0,
                1.0,
            )

            if config.enable_centering_reward:
                shaping_reward_total += config.centering_reward_scale * (1.0 - normalized_abs_gap_error) * proximity

            if config.enable_progress_reward and previous_abs_gap_error_norm is not None:
                progress = previous_abs_gap_error_norm - normalized_abs_gap_error
                progress = clamp(progress, -config.progress_reward_clamp, config.progress_reward_clamp)
                shaping_reward_total += config.progress_reward_scale * progress * proximity

            previous_abs_gap_error_norm = normalized_abs_gap_error
        return shaping_reward_total

    def test_generation_stats_include_pipe_and_step_metrics(self) -> None:
        config = SimulationConfig(population_size=6, generations=2, max_steps=20, seed=7)

        simulation_data = run_simulation(config)

        self.assertEqual(len(simulation_data["generations"]), 2)
        for generation in simulation_data["generations"]:
            self.assertIn("best_steps", generation)
            self.assertIn("best_pipes_passed", generation)
            self.assertIn("mean_pipes_passed", generation)
            self.assertIn("best_avg_shaping_reward", generation)
            self.assertIn("best_steps_component", generation)
            self.assertIn("best_pipes_reward", generation)
            self.assertIn("best_shaping_reward", generation)
            self.assertIn("best_reached_first_pipe_bonus", generation)
            self.assertIn("best_avg_abs_gap_error", generation)
            self.assertIn("best_mean_proximity_weight", generation)
            self.assertIn("best_hidden_nodes", generation)
            self.assertIn("best_enabled_connections", generation)
            self.assertIn("population_mean_hidden_nodes", generation)
            self.assertIn("population_mean_enabled_connections", generation)

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
                generation["best_pipes_passed"],
                best_genome_result["pipes_passed"],
            )

            self.assertAlmostEqual(
                generation["best_avg_shaping_reward"],
                best_genome_result.get("average_shaping_reward", 0.0),
            )
            self.assertAlmostEqual(generation["best_steps_component"], best_genome_result.get("steps", 0.0))
            self.assertAlmostEqual(generation["best_pipes_reward"], best_genome_result.get("pipes_reward", 0.0))
            self.assertAlmostEqual(generation["best_shaping_reward"], best_genome_result.get("shaping_reward", 0.0))
            self.assertAlmostEqual(
                generation["best_reached_first_pipe_bonus"],
                best_genome_result.get("reached_first_pipe_bonus", 0.0),
            )
            self.assertAlmostEqual(
                generation["best_avg_abs_gap_error"],
                best_genome_result.get("avg_abs_gap_error", 0.0),
            )
            self.assertAlmostEqual(
                generation["best_mean_proximity_weight"],
                best_genome_result.get("mean_proximity_weight", 0.0),
            )




    def test_compute_curriculum_params_uses_highest_achieved_milestone(self) -> None:
        config = SimulationConfig(enable_curriculum=True)

        gap, speed, spacing, level = compute_curriculum_params(25, 180.0, 3.0, 220.0, config)

        self.assertEqual(level, 1)
        self.assertEqual(gap, 175.0)
        self.assertEqual(speed, 3.1)
        self.assertEqual(spacing, 220.0)

    def test_compute_curriculum_params_clamps_gap_and_spacing(self) -> None:
        config = SimulationConfig(
            enable_curriculum=True,
            curriculum_milestones=(5,),
            curriculum_gap_deltas=(300.0,),
            curriculum_speed_deltas=(0.5,),
            curriculum_spacing_deltas=(200.0,),
        )

        gap, speed, spacing, level = compute_curriculum_params(5, 180.0, 3.0, 220.0, config)

        self.assertEqual(level, 0)
        self.assertEqual(gap, 80.0)
        self.assertEqual(speed, 3.5)
        self.assertEqual(spacing, 140.0)

    def test_generation_stats_include_curriculum_fields_when_enabled(self) -> None:
        config = SimulationConfig(population_size=6, generations=2, max_steps=20, seed=7, enable_curriculum=True)

        simulation_data = run_simulation(config)

        for generation in simulation_data["generations"]:
            self.assertIn("curriculum_enabled", generation)
            self.assertIn("curriculum_level", generation)
            self.assertIn("curriculum_best_pipes_ever", generation)
            self.assertIn("curriculum_gap", generation)
            self.assertIn("curriculum_pipe_speed", generation)
            self.assertIn("curriculum_pipe_spacing", generation)
            self.assertTrue(generation["curriculum_enabled"])

    def test_seeded_run_is_deterministic_for_first_five_generations(self) -> None:
        config_a = SimulationConfig(generations=5, population_size=20, max_steps=60, seed=7, deterministic_pipes=True, flap_policy="deterministic")
        config_b = SimulationConfig(generations=5, population_size=20, max_steps=60, seed=7, deterministic_pipes=True, flap_policy="deterministic")

        run_a = run_simulation(config_a)
        run_b = run_simulation(config_b)

        fields = [
            "best_fitness",
            "mean_fitness",
            "median_fitness",
            "best_pipes_passed",
            "best_hidden_nodes",
            "best_enabled_connections",
            "best_avg_abs_gap_error",
        ]
        for generation_a, generation_b in zip(run_a["generations"], run_b["generations"]):
            for field in fields:
                self.assertEqual(generation_a[field], generation_b[field])

    def test_evaluate_genome_returns_scalar_metrics(self) -> None:
        config = SimulationConfig(max_steps=25, seed=8)
        tracker = InnovationTracker()
        genome = create_initial_genome(input_size=NETWORK_INPUT_SIZE, output_size=1, tracker=tracker)

        evaluated = evaluate_genome(genome, config, generation_index=0, genome_index=0)

        self.assertNotIn("eval_episodes", evaluated)
        self.assertNotIn("episode_fitnesses", evaluated)
        self.assertNotIn("episode_pipes_passed", evaluated)
        self.assertAlmostEqual(evaluated["pipes_reward"], 5000.0 * evaluated["pipes_passed"])

    def test_deterministic_pipe_schedule_repeats_for_same_generation(self) -> None:
        config = SimulationConfig(max_steps=40, seed=9, deterministic_pipes=True, flap_policy="deterministic")
        tracker = InnovationTracker()
        genome = create_initial_genome(input_size=NETWORK_INPUT_SIZE, output_size=1, tracker=tracker)

        pipe_seed = derive_seed(config.seed or 0, 2)
        action_seed = derive_seed(config.seed or 0, 2, 5, 17)

        first = simulate_genome(genome, config, pipe_rng_seed=pipe_seed, action_rng_seed=action_seed)
        second = simulate_genome(genome, config, pipe_rng_seed=pipe_seed, action_rng_seed=action_seed)

        first_pipes = [(frame["pipes"][0]["top"], frame["pipes"][0]["bottom"]) for frame in first["frames"] if frame["pipes"]]
        second_pipes = [(frame["pipes"][0]["top"], frame["pipes"][0]["bottom"]) for frame in second["frames"] if frame["pipes"]]

        self.assertEqual(first_pipes, second_pipes)
        self.assertEqual(first["fitness"], second["fitness"])
        self.assertEqual(first["pipes_passed"], second["pipes_passed"])

    def test_generation_log_includes_max_steps(self) -> None:
        config = SimulationConfig(population_size=3, generations=1, max_steps=17, seed=11)

        stream = io.StringIO()
        with redirect_stdout(stream):
            run_simulation(config)

        output = stream.getvalue()
        self.assertIn("max_steps=17", output)

    def test_generation_pipe_stats_use_scalar_best_pipes(self) -> None:
        config = SimulationConfig(
            population_size=8,
            generations=1,
            max_steps=120,
            seed=13,
            deterministic_pipes=True,
            flap_policy="deterministic",
        )

        simulation_data = run_simulation(config)
        generation = simulation_data["generations"][0]
        best_genome = max(generation["genomes"], key=lambda genome: genome["fitness"])
        self.assertEqual(generation["best_pipes_passed"], best_genome["pipes_passed"])
        self.assertNotIn("best_pipes_passed_mean", generation)
        self.assertNotIn("best_pipes_passed_max", generation)




    def test_default_shaping_scale_reduces_penalty_contribution(self) -> None:
        self.assertEqual(SimulationConfig().shaping_scale, SHAPING_SCALE)
        self.assertAlmostEqual(SimulationConfig().shaping_scale, 0.1)
        self.assertEqual(SimulationConfig().alive_bonus_after_first_pipe, ALIVE_BONUS_AFTER_FIRST_PIPE)
        self.assertEqual(SimulationConfig().population_size, 100)
        self.assertEqual(SimulationConfig().target_species, 8)
        self.assertAlmostEqual(SimulationConfig().compatibility_adjust_step, 0.02)

    def test_fitness_prioritizes_pipes_passed(self) -> None:
        config = SimulationConfig(max_steps=15, seed=3)
        tracker = InnovationTracker()
        genome = create_initial_genome(input_size=NETWORK_INPUT_SIZE, output_size=1, tracker=tracker)

        result = simulate_genome(genome, config)

        expected = (
            result["steps_alive"]
            + (result["pipes_passed"] * 5000.0)
            + (FIRST_PIPE_REACHED_BONUS if result["reached_first_pipe_bonus"] > 0 else 0.0)
            + result["alive_bonus"]
            + (
                self._expected_shaping_reward_total(result, config)
                * config.shaping_weight
                * config.shaping_scale
            )
            - result["ceiling_penalty"]
        )
        self.assertAlmostEqual(result["fitness"], expected)
        self.assertEqual(result["pipes_reward"], result["pipes_passed"] * 5000.0)

    def test_simulate_genome_reports_death_details(self) -> None:
        config = SimulationConfig(max_steps=10, seed=5)
        tracker = InnovationTracker()
        genome = create_initial_genome(input_size=NETWORK_INPUT_SIZE, output_size=1, tracker=tracker)

        result = simulate_genome(genome, config)

        self.assertIn("death_reason", result)
        self.assertIn(result["death_reason"], {"hit_ground", "hit_pipe", "max_steps"})
        self.assertIn("death_bird_y", result)
        self.assertIn("death_bird_velocity", result)
        self.assertIn("screen_bounds", result)
        self.assertEqual(result["screen_bounds"]["y_min"], 0.0)
        self.assertEqual(result["screen_bounds"]["y_max"], config.world_height)



class CliParsingTests(unittest.TestCase):
    def test_parse_args_uses_default_generations(self) -> None:
        with patch("sys.argv", ["main.py"]):
            args = parse_args()

        self.assertEqual(args.generations, 10)

    def test_parse_args_accepts_generations(self) -> None:
        with patch("sys.argv", ["main.py", "--generations", "25"]):
            args = parse_args()

        self.assertEqual(args.generations, 25)

    def test_parse_args_uses_updated_default_max_steps(self) -> None:
        with patch("sys.argv", ["main.py"]):
            args = parse_args()

        self.assertEqual(args.max_steps, 5000)

    def test_parse_args_accepts_curriculum_arguments(self) -> None:
        with patch(
            "sys.argv",
            [
                "main.py",
                "--enable-curriculum",
                "--curriculum-mode",
                "species",
                "--curriculum-milestones",
                "5,10",
                "--curriculum-gap-deltas",
                "1,2",
                "--curriculum-speed-deltas",
                "0.1,0.2",
                "--curriculum-spacing-deltas",
                "0,5",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.enable_curriculum)
        self.assertEqual(args.curriculum_mode, "species")
        self.assertEqual(args.curriculum_milestones, "5,10")

    def test_parse_args_accepts_max_steps(self) -> None:
        with patch("sys.argv", ["main.py", "--max-steps", "321", "--deterministic-pipes", "--flap-policy", "deterministic"]):
            args = parse_args()

        self.assertEqual(args.max_steps, 321)
        self.assertTrue(args.deterministic_pipes)
        self.assertEqual(args.flap_policy, "deterministic")



class DebugOneEpisodeTests(unittest.TestCase):
    def test_debug_one_episode_logs_requested_fields(self) -> None:
        config = SimulationConfig(max_steps=41, seed=4)

        stream = io.StringIO()
        with redirect_stdout(stream):
            result = run_debug_one_episode(config)

        output = stream.getvalue()
        self.assertIn("initial_pipe_spawn_x=", output)
        self.assertIn("initial_pipe_speed=", output)
        self.assertIn("t=0", output)
        self.assertIn("t=20", output)
        self.assertIn("bird_x=", output)
        self.assertIn("bird_y=", output)
        self.assertIn("next_pipe_x=", output)
        self.assertIn("dx_to_next_pipe=", output)
        self.assertIn("reached_first_pipe=", output)
        self.assertIn("pipes_passed=", output)
        self.assertIn("death_reason=", output)
        self.assertIn("death_bird_y=", output)
        self.assertIn("death_bird_vel=", output)
        self.assertIn("screen_bounds_y=", output)

        self.assertIn("steps_executed", result)
        self.assertIn("reached_first_pipe", result)
        self.assertIn("pipes_passed", result)
        self.assertIn("death_reason", result)
        self.assertIn("death_bird_y", result)
        self.assertIn("death_bird_velocity", result)
        self.assertIn("screen_bounds", result)



class ShapingSignalTests(unittest.TestCase):
    def test_reported_shaping_reward_matches_frame_reconstruction(self) -> None:
        config = SimulationConfig(max_steps=20, seed=19)
        tracker = InnovationTracker()
        genome = create_initial_genome(input_size=NETWORK_INPUT_SIZE, output_size=1, tracker=tracker)

        result = simulate_genome(genome, config)

        expected_total = SimulationStatsTests._expected_shaping_reward_total(result, config)
        self.assertAlmostEqual(result["shaping_reward_total"], expected_total)

    def test_proximity_weight_scales_with_distance_to_next_pipe(self) -> None:
        ramp = 220.0
        self.assertEqual(proximity_weight(dx_to_next_pipe=500.0, ramp_distance=ramp), 0.0)
        self.assertAlmostEqual(proximity_weight(dx_to_next_pipe=110.0, ramp_distance=ramp), 0.5)
        self.assertEqual(proximity_weight(dx_to_next_pipe=0.0, ramp_distance=ramp), 1.0)
        self.assertEqual(proximity_weight(dx_to_next_pipe=-5.0, ramp_distance=ramp), 1.0)




class InputNormalizationTests(unittest.TestCase):
    def test_normalize_inputs_scales_values_to_unit_ranges(self) -> None:
        config = SimulationConfig(world_width=500.0, world_height=800.0, velocity_min=-12.0, velocity_max=12.0)
        bird = Bird(y=260.0, velocity=-6.0, x=100.0, world_width=500.0, world_height=800.0)
        pipe = Pipe(x=220.0, world_height=800.0)
        pipe.top = 200.0
        pipe.bottom = 360.0

        inputs = normalize_inputs(bird, [pipe], config)

        self.assertEqual(len(inputs), 6)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in inputs[:2] + inputs[3:]))
        self.assertTrue(0.0 <= inputs[2] <= 1.0)


class FlapPolicyTests(unittest.TestCase):
    def test_probabilistic_flap_policy_uses_output_as_probability(self) -> None:
        with patch("main.random.random", return_value=0.2):
            self.assertTrue(decide_flap(output=0.3, policy="probabilistic"))
        with patch("main.random.random", return_value=0.4):
            self.assertFalse(decide_flap(output=0.3, policy="probabilistic"))

    def test_hysteresis_flap_policy_turns_on_off_with_thresholds(self) -> None:
        self.assertTrue(decide_flap(output=0.8, policy="hysteresis", is_flapping=False))
        self.assertFalse(decide_flap(output=0.2, policy="hysteresis", is_flapping=True))

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






class CeilingAndDebugBehaviorTests(unittest.TestCase):
    def test_ceiling_clamp_applies_penalty_without_death(self) -> None:
        config = SimulationConfig(max_steps=1, ceiling_touch_penalty=123.0)
        tracker = InnovationTracker()
        genome = create_initial_genome(input_size=NETWORK_INPUT_SIZE, output_size=1, tracker=tracker)
        with patch("main.decide_flap", return_value=False), patch.object(Bird, "update_physics", autospec=True) as update:
            def force_above_ceiling(self):
                self.y = -3.0
                self.velocity = -4.0

            update.side_effect = force_above_ceiling
            result = simulate_genome(genome, config)

        self.assertEqual(result["ceiling_touches"], 1)
        self.assertEqual(result["ceiling_penalty"], 123.0)
        self.assertEqual(result["frames"][0]["bird"]["y"], 0)
        self.assertEqual(result["frames"][0]["bird"]["velocity"], 0)

    def test_debug_flags_force_flap_behavior(self) -> None:
        config = SimulationConfig(max_steps=5, debug_no_flap=True, debug_always_flap=True, seed=1)
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = run_debug_one_episode(config, interval=1)

        output = stream.getvalue()
        self.assertIn("steps_executed=", output)
        self.assertIn("death_reason=", output)
        self.assertGreaterEqual(result["steps_executed"], 1)

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
            adjust_compatibility_threshold(1.2, species_count=3, target_species=8, step=0.02, min_threshold=0.3),
            1.18,
        )

    def test_threshold_increases_when_species_above_target(self) -> None:
        self.assertEqual(
            adjust_compatibility_threshold(1.2, species_count=9, target_species=8, step=0.02, min_threshold=0.3),
            1.22,
        )

    def test_threshold_respects_minimum_floor(self) -> None:
        self.assertEqual(
            adjust_compatibility_threshold(0.31, species_count=1, target_species=8, step=0.02, min_threshold=0.3),
            0.3,
        )


if __name__ == "__main__":
    unittest.main()
