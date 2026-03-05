import json
import tempfile
import unittest
from pathlib import Path

from main import SimulationConfig, run_simulation, write_web_evolution


class WebEvolutionExportTests(unittest.TestCase):
    def test_export_schema_contains_generation_pipe_seed_and_top_k_genomes(self) -> None:
        config = SimulationConfig(population_size=8, generations=3, max_steps=20, seed=21)
        simulation_data = run_simulation(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "evolution.json"
            write_web_evolution(
                simulation_data=simulation_data,
                config=config,
                output_path=out_path,
                top_k=4,
            )

            payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertIn("metadata", payload)
        self.assertEqual(len(payload["generations"]), config.generations)

        for generation in payload["generations"]:
            self.assertIn("pipe_seed", generation)
            self.assertLessEqual(len(generation["genomes"]), 4)
            for genome in generation["genomes"]:
                self.assertIn("pipes_passed", genome)
                self.assertIn("genome_json", genome)
                self.assertIn("node_genes", genome["genome_json"])
                self.assertIn("connection_genes", genome["genome_json"])

    def test_export_includes_curriculum_schema_when_enabled(self) -> None:
        config = SimulationConfig(
            population_size=8,
            generations=2,
            max_steps=20,
            seed=33,
            enable_curriculum=True,
        )
        simulation_data = run_simulation(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "evolution.json"
            write_web_evolution(
                simulation_data=simulation_data,
                config=config,
                output_path=out_path,
                top_k=3,
            )

            payload = json.loads(out_path.read_text(encoding="utf-8"))

        export_config = payload["metadata"]["config"]
        self.assertIn("base_pipe_gap", export_config)
        self.assertIn("base_pipe_speed", export_config)
        self.assertIn("base_pipe_spacing", export_config)

        for generation in payload["generations"]:
            self.assertIn("curriculum_level", generation)
            self.assertIn("curriculum_best_pipes_ever", generation)
            self.assertIn("curriculum_gap", generation)
            self.assertIn("curriculum_pipe_speed", generation)
            self.assertIn("curriculum_pipe_spacing", generation)


if __name__ == "__main__":
    unittest.main()
