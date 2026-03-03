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
                eval_episode=0,
            )

            payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertIn("metadata", payload)
        self.assertEqual(len(payload["generations"]), config.generations)

        for generation in payload["generations"]:
            self.assertIn("pipe_seed", generation)
            self.assertLessEqual(len(generation["genomes"]), 4)
            for genome in generation["genomes"]:
                self.assertIn("genome_json", genome)
                self.assertIn("node_genes", genome["genome_json"])
                self.assertIn("connection_genes", genome["genome_json"])


if __name__ == "__main__":
    unittest.main()
