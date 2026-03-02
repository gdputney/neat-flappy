import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class VisualizeGenomeScriptTests(unittest.TestCase):
    def test_visualize_genome_writes_dot_and_png(self) -> None:
        sample = {
            "genome": {
                "node_genes": [
                    {"id": 0, "type": "input", "bias": 0.0},
                    {"id": 1, "type": "input", "bias": 0.0},
                    {"id": 2, "type": "output", "bias": 0.1},
                ],
                "connection_genes": [
                    {"in_node": 0, "out_node": 2, "weight": 1.25, "enabled": True, "innovation": 1},
                    {"in_node": 1, "out_node": 2, "weight": -0.75, "enabled": False, "innovation": 2},
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            genome_path = tmp_path / "best_genome.json"
            dot_path = tmp_path / "best_genome.dot"
            png_path = tmp_path / "best_genome.png"
            genome_path.write_text(json.dumps(sample), encoding="utf-8")

            subprocess.run(
                [
                    "python",
                    "tools/visualize_genome.py",
                    str(genome_path),
                    "--dot-out",
                    str(dot_path),
                    "--png-out",
                    str(png_path),
                ],
                check=True,
            )

            self.assertTrue(dot_path.exists())
            self.assertTrue(png_path.exists())
            dot_text = dot_path.read_text(encoding="utf-8")
            self.assertIn("input_0", dot_text)
            self.assertIn("output_2", dot_text)
            self.assertIn("+1.25", dot_text)
            self.assertNotIn("-0.75", dot_text)
            self.assertIn('shape="box"', dot_text)
            self.assertIn('shape="doublecircle"', dot_text)

    def test_visualize_genome_supports_best_genome_wrapper(self) -> None:
        sample = {
            "best_genome": {
                "generation": 12,
                "genome": {
                    "node_genes": [
                        {"id": 10, "type": "input"},
                        {"id": 11, "type": "hidden"},
                        {"id": 12, "type": "output"},
                    ],
                    "connection_genes": [
                        {"in_node": 10, "out_node": 11, "weight": 0.5, "enabled": True},
                        {"in_node": 11, "out_node": 12, "weight": -0.3, "enabled": True},
                    ],
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            genome_path = tmp_path / "stats.json"
            dot_path = tmp_path / "best_genome.dot"
            png_path = tmp_path / "best_genome.png"
            genome_path.write_text(json.dumps(sample), encoding="utf-8")

            subprocess.run(
                [
                    "python",
                    "tools/visualize_genome.py",
                    str(genome_path),
                    "--dot-out",
                    str(dot_path),
                    "--png-out",
                    str(png_path),
                ],
                check=True,
            )

            dot_text = dot_path.read_text(encoding="utf-8")
            self.assertIn("n10", dot_text)
            self.assertIn("n11", dot_text)
            self.assertIn("n12", dot_text)
            self.assertIn("n10 -> n11", dot_text)
            self.assertIn("n11 -> n12", dot_text)


if __name__ == "__main__":
    unittest.main()
