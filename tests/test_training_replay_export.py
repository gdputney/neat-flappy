import json
import ast
import subprocess
import sys
import unittest
from pathlib import Path


class TrainingReplayExportTests(unittest.TestCase):
    def test_training_replay_config_literal_has_unique_keys(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = ast.parse((repo_root / "main.py").read_text(encoding="utf-8"))

        write_training_replay = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "write_training_replay"
        )
        payload_assignment = next(
            node
            for node in write_training_replay.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "payload" for target in node.targets)
        )
        config_literal = next(
            value
            for key, value in zip(payload_assignment.value.keys, payload_assignment.value.values)
            if isinstance(key, ast.Constant) and key.value == "config"
        )

        config_keys = [
            key.value for key in config_literal.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        self.assertEqual(len(config_keys), len(set(config_keys)), "duplicate config keys in replay payload literal")

    def test_record_training_replay_exports_expected_schema(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        runs_dir = repo_root / "runs"
        before = {path.name for path in runs_dir.glob("run_*")} if runs_dir.exists() else set()

        cmd = [
            sys.executable,
            "main.py",
            "--seed",
            "7",
            "--population-size",
            "6",
            "--generations",
            "2",
            "--max-steps",
            "50",
            "--record-training-replay",
            "--replay-top-k",
            "4",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True, text=True)

        after = {path.name for path in runs_dir.glob("run_*")} if runs_dir.exists() else set()
        created = sorted(after - before)
        self.assertTrue(created, "expected a new run directory")
        run_dir = runs_dir / created[-1]

        replay_path = repo_root / "web" / "training_replay.json"
        self.assertTrue(replay_path.exists())

        payload = json.loads(replay_path.read_text(encoding="utf-8"))
        self.assertSetEqual(
            {
                "seed",
                "max_steps",
                "replay_top_k",
                "flap_policy",
                "world_width",
                "world_height",
                "pipe_gap",
                "pipe_speed",
                "pipe_spacing",
                "bird_x",
            },
            set(payload.get("config", {}).keys()),
        )
        self.assertEqual(2, len(payload.get("generations", [])))

        first_generation = payload.get("generations", [{}])[0]
        first_genome = (first_generation.get("genomes") or [{}])[0]
        first_frame = (first_genome.get("frames") or [{}])[0]
        frame_pipes = first_frame.get("pipes") or []
        self.assertGreater(len(frame_pipes), 0, "expected at least one pipe in first replay frame")
        self.assertIn("x", frame_pipes[0])
        self.assertIn("gap_y", frame_pipes[0])
        self.assertIn("gap_h", frame_pipes[0])

        for generation in payload.get("generations", []):
            self.assertNotIn("episode_index", generation)
            genomes = generation.get("genomes", [])
            self.assertLessEqual(len(genomes), 4)
            for genome in genomes:
                frames = genome.get("frames", [])
                self.assertLessEqual(len(frames), 50)
                self.assertNotIn("episode_index", genome.get("meta", {}))
                if frames:
                    self.assertEqual(frames[-1].get("pipes_passed"), genome.get("pipes_passed"))


if __name__ == "__main__":
    unittest.main()
