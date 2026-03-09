import json
import ast
import subprocess
import sys
import unittest
from pathlib import Path


class TrainingReplayExportTests(unittest.TestCase):
    def test_training_replay_config_payload_uses_expected_keys(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = ast.parse((repo_root / "main.py").read_text(encoding="utf-8"))

        write_training_replay = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "write_training_replay"
        )
        config_assignment = next(
            node
            for node in write_training_replay.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "config_payload" for target in node.targets)
        )

        config_literal = config_assignment.value
        config_keys = [
            key.value for key in config_literal.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        expected_keys = {
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
        }
        self.assertSetEqual(expected_keys, set(config_keys))
        self.assertEqual(1, config_keys.count("bird_x"), "config payload should include exactly one bird_x key")

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
        self.assertEqual(2, len(payload.get("generation_files", [])))
        self.assertEqual(2, len(payload.get("generations", [])))

        first_generation_summary = payload.get("generations", [{}])[0]
        self.assertIn("file", first_generation_summary)
        first_generation_path = repo_root / "web" / first_generation_summary["file"]
        self.assertTrue(first_generation_path.exists())

        first_generation = json.loads(first_generation_path.read_text(encoding="utf-8"))
        first_genome = (first_generation.get("genomes") or [{}])[0]
        first_frame = (first_genome.get("frames") or [{}])[0]
        frame_pipes = first_frame.get("pipes") or []
        self.assertGreater(len(frame_pipes), 0, "expected at least one pipe in first replay frame")
        self.assertIn("x", frame_pipes[0])
        self.assertIn("gap_y", frame_pipes[0])
        self.assertIn("gap_h", frame_pipes[0])

        for generation_summary in payload.get("generations", []):
            shard_path = repo_root / "web" / generation_summary.get("file", "")
            generation = json.loads(shard_path.read_text(encoding="utf-8"))
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
