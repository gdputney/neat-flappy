import json
import subprocess
import sys
import unittest
from pathlib import Path


class TrainingReplayExportTests(unittest.TestCase):
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
            "--eval-episodes",
            "1",
            "--record-training-replay",
            "--replay-top-k",
            "4",
            "--replay-episode",
            "0",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True, text=True)

        after = {path.name for path in runs_dir.glob("run_*")} if runs_dir.exists() else set()
        created = sorted(after - before)
        self.assertTrue(created, "expected a new run directory")
        run_dir = runs_dir / created[-1]

        replay_path = run_dir / "web" / "training_replay.json"
        self.assertTrue(replay_path.exists())

        payload = json.loads(replay_path.read_text(encoding="utf-8"))
        self.assertEqual(2, len(payload.get("generations", [])))

        for generation in payload.get("generations", []):
            genomes = generation.get("genomes", [])
            self.assertLessEqual(len(genomes), 4)
            for genome in genomes:
                frames = genome.get("frames", [])
                self.assertLessEqual(len(frames), 50)
                if frames:
                    self.assertEqual(frames[-1].get("pipes_passed"), genome.get("pipes_passed"))


if __name__ == "__main__":
    unittest.main()
