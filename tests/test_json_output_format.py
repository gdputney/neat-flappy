import json
import tempfile
import unittest
from pathlib import Path

from main import SimulationConfig, write_json_atomic


class JsonOutputFormatTests(unittest.TestCase):
    def test_write_json_atomic_compact_output(self) -> None:
        payload = {"a": 1, "nested": {"b": 2}}
        config = SimulationConfig(json_compact=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compact.json"
            write_json_atomic(output_path, payload, config)
            text = output_path.read_text(encoding="utf-8")

        self.assertEqual(json.loads(text), payload)
        self.assertNotIn("\n", text)
        self.assertIn('"a":1', text)

    def test_write_json_atomic_pretty_output(self) -> None:
        payload = {"a": 1, "nested": {"b": 2}}
        config = SimulationConfig(json_compact=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pretty.json"
            write_json_atomic(output_path, payload, config)
            text = output_path.read_text(encoding="utf-8")

        self.assertEqual(json.loads(text), payload)
        self.assertIn("\n", text)
        self.assertIn('  "a": 1', text)


if __name__ == "__main__":
    unittest.main()
