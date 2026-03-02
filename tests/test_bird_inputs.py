import unittest
from types import SimpleNamespace

from bird import Bird


class BirdInputTests(unittest.TestCase):
    def test_get_inputs_includes_gap_center_offset(self) -> None:
        bird = Bird(y=260.0, velocity=-4.0, x=100.0, world_width=500.0, world_height=800.0)
        pipe = SimpleNamespace(x=220.0, top=200.0, bottom=360.0)

        inputs = bird.get_inputs([pipe])

        self.assertEqual(len(inputs), 6)
        self.assertAlmostEqual(inputs[0], 260.0 / 800.0)
        self.assertAlmostEqual(inputs[1], -4.0 / 8.0)
        self.assertAlmostEqual(inputs[2], (220.0 - 100.0) / 500.0)
        self.assertAlmostEqual(inputs[3], (260.0 - 280.0) / 800.0)
        self.assertAlmostEqual(inputs[4], (260.0 - 200.0) / 800.0)
        self.assertAlmostEqual(inputs[5], (360.0 - 260.0) / 800.0)

    def test_get_inputs_without_pipe_returns_neutral_defaults(self) -> None:
        bird = Bird(y=100.0, velocity=20.0, x=100.0, world_height=800.0)

        inputs = bird.get_inputs([])

        self.assertEqual(inputs, [0.125, 1.0, 1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
