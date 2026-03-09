import unittest
from types import SimpleNamespace

from bird import Bird
from main import SimulationConfig, normalise_inputs


class BirdInputTests(unittest.TestCase):
    def test_normalise_inputs_are_centred_and_normalised(self) -> None:
        config = SimulationConfig(world_width=500.0, world_height=800.0, velocity_min=-12.0, velocity_max=12.0)
        bird = Bird(y=260.0, velocity=-4.0, x=100.0, world_width=500.0, world_height=800.0)
        pipe = SimpleNamespace(x=220.0, width=80.0, top=200.0, bottom=360.0)

        inputs = normalise_inputs(bird, [pipe], config)

        self.assertEqual(len(inputs), 6)
        self.assertAlmostEqual(inputs[0], (2.0 * (260.0 / 800.0)) - 1.0)
        self.assertAlmostEqual(inputs[1], -4.0 / 12.0)
        self.assertAlmostEqual(inputs[2], (220.0 - 100.0) / 500.0)
        self.assertAlmostEqual(inputs[3], (260.0 - 280.0) / 80.0)
        self.assertAlmostEqual(inputs[4], (200.0 - 260.0) / 800.0)
        self.assertAlmostEqual(inputs[5], (360.0 - 260.0) / 800.0)

    def test_normalise_inputs_clips_distance_velocity_and_offsets(self) -> None:
        config = SimulationConfig(world_width=500.0, world_height=800.0, velocity_min=-12.0, velocity_max=12.0)
        bird = Bird(y=2000.0, velocity=200.0, x=100.0, world_width=500.0, world_height=800.0)
        far_pipe = SimpleNamespace(x=5000.0, width=80.0, top=-1000.0, bottom=3000.0)

        inputs = normalise_inputs(bird, [far_pipe], config)

        self.assertTrue(-1.0 <= inputs[0] <= 1.0)
        self.assertEqual(inputs[1], 1.0)
        self.assertEqual(inputs[2], 1.0)
        self.assertTrue(-1.0 <= inputs[3] <= 1.0)
        self.assertTrue(-1.0 <= inputs[4] <= 1.0)
        self.assertTrue(-1.0 <= inputs[5] <= 1.0)

    def test_normalise_inputs_without_pipe_returns_neutral_defaults(self) -> None:
        config = SimulationConfig(world_width=500.0, world_height=800.0, velocity_min=-12.0, velocity_max=12.0)
        bird = Bird(y=100.0, velocity=20.0, x=100.0, world_height=800.0)

        inputs = normalise_inputs(bird, [], config)

        self.assertEqual(inputs, [-0.75, 1.0, 1.0, 0.0, 0.0, 0.0])

    def test_jump_respects_flap_cooldown(self) -> None:
        bird = Bird(velocity=0.0, flap_cooldown_frames=2)

        self.assertTrue(bird.jump())
        self.assertEqual(bird.velocity, bird.jump_strength)

        self.assertFalse(bird.jump())
        self.assertEqual(bird.velocity, bird.jump_strength)

        bird.update()
        self.assertFalse(bird.jump())

        bird.update()
        self.assertTrue(bird.jump())

    def test_zero_cooldown_allows_consecutive_jumps(self) -> None:
        bird = Bird(velocity=0.0, flap_cooldown_frames=0)

        self.assertTrue(bird.jump())
        self.assertTrue(bird.jump())


if __name__ == "__main__":
    unittest.main()
