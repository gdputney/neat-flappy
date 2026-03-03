import unittest
from types import SimpleNamespace

from main import adjust_next_pipe_index, count_passed_pipes


class PipeScoringTests(unittest.TestCase):
    def test_pipe_counting_continues_past_active_pipe_count(self) -> None:
        bird_x = 100.0
        pipe_width = 30.0
        spacing = 120.0
        speed = 15.0
        active_pipe_count = 3
        target_passes = 10

        pipes = [
            SimpleNamespace(x=bird_x + 120.0 + (index * spacing), width=pipe_width)
            for index in range(active_pipe_count)
        ]

        pipes_passed = 0
        next_pipe_index = 0

        while pipes_passed < target_passes:
            for pipe in pipes:
                pipe.x -= speed

            pipes_passed, next_pipe_index = count_passed_pipes(
                pipes=pipes,
                bird_x=bird_x,
                next_pipe_index=next_pipe_index,
                pipes_passed=pipes_passed,
            )

            while pipes and (pipes[0].x + pipes[0].width) <= -5.0:
                recycled = pipes.pop(0)
                recycled.x = pipes[-1].x + spacing if pipes else bird_x + spacing
                pipes.append(recycled)
                next_pipe_index = adjust_next_pipe_index(next_pipe_index, removed_from_front=1)

        self.assertEqual(pipes_passed, target_passes)


if __name__ == "__main__":
    unittest.main()
