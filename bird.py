"""Bird model for a Flappy Bird NEAT simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol


class PipeLike(Protocol):
    """Protocol describing the pipe fields used by Bird.get_inputs."""

    x: float
    top: float
    bottom: float


@dataclass
class Bird:
    """Simple bird physics and NEAT input generation.

    Attributes:
        y: Current vertical position (pixels).
        velocity: Current vertical velocity (pixels/tick).
        x: Horizontal position (pixels), typically fixed in Flappy Bird.
        gravity: Gravity acceleration applied each update (pixels/tick^2).
        jump_strength: Velocity set when jump() is called (negative goes up).
        max_fall_speed: Clamp for downward velocity.
        world_width: Width used to normalize horizontal distances.
        world_height: Height used to normalize vertical values.
    """

    y: float = 250.0
    velocity: float = 0.0
    x: float = 100.0
    gravity: float = 0.5
    jump_strength: float = -8.0
    max_fall_speed: float = 10.0
    world_width: float = 500.0
    world_height: float = 800.0
    _last_y: float = field(default=250.0, init=False, repr=False)

    def jump(self) -> None:
        """Apply an instant upward impulse."""
        self.velocity = self.jump_strength

    def update_physics(self) -> None:
        """Advance one simulation step using simple gravity physics."""
        self._last_y = self.y
        self.velocity = min(self.velocity + self.gravity, self.max_fall_speed)
        self.y += self.velocity

    def get_inputs(self, pipes: Iterable[PipeLike]) -> list[float]:
        """Return normalized neural-network inputs for NEAT.

        The output is a list of values in roughly [-1, 1] or [0, 1], depending
        on input meaning:
          1) bird y position normalized by world height
          2) bird vertical velocity normalized by max fall speed
          3) horizontal distance to next pipe normalized by world width
          4) vertical distance from bird to next gap top normalized by height
          5) vertical distance from bird to next gap bottom normalized by height
        """
        pipe_list = list(pipes)
        ahead_pipes = [p for p in pipe_list if p.x + 1 >= self.x]
        next_pipe = min(ahead_pipes or pipe_list, key=lambda p: p.x, default=None)

        if next_pipe is None:
            # No pipes visible; provide neutral/default distances.
            return [
                self._normalize_height(self.y),
                self._normalize_velocity(self.velocity),
                1.0,
                0.0,
                0.0,
            ]

        horizontal_distance = next_pipe.x - self.x
        to_gap_top = self.y - next_pipe.top
        to_gap_bottom = next_pipe.bottom - self.y

        return [
            self._normalize_height(self.y),
            self._normalize_velocity(self.velocity),
            self._clamp(horizontal_distance / self.world_width, -1.0, 1.0),
            self._clamp(to_gap_top / self.world_height, -1.0, 1.0),
            self._clamp(to_gap_bottom / self.world_height, -1.0, 1.0),
        ]

    def _normalize_height(self, value: float) -> float:
        return self._clamp(value / self.world_height, 0.0, 1.0)

    def _normalize_velocity(self, value: float) -> float:
        max_upward = abs(self.jump_strength) if self.jump_strength else 1.0
        max_downward = self.max_fall_speed if self.max_fall_speed else 1.0
        if value >= 0:
            return self._clamp(value / max_downward, -1.0, 1.0)
        return self._clamp(value / max_upward, -1.0, 1.0)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
