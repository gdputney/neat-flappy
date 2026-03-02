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
    flap_cooldown_frames: int = 8
    world_width: float = 500.0
    world_height: float = 800.0
    _last_y: float = field(default=250.0, init=False, repr=False)
    _flap_cooldown: int = field(default=0, init=False, repr=False)

    def jump(self) -> bool:
        """Apply an instant upward impulse when cooldown allows it."""
        if self._flap_cooldown > 0:
            return False
        self.velocity = self.jump_strength
        self._flap_cooldown = max(0, self.flap_cooldown_frames)
        return True

    def update(self) -> None:
        """Advance one simulation step and tick down flap cooldown."""
        if self._flap_cooldown > 0:
            self._flap_cooldown -= 1

        self._last_y = self.y
        self.velocity = min(self.velocity + self.gravity, self.max_fall_speed)
        self.y += self.velocity

    def update_physics(self) -> None:
        """Backward-compatible alias for update()."""
        self.update()

    def get_inputs(self, pipes: Iterable[PipeLike]) -> list[float]:
        """Return clipped neural-network inputs for NEAT.

        Input layout:
          1) bird y position, centered to [-1, 1]
          2) bird vertical velocity clipped to [-1, 1]
          3) dx to next pipe in [0, 1]
          4) dy to next gap center clipped to [-1, 1]
          5) dy to next gap top clipped to [-1, 1]
          6) dy to next gap bottom clipped to [-1, 1]
        """
        pipe_list = list(pipes)
        ahead_pipes = [p for p in pipe_list if p.x + 1 >= self.x]
        next_pipe = min(ahead_pipes or pipe_list, key=lambda p: p.x, default=None)

        if next_pipe is None:
            return [
                self._normalize_height_centered(self.y),
                self._normalize_velocity(self.velocity),
                1.0,
                0.0,
                0.0,
                0.0,
            ]

        horizontal_distance = next_pipe.x - self.x
        gap_center = (next_pipe.top + next_pipe.bottom) / 2.0

        return [
            self._normalize_height_centered(self.y),
            self._normalize_velocity(self.velocity),
            self._normalize_forward_distance(horizontal_distance),
            self._normalize_vertical_delta(gap_center - self.y),
            self._normalize_vertical_delta(next_pipe.top - self.y),
            self._normalize_vertical_delta(next_pipe.bottom - self.y),
        ]

    def _normalize_height_centered(self, value: float) -> float:
        if self.world_height <= 0:
            return 0.0
        return self._clamp((2.0 * (value / self.world_height)) - 1.0, -1.0, 1.0)

    def _normalize_velocity(self, value: float) -> float:
        max_upward = abs(self.jump_strength) if self.jump_strength else 1.0
        max_downward = self.max_fall_speed if self.max_fall_speed else 1.0
        if value >= 0:
            return self._clamp(value / max_downward, -1.0, 1.0)
        return self._clamp(value / max_upward, -1.0, 1.0)

    def _normalize_forward_distance(self, value: float) -> float:
        width = self.world_width if self.world_width > 0 else 1.0
        return self._clamp(value / width, 0.0, 1.0)

    def _normalize_vertical_delta(self, value: float) -> float:
        height = self.world_height if self.world_height > 0 else 1.0
        return self._clamp(value / height, -1.0, 1.0)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
