"""Bird model for a Flappy Bird NEAT simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


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
        world_width: Width used to normalise horizontal distances.
        world_height: Height used to normalise vertical values.
    """

    y: float = 250.0
    velocity: float = 0.0
    x: float = 100.0
    gravity: float = 0.5
    jump_strength: float = -8.0
    velocity_min: float = -12.0
    velocity_max: float = 12.0
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
        self.velocity = self._clamp(self.velocity + self.gravity, self.velocity_min, self.velocity_max)
        self.y += self.velocity

    def update_physics(self) -> None:
        """Backward-compatible alias for update()."""
        self.update()

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
