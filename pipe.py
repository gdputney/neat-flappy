"""Pipe model for a Flappy Bird simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass
class Pipe:
    """Represents one Flappy Bird pipe pair with a vertical gap."""

    x: float
    width: float = 80.0
    world_height: float = 800.0
    gap_size: float = 180.0
    speed: float = 3.0
    min_margin: float = 50.0
    rng: random.Random | None = None
    top: float = field(init=False)
    bottom: float = field(init=False)

    def __post_init__(self) -> None:
        """Initialize a randomized gap position for the pipe pair."""
        generator = self.rng or random
        min_center = self.min_margin + (self.gap_size / 2)
        max_center = self.world_height - self.min_margin - (self.gap_size / 2)

        if min_center > max_center:
            raise ValueError("gap_size and min_margin do not fit within world_height")

        gap_center = generator.uniform(min_center, max_center)
        self.top = gap_center - (self.gap_size / 2)
        self.bottom = gap_center + (self.gap_size / 2)

    def update(self) -> None:
        """Move the pipe left by its configured speed."""
        self.x -= self.speed
