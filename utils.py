from typing import Tuple, List, Set
from rich import print

import numpy as np

import pygame
from pygame import Vector2

from entity import EntityType, EntityRegistry


LIGHT_BLUE = "#8BCEF7"


def smooth_arc(surface, color, rect, start_angle, stop_angle, width=1, segments=100):
    cx, cy = rect.center
    rx = rect.width / 2
    ry = rect.height / 2

    prev_point = None

    for i in range(segments + 1):
        t = i / segments
        angle = start_angle + (stop_angle - start_angle) * t

        x = cx + rx * np.cos(angle)
        y = cy + ry * np.sin(angle)

        point = (x, y)

        if prev_point:
            pygame.draw.aaline(surface, color, prev_point, point)

            # thickness (draw multiple parallel lines)
            for w in range(1, width):
                pygame.draw.aaline(surface, color,
                                   (prev_point[0], prev_point[1] + w),
                                   (point[0], point[1] + w))

        prev_point = point


class AverageMeter:
    def __init__(self, max_samples = None):
        self.max_samples = max_samples
        self.samples: List[float] = []

    def add_sample(self, value: float):
        self.samples.append(value)
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples.pop(0)

    def average(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    def window_average(self, window_size: int) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples[-window_size:]) / min(window_size, len(self.samples))


""" logging

import logging
import sys

logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

logger.addHandler(handler)

# Example logs
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
"""