import pygame
from pygame import Vector2

import numpy as np

from entities.building import Building

class CrownTower(Building):
    def __init__(self, owner, row: int, col: int, width: int, height: int):
        super().__init__(owner, row, col, width, height)

    def render(self, screen) -> None:
        # super().render(screen, "gray")
        return 
