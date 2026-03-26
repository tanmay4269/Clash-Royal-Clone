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


    def update(self, dt, arena_cell_occupancy) -> None:
        # TODO: shooting mechanics yet to be implemented
        return


    def get_deploy_cost(self) -> int:
        return 0
