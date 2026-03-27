import numpy as np
from pygame import Vector2


class Entity:
    CELL_OCCUPANCY_LAYERS = 3  # Excluding the trivial 0th layer

    def __init__(self, owner, row, col):
        self.owner = owner

        # row col are the entity centres
        self.row = row
        self.col = col

        self.position = Vector2(col * 16, row * 16)  # TODO: remove the hardcoding

        self.is_alive = True


    def render(self, screen) -> None:
        raise NotImplementedError


    def update(self, dt, arena_cell_occupancy) -> None:
        raise NotImplementedError


    def get_deploy_cost(self) -> int:
        raise NotImplementedError


    def get_cell_occupancy_index(self) -> int:
        """
        0 => Unoccupied
        1 => Permanent occupancy
        2 => Building
        3 => Troop
        """
        return 0


    def get_cell_occupancy(self):
        # occupancy grid and top left position
        raise NotImplementedError
