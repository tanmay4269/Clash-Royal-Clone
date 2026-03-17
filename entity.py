import numpy as np
from pygame import Vector2
# from player_side import PlayerSide

class Entity:
    def __init__(self, owner, row, col):
        self.owner = owner

        # row col are the entity centres
        self.row = row
        self.col = col

        self.is_alive = True


    def render(self, screen) -> None:
        raise NotImplementedError


    def update(self, dt) -> None:
        raise NotImplementedError


    def get_deploy_cost(self) -> int:
        raise NotImplementedError


    def get_cell_occupancy(self) -> np.ndarray:
        raise NotImplementedError
