import pygame
from pygame import Vector2

import numpy as np

from entity import Entity
# from player_side import PlayerSide

class CrownTower(Entity):
    def __init__(self, owner, row: int, col: int, width: int, height: int):
        super().__init__(owner, row, col)

        self.width = width
        self.height = height

        self.position.x = self.col * 16 
        self.position.y = self.row * 16 


    def render(self, screen) -> None:
        return 
        pygame.draw.rect(
            screen, "gray", 
            (
                self.position.x - self.width/2 * 16,
                self.position.y - self.height/2 * 16,
                self.width * 16, 
                self.height * 16, 
            )
        )


    def update(self, dt) -> None:
        pass


    def get_deploy_cost(self) -> int:
        return 0


    def get_cell_occupancy(self):
        mask = np.ones((int(self.height * 16), int(self.width * 16)), dtype=bool)
        return mask, self.position - Vector2(self.width/2 * 16, self.height/2 * 16)