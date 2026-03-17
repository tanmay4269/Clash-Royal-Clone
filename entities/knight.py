import pygame
from pygame import Vector2
import numpy as np

from entity import Entity
from player_side import PlayerSide

class Knight(Entity):
    def __init__(self, owner: PlayerSide, row, col):
        super().__init__(owner, row, col)
        self.target = None
        self.radius = 8

        self.position = Vector2(row * 16, col * 16)  # TODO: remove the hardcoding
        self.velocity = Vector2()
        
        self.velocity = Vector2(np.random.rand() - 0.5, np.random.rand() - 0.5).normalize() * 10


    def render(self, screen) -> None:
        pygame.draw.circle(screen, "black", self.position - Vector2(self.radius, self.radius), self.radius, width=2)


    def update(self, dt) -> None:
        # If target is not None, pathfind to that and take incremental steps towards it
        # Else set the target to opponent's closest tower and update towards it
        self.position += dt * self.velocity
        # print(self.position)


    def get_deploy_cost(self) -> int:
        return 3


    def set_target(self, target):
        self.target = target
