import pygame
from pygame import Vector2
import numpy as np

from entity import Entity
# from player_side import PlayerSide

# Need an intermediate class: Troup, which has the moving charecteristics needed across troups
class Knight(Entity):
    def __init__(self, owner, row, col):
        super().__init__(owner, row, col)
        self.target = None
        self.radius = 8

        self.target_reached_dist = 16  # distance between self and target below which its considered reached
        self.speed = 10.0  # TODO: Parametrise this
        self.velocity = Vector2()


    def render(self, screen) -> None:
        pygame.draw.circle(screen, "black", self.position - Vector2(self.radius, self.radius), self.radius, width=2)


    def update(self, dt) -> None:
        # If target is not None, pathfind to that and take incremental steps towards it
        # Else set target to the closest compatable victim 

        if self.target is None:
            ...  # Set a target
            return 
        
        displacement = self.target - self.position
        if displacement.length() < self.target_reached_dist:
            self.target = None
            return

        self.velocity = displacement.normalize() * self.speed
        self.position += dt * self.velocity


    def get_deploy_cost(self) -> int:
        return 3


    def set_target(self, target: Vector2):
        # Doesn't need to know who the target is, just knowing the location is fine
        self.target = target
