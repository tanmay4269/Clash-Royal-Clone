import pygame
from pygame import Vector2

import numpy as np
import heapq
from collections import deque

from entities.troop import Troop


class Knight(Troop):
    def __init__(self, owner, row, col):
        super().__init__(owner, row, col, radius=0.5, speed=10.0, mass=1.0, attack_radius=1.0)
        
    
    def render(self, screen) -> None:
        if self.owner.side_index == 1:
            color = "red"
        else:
            color = "blue"

        super().render(screen, color)


    def get_deploy_cost(self) -> int:
        return 3
