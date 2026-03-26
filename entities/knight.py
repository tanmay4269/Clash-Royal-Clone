import pygame
from pygame import Vector2

import numpy as np
import heapq
from collections import deque

from entities.troup import Troup


class Knight(Troup):
    def __init__(self, owner, row, col):
        super().__init__(owner, row, col, radius=0.5, speed=10.0, mass=1.0, attack_radius=1.0)

    def get_deploy_cost(self) -> int:
        return 3
