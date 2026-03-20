import pygame
from pygame import Vector2

import numpy as np
import heapq
from collections import deque

from entities.troup import Troup


class Knight(Troup):
    def __init__(self, owner, row, col):
        super().__init__(owner, row, col, radius=8, speed=10.0)

    def get_deploy_cost(self) -> int:
        return 3
