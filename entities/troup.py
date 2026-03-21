import pygame
from pygame import Vector2

import numpy as np
import heapq
from collections import deque

from entity import Entity

from utils import *


class Troup(Entity):
    COLLISION_COEF  = 10.0  # This times the overlap while collision is applied to this object
    FORCE_DECAY     = 0.8

    def __init__(self, owner, row, col, radius, speed, mass):
        super().__init__(owner, row, col)
        self.radius = radius
        self.size = self.radius * 16
        self.position -= Vector2(self.size, self.size)  # Position adjusting to center to the cell

        self.mass = mass

        self.speed = speed 
        self.velocity = Vector2()
        self.acceleration = Vector2()

        self.target = None
        self.waypoint_reached_dist = 1.0
        self.waypoints = deque()  # left to right is the traversal pattern. self.find_path populates this and update pops from this

        self.cell_occupancy = np.zeros((int(self.size * 2), int(self.size * 2)))
        for r in range(int(self.size * 2)):
            for c in range(int(self.size * 2)):
                if (r - self.size) ** 2 + (c - self.size) ** 2 < self.size ** 3:
                    self.cell_occupancy[r, c] = True


    def render(self, screen) -> None:
        # pygame.draw.circle(screen, "black", self.position - Vector2(self.size, self.size), self.size, width=2)
        pygame.draw.circle(screen, "black", self.position, self.size, width=2)

        for i in range(len(self.waypoints)-1):
            pygame.draw.line(screen, "green", self.waypoints[i], self.waypoints[i+1], width=5)


    def update(self, dt) -> None:
        # If target is not None, pathfind to that and take incremental steps towards it
        # Else set target to the closest compatable victim 

        if self.target is None:
            ...  # TODO: Set a target
            return 

        if len(self.waypoints) == 0:
            self.target = None
            return
        
        displacement = self.waypoints[0] - self.position

        if displacement.length() < self.waypoint_reached_dist:
            self.waypoints.popleft()

        self.velocity = displacement.normalize() * self.speed


        # TODO: I forgot the optimal ordering here
        self.velocity += dt * self.acceleration
        self.position += dt * self.velocity

        self.acceleration *= FORCE_DECAY  # ! Assuming all forces are impulsive in the game


    def get_deploy_cost(self) -> int:
        raise NotImplementedError


    def set_target(self, target: Vector2):
        # Doesn't need to know who the target is, just knowing the location is fine
        self.target = target


    def get_cell_occupancy_index(self):
        return 3


    def get_cell_occupancy(self):
        return np.ones([1, 1]) * self.get_cell_occupancy_index(), self.position - Vector2(self.size, self.size)
        # return self.cell_occupancy * self.get_cell_occupancy_index(), self.position - Vector2(self.size, self.size)


    def apply_force(self, force: Vector2) -> None:
        self.acceleration += force / self.mass

        # Hacky -- Doesn't look great
        # self.position += force.normalize() * 0.5


    def find_path(self, occupancy_grid: np.ndarray) -> bool:
        """
        Given curent position and target position, this computes the right sequence
        of waypoints that this entity needs to navigate through.

        Returns False if path couldn't be found for some reason
        """

        if self.target is None:
            return False

        SCALE = 16  # Reduction by this much on each axis

        # 1 => occupied
        tiled_occupancy_grid = (
            np.where(occupancy_grid == 1, 1, 0)
            .reshape(occupancy_grid.shape[0] // SCALE, SCALE, occupancy_grid.shape[1] // SCALE, SCALE)
            .transpose(0, 2, 1, 3)
            .max(axis=(2, 3))
        )

        start = (int(self.position.x / SCALE), int(self.position.y / SCALE))
        target = (int(self.target.x / SCALE), int(self.target.y / SCALE))

        path = self.a_star(tiled_occupancy_grid, start, target)

        if path is None:
            return False

        self.waypoints = deque()
        for wp in path[1:-1]:
            self.waypoints.append(Vector2(wp) * SCALE + Vector2(SCALE/2, SCALE/2))
        self.waypoints.append(Vector2(path[-1]) * SCALE)  # Offset on the last waypoint looks awkward


    def a_star(self, grid, start, goal):
        """
        8-way connected on the grid
        Uses Octile Distance huristic
        """

        rows, cols = grid.shape

        def heuristic(a, b):
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)  # Octile

        neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                    (-1,-1),(-1,1),(1,-1),(1,1)]

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return [start] + path[::-1]

            for dr, dc in neighbors:
                neighbor = (current[0]+dr, current[1]+dc)
                r, c = neighbor
                if not (0 <= r < rows and 0 <= c < cols):
                    continue
                if grid[r, c] != 0:
                    continue

                # Diagonal moves cost sqrt(2), cardinal cost 1
                move_cost = np.sqrt(2) if dr != 0 and dc != 0 else 1
                new_g = g_score[current] + move_cost

                if new_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = new_g
                    f = new_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return None  # No path found

