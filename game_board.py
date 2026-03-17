from typing import List, Tuple
import numpy as np

import pygame
from pygame import Vector2

from player_side import PlayerSide, PlayerSide1, PlayerSide2
from entity import Entity

from entities.knight import Knight
from entities.crown_tower import CrownTower

class GameBoard:
    def __init__(self):
        self.tile_size = 16  # Sub-tile cells

        self.width = 18  # In tiles
        self.height = 32

        self.player_side_1 = PlayerSide1()  # The one closer to (0, 0)
        self.player_side_2 = PlayerSide2()

        self.objects: List[Entity] = []  # Contains all in game objects like buildings, troupes, etc.

        self.objects.extend([
            self.player_side_1.king_tower,
            self.player_side_1.king_tower,
        ])

        self.objects.extend(self.player_side_1.get_objects())
        self.objects.extend(self.player_side_2.get_objects())

        self.cell_occupancy = np.zeros(
            (self.width * self.tile_size, 
            self.height * self.tile_size), 
            dtype=bool
        )

        # Top and bottom rows
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size), dtype=bool), (0, 0))
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size), dtype=bool), (self.tile_size * 12, 0))
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size), dtype=bool), (0, (self.height-1) * self.tile_size))
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size), dtype=bool), (self.tile_size * 12, (self.height-1) * self.tile_size))

        # The divider
        self.occupy_cells(np.ones((self.tile_size * 5//2, self.tile_size * 2), dtype=bool), (0, self.tile_size * 15))
        self.occupy_cells(np.ones((self.tile_size * 5//2, self.tile_size * 2), dtype=bool), (self.tile_size * 31//2, self.tile_size * 15))
        self.occupy_cells(np.ones((self.tile_size * 9, self.tile_size * 2), dtype=bool), (self.tile_size * 9//2, self.tile_size * 15))

    
    def render(self, screen) -> None:
        # For simplicity, I'll keep each sub-tile cell as one pixel

        # Ground layer
        screen.fill("#D1CC95")

        # Occupancy map
        if True:
            rgb_occupancy_map = np.empty((self.cell_occupancy.shape[0], self.cell_occupancy.shape[1], 3), dtype=np.uint8)
            rgb_occupancy_map[:, :, 0] = 255 * (1 - self.cell_occupancy)
            rgb_occupancy_map[:, :, 1] = 255 * (1 - self.cell_occupancy)
            rgb_occupancy_map[:, :, 2] = 255 * (1 - self.cell_occupancy)

            surface = pygame.surfarray.make_surface(rgb_occupancy_map)
            surface.set_alpha(127)
            screen.blit(surface, (0, 0))


        # Faint gridlines 
        for r in range(self.height):
            for c in range(self.width):
                pygame.draw.line(screen, (128, 128, 128), (0, self.tile_size * r), (self.tile_size * self.width, self.tile_size * r), 1)
                pygame.draw.line(screen, (128, 128, 128), (self.tile_size * c, 0), (self.tile_size * c, self.tile_size * self.height), 1)


        # All objects
        for obj in self.objects:
            obj.render(screen)

        
        # Highlighted cell
        (mouse_x, mouse_y) = pygame.mouse.get_pos()
        tile_x = mouse_x // self.tile_size
        tile_y = mouse_y // self.tile_size

        surface = pygame.Surface((self.tile_size, self.tile_size))
        surface.set_alpha(127)
        surface.fill((128, 128, 128))
        screen.blit(surface, (tile_x * self.tile_size, tile_y * self.tile_size))
 

    def update(self, dt) -> None:
        for obj in self.objects:
            obj.update(dt)


    def on_click(self) -> None:
        (mouse_x, mouse_y) = pygame.mouse.get_pos()
        tile_x = mouse_x // self.tile_size
        tile_y = mouse_y // self.tile_size

        self.objects.append(
            Knight(self.player_side_2, tile_x + 1, tile_y + 1)
        )


    def occupy_cells(self, mask: np.ndarray, mask_pos: Tuple[int, int]) -> bool:
        # If mask overlaps with something, return false
        # mask_pos is expected to be x, y and mask is to be width, height shaped

        assert len(mask.shape) == 2
        assert mask.dtype == bool

        # 1. Check with self.cell_occupancy, if any intersection, return false
        tmp_mask = self.cell_occupancy[
            mask_pos[0] : mask_pos[0] + mask.shape[0], 
            mask_pos[1] : mask_pos[1] + mask.shape[1], 
        ]

        if np.any(tmp_mask & mask):
            return False

        # 2. Else just OR it to the cell occupancy
        self.cell_occupancy[
            mask_pos[0] : mask_pos[0] + mask.shape[0], 
            mask_pos[1] : mask_pos[1] + mask.shape[1], 
        ] = mask

        return True


    def deploy_at(self, deploy_me: Entity) -> bool:
        # Return false if the entity can't be deployed in its current form

        # 1. Check if player has enough elixir
        if deploy_me.owner.elixirs < deploy_me.get_deploy_cost():
            return False
        
        # 2. Check if the deploy location (already written into the object, 
        #   access via public method) is available to deploy, if not return False
        # Accept all for now

        # 3. Add to self.objects
        self.objects.append(deploy_me)

        # 4. Subtract player's elixirs and return True
        deploy_me.owner.elixirs -= deploy_me.get_deploy_cost()
        return True
