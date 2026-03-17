import pygame
from typing import List
from player_side import PlayerSide
from entity import Entity

from entities.knight import Knight

class GameBoard:
    def __init__(self):
        self.tile_size = 16  # Sub-tile cells

        self.width = 18  # In tiles
        self.height = 32

        self.player_side_1 = PlayerSide()  # The one closer to (0, 0)
        self.player_side_2 = PlayerSide()

        self.objects: List[Entity] = []  # Contains all in game objects like buildings, troupes, etc.

    
    def render(self, screen) -> None:
        # For simplicity, I'll keep each sub-tile cell as one pixel

        # Ground layer
        screen.fill("#D1CC95")

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

        temp_surface = pygame.Surface((self.tile_size, self.tile_size))
        temp_surface.set_alpha(127)
        temp_surface.fill((128, 128, 128))
        screen.blit(temp_surface, (tile_x * self.tile_size, tile_y * self.tile_size))
 

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


    def deploy_at(self, deploy_me: Entity) -> bool:
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
