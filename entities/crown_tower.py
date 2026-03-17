import pygame

from entity import Entity
# from player_side import PlayerSide

class CrownTower(Entity):
    def __init__(self, owner, row: int, col: int, width: int, height: int):
        super().__init__(owner, row, col)

        self.width = width
        self.height = height


    def render(self, screen) -> None:
        pygame.draw.rect(
            screen, "gray", 
            (
                self.col * 16 - self.width/2 * 16, 
                self.row * 16 - self.height/2 * 16, 
                self.width * 16, 
                self.height * 16, 
                # self.col * 16 + self.width/2 * 16, self.row * 16 + self.height/2 * 16, 
                # self.col * 16 + self.width/2 * 16, self.row * 16 + self.height/2 * 16, 
            )
        )


    def update(self, dt) -> None:
        pass


    def get_deploy_cost(self) -> int:
        return 0
