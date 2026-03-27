from utils import *
from entities.troop import Troop


class Knight(Troop):
    def __init__(self, owner, row, col):
        super().__init__(
            owner, row, col, 
            radius=0.5, 
            speed=10.0, 
            mass=1.0, 
            hitpoints=1766,
            damage=202,
            attack_radius=1.0,
            hit_speed=1.2
        )
        
        # self.sprite = pygame.image.load("assets/knight.png")
        # self.sprite = pygame.transform.scale(self.sprite, 2 * Vector2(self.size, self.size))


    def get_deploy_cost(self) -> int:
        return 3
    

    def render(self, screen) -> None:
        if self.owner.side_index == 1:
            color = "red"
        else:
            color = LIGHT_BLUE

        super().render(screen, color)

        # screen.blit(self.sprite, self.position - Vector2(self.size, self.size))

        font = pygame.font.SysFont(None, 12)
        text = font.render("K", True, (0, 0, 0))  # text, antialias, color
        screen.blit(text, self.position - Vector2(3, 3))
