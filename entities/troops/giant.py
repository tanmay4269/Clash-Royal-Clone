from utils import *
from entities.troop import Troop
from entity import EntityType


class Giant(Troop):
    def __init__(self, owner, row, col):
        super().__init__(
            owner, row, col, 
            deploy_cost=5,
            entity_type=EntityType.GROUND,
            radius=1.0, 
            speed=5.0, 
            mass=10.0, 
            hitpoints=3968,
            damage=253,
            attack_radius=1.05,
            hit_speed=1.5,
            target_types=EntityType.BUILDING,
        )
        

    def render(self, screen) -> None:
        if self.owner.side_index == 1:
            color = "red"
        else:
            color = LIGHT_BLUE

        super().render(screen, color)

        font = pygame.font.SysFont(None, 12)
        text = font.render("G", True, (0, 0, 0))  # text, antialias, color
        screen.blit(text, self.position - Vector2(3, 3))
