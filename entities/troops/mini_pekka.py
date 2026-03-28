from utils import *
from entities.troop import Troop
from entity import EntityType


class MiniPEKKA(Troop):
    def __init__(self, owner, row, col):
        super().__init__(
            owner, row, col, 
            deploy_cost=4, deploy_delay=1.0,
            entity_type=EntityType.GROUND,
            radius=0.3, 
            speed=Troop.Speed.FAST, 
            mass=0.8, 
            hitpoints=1390,
            damage=755,
            attack_radius=Troop.AttackRadius.MELEE_SHORT,
            hit_speed=1.6, first_hit_speed=0.5,
            target_types=EntityType.get_all(),
        )
        

    def render(self, screen) -> None:
        if self.owner.side_index == 1:
            color = "red"
        else:
            color = LIGHT_BLUE

        super().render(screen, color)

        font = pygame.font.SysFont(None, 12)
        text = font.render("M", True, (0, 0, 0))  # text, antialias, color
        screen.blit(text, self.position - Vector2(3, 14))
