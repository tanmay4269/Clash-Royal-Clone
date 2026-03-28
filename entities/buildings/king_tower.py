from utils import *
from entities.buildings.crown_tower import CrownTower


class KingTower(CrownTower):
    def __init__(
        self, owner, row: int, col: int, 
        width: int, height: int,
    ):
        """
        Level 9
        """
        super().__init__(
            owner, row, col,
            width, height,
            hitpoints=4008,
            damage=90,
            attack_radius=7,
            hit_speed=1.0, first_hit_speed=0.0
        )

    
    def render(self, screen) -> None:
        if self.owner.side_index == 1:
            color = "red"
        else:
            color = LIGHT_BLUE

        super().render(screen, color)


    def update(self, dt, arena_cell_occupancy) -> bool:
        # TODO: shooting mechanics yet to be implemented
        

        if self.health < 0:
            return False

        return True


    def attack_mechanics(self) -> None:
        # TODO
        ...
