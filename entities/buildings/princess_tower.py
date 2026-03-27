from utils import *
from entities.buildings.crown_tower import CrownTower


class PrincessTower(CrownTower):
    def __init__(
        self, owner, row: int, col: int, 
        width: int, height: int,
    ):
        super().__init__(
            owner, row, col,
            width, height,
            hitpoints=2534,
            damage=90,
            attack_radius=7.5,
            hit_speed=0.8,
        )

    
    def render(self, screen) -> None:
        if self.owner.side_index == 1:
            color = "red"
        else:
            color = "blue"

        super().render(screen, color)


    def update(self, dt, arena_cell_occupancy) -> bool:
        # TODO: shooting mechanics yet to be implemented
        if self.health < 0:
            return False

        return True


    def attack_mechanics(self) -> None:
        # TODO
        ...
