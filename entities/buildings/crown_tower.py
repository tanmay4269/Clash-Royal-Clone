from utils import *
from entities.building import Building


class CrownTower(Building):
    def __init__(
        self, owner, row: int, col: int, 
        width: int, height: int,
        hitpoints,
        damage,
        attack_radius,
        hit_speed, first_hit_speed,
    ):
        super().__init__(
            owner, row, col,
            0,  # deploy cost
            0,  # deploy delay
            EntityType.BUILDING,
            width, height,
            hitpoints,
            damage,
            attack_radius,
            hit_speed, first_hit_speed,
            EntityType.get_all()
        )

    