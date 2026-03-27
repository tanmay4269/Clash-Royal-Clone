from utils import *
from entities.building import Building


class CrownTower(Building):
    def __init__(
        self, owner, row: int, col: int, 
        width: int, height: int,
        hitpoints,
        damage,
        attack_radius,
        hit_speed,
    ):
        super().__init__(
            owner, row, col,
            width, height,
            hitpoints,
            damage,
            attack_radius,
            hit_speed,
        )

    
    def get_deploy_cost(self) -> int:
        return 0
    
    