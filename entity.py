from pygame import Vector2
from player_side import PlayerSide

class Entity:
    def __init__(self, owner: PlayerSide, row, col):
        self.owner = owner

        self.row = row
        self.col = col


    def render(self, screen) -> None:
        raise NotImplementedError


    def update(self, dt) -> None:
        raise NotImplementedError


    def get_deploy_cost(self) -> int:
        raise NotImplementedError

