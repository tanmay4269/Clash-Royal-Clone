from utils import *


class Entity:
    CELL_OCCUPANCY_LAYERS = 3  # Excluding the trivial 0th layer

    def __init__(
        self, owner, row, col,
        hitpoints,
        damage,
        attack_radius,
        hit_speed,
    ):
        """
        owner: of type PlayerSide
        row, col: in tiles
        hitpoints: health
        damage: attack damage per shot
        hit_speed: in sec
        """

        self.owner = owner

        # row col are the entity centres
        self.row = row
        self.col = col

        self.position = Vector2(col * 16, row * 16)  # TODO: remove the hardcoding

        self.is_alive = True


        # Attack Attributes
        self.hitpoints = hitpoints  # The maximum health
        self.health = hitpoints     # The real time health
        self.damage = damage
        self.attack_radius_cells = attack_radius * 16
        self.hit_speed = hit_speed

        self._attack_timer = 0


    def get_deploy_cost(self) -> int:
        # TODO: migrate to just a public variable
        raise NotImplementedError


    def render(self, screen) -> None:
        raise NotImplementedError


    def update(self, dt, arena_cell_occupancy) -> bool:
        """
        return: True means alive
        """
        raise NotImplementedError


    def get_cell_occupancy_index(self) -> int:
        """
        0 => Unoccupied
        1 => Permanent occupancy
        2 => Building
        3 => Troop
        """
        return 0


    def get_cell_occupancy(self):
        # occupancy grid and top left position
        raise NotImplementedError


    ########################
    ### Combat Mechanics ###
    ########################

    def attack_mechanics(self) -> None:
        raise NotImplementedError


    def apply_damage(self, damage) -> None:
        """
        blindly trust whoever called this!
        """
        self.health -= damage
