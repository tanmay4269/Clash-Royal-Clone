from utils import *


class EntityType:
    GROUND      = 1
    AIR         = 2
    BUILDING    = 3
    PROJECTILE  = 4

    def get_all():
        return set({
            EntityType.GROUND,
            EntityType.AIR,
            EntityType.BUILDING,
        })


class Entity:
    CELL_OCCUPANCY_LAYERS = 3  # Excluding the trivial 0th layer

    def __init__(
        self, owner, row, col,
        deploy_cost, 
        entity_type: EntityType,
        hitpoints,
        damage,
        attack_radius,
        hit_speed,
        target_types: Set[EntityType],
    ):
        """
        owner: of type PlayerSide
        row, col: in tiles
        deploy_cost: in elexirs
        entity_type: of type EntityType
        hitpoints: health
        damage: attack damage per shot
        hit_speed: in sec
        target_types: set of EntityType, only these types
            of enemies will be targeted
        """

        # Abstract Attributes
        self.owner = owner
        self.is_alive = True

        # Physical Attributes

        # (row, col) is the entity's centre
        self.row = row
        self.col = col

        self.position = Vector2(col * 16, row * 16)  # TODO: remove the hardcoding

        self.entity_type = entity_type
        self.deploy_cost = deploy_cost

        # Attack Attributes
        self.hitpoints = hitpoints  # The maximum health
        self.health = hitpoints     # The real time health
        self.damage = damage
        self.attack_radius_cells = attack_radius * 16
        self.hit_speed = hit_speed
        self.target_types = target_types  # Set of entity types
        if not isinstance(self.target_types, set):
            self.target_types = set({self.target_types})

        self._attack_timer = 0


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
