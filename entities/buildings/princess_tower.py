from utils import *
from entities.buildings.crown_tower import CrownTower
from entities.projectiles.arrow import Arrow


class PrincessTower(CrownTower):
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
            hitpoints=2534,
            damage=90,
            attack_radius=7.5,
            hit_speed=0.8, first_hit_speed=0.8
        )

        self.projectiles = set()

    
    def render(self, screen) -> None:
        if self.owner.side_index == 1:
            color = "red"
        else:
            color = LIGHT_BLUE

        super().render(screen, color)

        for arrow in self.projectiles:
            arrow.render(screen)


    def update(self, dt, arena_cell_occupancy) -> bool:
        if self.health < 0:
            return False

        
        delete_projectiles = set()
        for proj in self.projectiles:
            if not proj.update(dt):
                delete_projectiles.add(proj)

        while len(delete_projectiles):
            proj = delete_projectiles.pop()
            self.projectiles.remove(proj)
            del proj
            

        ### Combat Mechanics ###
        # TODO target locking

        self._attack_timer += dt
        if self._attack_timer < self.hit_speed:
            return True

        self._attack_timer = 0  # Reset

        for obj in self.owner.opponent.objects:
            delta = obj.position - self.position
            if delta.length() > self.attack_radius_cells:
                continue
                
            self.attack_mechanics(delta.normalize(), obj.entity_type == EntityType.AIR)
            break  # Only one arrow at a time


        return True


    def attack_mechanics(self, direction, is_air_target) -> None:
        target_types = EntityType.AIR if is_air_target else set({EntityType.GROUND, EntityType.BUILDING})
        arrow = Arrow(self.owner, self.position.copy(), direction, self.damage, self.attack_radius_cells / 16, target_types)
        self.projectiles.add(arrow)
