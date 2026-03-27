from utils import *
from entity import Entity


class Building(Entity):
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
            hitpoints,
            damage,
            attack_radius,
            hit_speed,
        )

        self.width = width
        self.height = height

        self.size = Vector2(self.height * 16, self.width * 16)

        self.position.x = self.col * 16 
        self.position.y = self.row * 16 


    def render(self, screen, color) -> None:
        pygame.draw.rect(
            screen, color, 
            (
                self.position.x - self.size.x/2,
                self.position.y - self.size.y/2,
                self.size.x, 
                self.size.y, 
            )
        )


    def get_cell_occupancy_index(self) -> int:
        return 2


    def get_cell_occupancy(self):
        mask = np.ones((int(self.size.x), int(self.size.y))) * self.get_cell_occupancy_index()
        return mask, self.position - Vector2(self.size.x/2, self.size.y/2)