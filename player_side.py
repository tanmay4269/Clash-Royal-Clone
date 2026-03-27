from entities.crown_tower import CrownTower

class PlayerSide:
    def __init__(self):
        self.side_index = None
        self.opponent = None

        self.king_tower = None
        self.princess_tower_1 = None  # The one closer to (0, 0) 
        self.princess_tower_2 = None

        # TODO: GameBoard should manage this instead
        self.elixirs = 5  # init
        self.max_elixirs = 10
        self.elixirs_per_sec = 1

        self.objects = None  # Add towers once initialised

    
    def get_objects(self):
        return self.objects


    def add_object(self, obj):
        self.objects.append(obj)


    def set_opponent(self, opponent):
        assert opponent is not self
        self.opponent = opponent

# The one closer to (0, 0)
class PlayerSide1(PlayerSide):
    def __init__(self):
        super().__init__()
        self.side_index = 1

        self.king_tower = CrownTower(self, 3, 18/2, 2.5, 2.5)
        self.princess_tower_1 = CrownTower(self, 6.5, 3.5, 2, 2)
        self.princess_tower_2 = CrownTower(self, 6.5, 18 - 3.5, 2, 2)

        self.objects = [
            self.king_tower, self.princess_tower_1, self.princess_tower_2
        ]


class PlayerSide2(PlayerSide):
    def __init__(self):
        super().__init__()
        self.side_index = 2

        self.king_tower = CrownTower(self, 32 - 3, 18/2, 2.5, 2.5)
        self.princess_tower_1 = CrownTower(self, 32 - 6.5, 3.5, 2, 2)
        self.princess_tower_2 = CrownTower(self, 32 - 6.5, 18 - 3.5, 2, 2)
        
        self.objects = [
            self.king_tower, self.princess_tower_1, self.princess_tower_2
        ]