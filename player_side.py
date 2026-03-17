class PlayerSide:
    def __init__(self):
        # TODO: Replace with actual tower instances 
        self.king_alive = True
        self.princess_1_alive = True  # The one closer to (0, 0) 
        self.princess_2_alive = True

        # TODO: GameBoard should manage this instead
        self.elixirs = 5  # init
        self.max_elixirs = 10
        self.elixirs_per_sec = 1
