from utils import *
from arena import Arena

import psutil
import os


class Game:
    def __init__(self):
        self.arena = Arena()
        
        # * For simplicity, each sub-tile cell is a pixel
        self.width = self.arena.width * self.arena.tile_size
        self.height = self.arena.height * self.arena.tile_size

        # pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.dt = 0


    def update(self):
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left mouse button
                    self.arena.on_click()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.arena._debug_active_player = 1
                    print("Setting active player to `1`")
                elif event.key == pygame.K_2:
                    self.arena._debug_active_player = 2
                    print("Setting active player to `2`")

        self.arena.render(self.screen)
        if not self.arena.update(self.dt):
            self.running = False
            return

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        self.dt = self.clock.tick(60) / 1000

        ### * DEBUG * ###
        if False:
        # if True:
            process = psutil.Process(os.getpid())
            memory = process.memory_info().rss  # in bytes

            print(f"FPS: {self.clock.get_fps():.2f}\t RAM Usage: {memory / 1024 / 1024:.2f} MB")


    def run(self):
        while self.running:
            self.update()

        pygame.quit()

