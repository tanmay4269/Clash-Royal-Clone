import gymnasium as gym
from gymnasium import spaces

from arena import Arena


class ClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.arena = Arena()

        # TODO
        self.observation_space = spaces.Dict({
                "game_time": ...,
                "my_elixirs": ...,
                "my_troops": spaces.Sequence(
                    spaces.Dict({
                        "position_x": spaces.Box(
                            low=0,
                            high=self.arena.width * self.arena.tile_size,
                            dtype=float
                        ),
                        # position y
                        # health
                        # hitpoints
                        # etc.
                    }),
                ),
                "my_towers": ...,
                "opponent_troops": ...,
                "opponent_towers": ...,
            })

        # TODO
        self.action_space = spaces.Dict({
            "do_i_play": ..., # bool
            "card_idx": ...,
            "card_position": ...,  # Tile index
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen = None
        self,clock = None
        self.dt = 0

    def _get_obs(self):
        # TODO
        ...

    def _get_info(self):
        """
        Oftentimes, info will also contain some data that is only available inside the step method 
        (e.g., individual reward terms). In that case, we would have to update the dictionary that 
        is returned by _get_info in step.
        """
        # TODO
        ...

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        terminated, truncated = self.arena.update(self.dt)  
            # ! Major draw back, would need to wait for real time seconds while this could
            # potentially all be sped up... but idk a better way than a multiplier on self.dt!
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_reward(self):
        # TODO
        ...

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.arena.render(self.screen)
            pygame.display.flip()
            self.dt = self.clock.tick(self.metadata["render_fps"]) / 1000
        else:  # rgb_array
            # TODO: return game window as a numpy array
            ...

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
