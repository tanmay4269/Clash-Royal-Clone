import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from arena import Arena
from entity import EntityType


register(
    id="ClashRoyaleEnv-v0",
    entry_point="gym_env:ClashRoyaleEnv",
)

class ClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.arena = Arena()

        ### Observation Space ###
        INT_MAX = 1_000_000
        FLOAT_MAX = 1_000_000.0
        
        # x, y
        # TODO: reconsider if this should rather be discrete
        position_space = spaces.Box(  
            low=np.array([0.0, 0.0]),
            high=np.array([
                self.arena.width * self.arena.tile_size,
                self.arena.height * self.arena.tile_size
            ])
        )

        def get_entity_space(is_card=True):
            # either its a card or its a crown tower

            ret = {
                "position": position_space,
                
                "health": spaces.Discrete(INT_MAX),
                "hitpoints": spaces.Discrete(INT_MAX),

                "damage": spaces.Discrete(INT_MAX),
                "attack_radius_cells": spaces.Discrete(INT_MAX),

                "hit_speed": spaces.Box(0.0, FLOAT_MAX),
                "first_hit_speed": spaces.Box(0.0, FLOAT_MAX),
                
                # Maybe add what stage of its hit cycle is it if the model can't seem to figure this out
            }

            if is_card:
                ret.update({
                    "deploy_cost": spaces.Discrete(self.arena.player_side_1.max_elixirs),
                    "deploy_delay": spaces.Box(0.0, FLOAT_MAX),
                    
                    "entity_type": spaces.Discrete(EntityType.num_types()), 

                    "target_types": spaces.MultiBinary(EntityType.num_types())
                })

            return spaces.Dict(ret)

        crown_towers_space = spaces.Dict({
            "king_tower": get_entity_space(is_card=False),
            "princess_tower_1": get_entity_space(is_card=False),
            "princess_tower_2": get_entity_space(is_card=False),
        })

        self.observation_space = spaces.Dict({
            "game_completion_fraction": spaces.Box(0.0, 1.0),
            "player_1_elixirs": spaces.Discrete(self.arena.player_side_1.max_elixirs),
            "player_1_cards": spaces.Sequence(get_entity_space()),
            "player_1_crown_towers": crown_towers_space,

            "player_2_elixirs": spaces.Discrete(self.arena.player_side_2.max_elixirs),
            "player_2_cards": spaces.Sequence(get_entity_space()),
            "player_2_crown_towers": crown_towers_space,
        })

        ### Action Space ###
        NUM_CARDS_IN_DECK = 3  # TODO: ofc find a better way

        self.action_space = spaces.Dict({
            "player_1_skip": spaces.Discrete(2), # bool
            "player_1_card_idx": spaces.Discrete(NUM_CARDS_IN_DECK),
            "player_1_card_position": position_space,

            "player_2_skip": spaces.Discrete(2), # bool
            "player_2_card_idx": spaces.Discrete(NUM_CARDS_IN_DECK),
            "player_2_card_position": position_space,
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen = None
        self.clock = None

        self.FIXED_DT = 1/4  # 4 fps simulator

    def _get_obs(self):
        obs = self.observation_space.sample()
        # TODO overwrite properly
        return obs

    def _get_info(self):
        """
        Oftentimes, info will also contain some data that is only available inside the step method 
        (e.g., individual reward terms). In that case, we would have to update the dictionary that 
        is returned by _get_info in step.
        """
        # TODO
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # TODO: actually take action, something like arena.on_click

        terminated, truncated = self.arena.update(self.FIXED_DT)
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_reward(self):
        # TODO: sum of damages to the opponents towers - our towers
        return 0.0

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
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            rgb_array = pygame.surfarray.array3d(self.screen)
            rgb_array = rgb_array.transpose(1, 0, 2)  # pygame is (w,h,c), numpy expects (h,w,c)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    # quick test
    from rich import print

    env = ClashRoyaleEnv()  # Works but the latter has some useful checks
    # env = gym.make("ClashRoyaleEnv-v0")
    state, _ = env.reset()

    # print("state", state)

    done = False

    while not done:
        action = env.action_space.sample()
        next_state, reward, termination, truncated, _ = env.step(action)

        # print("-----")
        # print("action", action)
        # print("next_state", next_state)
        # print("reward", reward)
        # print("termination", termination)
        # print("truncated", truncated)
        print(next_state["game_completion_fraction"])

        done = termination or truncated