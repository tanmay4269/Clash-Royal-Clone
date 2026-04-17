from utils import *
from cr_flatten_norm_wrapper import CRFlattenNormWrapper

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from arena import Arena
from entity import EntityType, Entity
from entities.buildings.crown_tower import CrownTower

from entities.troops import *


register(
    id="ClashRoyaleEnv-v0",
    entry_point="cr_gym_env:ClashRoyaleEnv",
)

class ClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.arena = Arena()

        ### Observation Space ###
        INT_MAX = 1_000_000
        FLOAT_MAX = 1_000_000.0

        def get_entity_space():
            return spaces.Dict({
                "deploy_cost": spaces.Box(0.0, self.arena.player_side_1.max_elixirs),
                "deploy_delay": spaces.Box(0.0, 10.0),
                
                "entity_type": spaces.Discrete(EntityType.num_types()), 

                "target_types": spaces.MultiBinary(EntityType.num_types()), 

                # x, y
                # TODO: reconsider if this should rather be discrete
                "position": spaces.Box(  
                    low=np.array([0.0, 0.0]),
                    high=np.array([
                        self.arena.width * self.arena.tile_size,
                        self.arena.height * self.arena.tile_size
                    ])
                ),
                
                # "health": spaces.Box(0.0, FLOAT_MAX),
                "health": spaces.Box(0.0, EntityRegistry.aggregate("health")["max"]),
                "hitpoints": spaces.Box(0.0, EntityRegistry.aggregate("hitpoints")["max"]),

                "damage": spaces.Box(0.0, EntityRegistry.aggregate("damage")["max"]),
                "attack_radius_cells": spaces.Box(0.0, EntityRegistry.aggregate("attack_radius_cells")["max"]),

                "hit_speed": spaces.Box(0.0, EntityRegistry.aggregate("hit_speed")["max"]),
                "first_hit_speed": spaces.Box(0.0, EntityRegistry.aggregate("first_hit_speed")["max"]),
                
                # Maybe add what stage of its hit cycle is it if the model can't seem to figure this out
            })

        crown_towers_space = spaces.Dict({
            "king_tower": get_entity_space(),
            "princess_tower_1": get_entity_space(),
            "princess_tower_2": get_entity_space(),
        })

        self.observation_space = spaces.Dict({
            "game_completion_fraction": spaces.Box(0.0, 1.0),
            "player_1_elixirs": spaces.Discrete(self.arena.player_side_1.max_elixirs + 1),
            "player_1_cards": spaces.Sequence(get_entity_space()),
            "player_1_crown_towers": crown_towers_space,

            "player_2_elixirs": spaces.Discrete(self.arena.player_side_2.max_elixirs + 1),
            "player_2_cards": spaces.Sequence(get_entity_space()),
            "player_2_crown_towers": crown_towers_space,
        })

        ### Action Space ###
        self.NUM_CARDS_IN_DECK = 3  # TODO: ofc find a better way
        position_space = spaces.Box(  
            low=np.array([0.0, 0.0]),
            high=np.array([
                self.arena.width,
                self.arena.height
            ]),
            dtype=int
        )

        self.action_space = spaces.Dict({
            "player_1_skip": spaces.Discrete(2), # bool
            "player_1_card_idx": spaces.Discrete(self.NUM_CARDS_IN_DECK),
            "player_1_card_position": position_space,

            "player_2_skip": spaces.Discrete(2), # bool
            "player_2_card_idx": spaces.Discrete(self.NUM_CARDS_IN_DECK),
            "player_2_card_position": position_space,
        })

        ### Helpers ###
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen = None
        self.clock = None

        self.FIXED_DT = 1/4  # 4 fps simulator

        self._cur_obs = None


    def _get_obs(self):
        obs = {}

        ### Basics ###
        obs["game_completion_fraction"] = self.arena.elapsed_time / self.arena.game_duration

        obs["player_1_elixirs"] = int(self.arena.player_side_1.elixirs)
        obs["player_2_elixirs"] = int(self.arena.player_side_2.elixirs)

        ### Cards ###
        obs["player_1_cards"], obs["player_2_cards"] = [], []
        for obj in self.arena.objects:
            if isinstance(obj, CrownTower):
                continue
            
            owner = obs["player_1_cards"] if obj.owner == self.arena.player_side_1 else obs["player_2_cards"]
            owner.append({
                "deploy_cost": obj.deploy_cost,
                "deploy_delay": obj.deploy_delay,

                "entity_type": obj.entity_type,
                "target_types": np.array([
                    int(EntityType.GROUND   in obj.target_types),
                    int(EntityType.AIR      in obj.target_types),
                    int(EntityType.BUILDING in obj.target_types),
                ], dtype=np.int8),

                "position": np.array([obj.position.x, obj.position.y]),
                
                "health": obj.health,
                "hitpoints": obj.hitpoints,
                
                "damage": obj.damage,
                "attack_radius_cells": obj.attack_radius_cells,

                "hit_speed": obj.hit_speed,
                "first_hit_speed": obj.first_hit_speed,
            })


        ### Crown Towers ###
        obs["player_1_crown_towers"], obs["player_2_crown_towers"] = {}, {}

        # TODO: this is too hidious, should have a getter in the entity class
        for side_name in ["player_1", "player_2"]:
            side = self.arena.player_side_1 if side_name == "player_1" else \
                self.arena.player_side_2

            for tower_name in ["king_tower", "princess_tower_1", "princess_tower_2"]:
                tower = None
                if tower_name == "king_tower":
                    tower = side.king_tower
                elif tower_name == "princess_tower_1":
                    tower = side.princess_tower_1
                else:
                    tower = side.princess_tower_2

                obs[f"{side_name}_crown_towers"][tower_name] = {
                    "deploy_cost": tower.deploy_cost,
                    "deploy_delay": tower.deploy_delay,

                    "entity_type": tower.entity_type,
                    "target_types": np.array([
                        int(EntityType.GROUND   in tower.target_types),
                        int(EntityType.AIR      in tower.target_types),
                        int(EntityType.BUILDING in tower.target_types),
                    ], dtype=np.int8),
                    
                    "position": np.array([tower.position.x, tower.position.y]),

                    "health": tower.health,
                    "hitpoints": tower.hitpoints,

                    "damage": tower.damage,
                    "attack_radius_cells": tower.attack_radius_cells,

                    "hit_speed": tower.hit_speed,
                    "first_hit_speed": tower.first_hit_speed
                }

        return obs


    def _get_info(self):
        """
        Oftentimes, info will also contain some data that is only available inside the step method 
        (e.g., individual reward terms). In that case, we would have to update the dictionary that 
        is returned by _get_info in step.
        """
        # TODO (maybe)
        return {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._cur_obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return self._cur_obs, info


    def step(self, action):
        for idx in [1, 2]:
            if action[f"player_{idx}_skip"] == 1:
                continue

            owner = self.arena.player_side_2 if idx == 1 \
                else self.arena.player_side_1

            x, y = action[f"player_{idx}_card_position"]
            
            # TODO: use entity registers instead
            card = None
            if action[f"player_{idx}_card_idx"] == 0:
                card = Knight
            elif action[f"player_{idx}_card_idx"] == 1:
                card = Giant
            elif action[f"player_{idx}_card_idx"] == 2:
                card = MiniPEKKA

            card_instance = card(owner, y, x)
            if self.arena.deploy_entity(card_instance):
                owner.add_object(card_instance)

        prev_obs = self._cur_obs  # Saving for reward calculation
        
        # For stable physics updates
        terminated, truncated = False, False
        for _ in range(10):
            _terminated, _truncated = self.arena.update(self.FIXED_DT / 10)
            terminated, truncated = terminated or _terminated, truncated or _truncated
        
        self._cur_obs = self._get_obs()

        reward = self._get_reward(prev_obs)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return self._cur_obs, reward, terminated, truncated, info


    def _get_reward(self, prev_obs):
        """
        TODOs:
        - additional reward when the princess towers are distroyed 
        - additional reward when the game is one by someone and lost by the other
        """

        if prev_obs is None:
            return 0.0, 0.0

        player_1_reward, player_2_reward = 0.0, 0.0

        for idx in [1, 2]:
            cur_dict  = self._cur_obs[f"player_{idx}_crown_towers"]
            prev_dict = prev_obs[f"player_{idx}_crown_towers"] 

            player_idx_reward = player_1_reward if idx == 1 else player_2_reward

            for tower_str in cur_dict.keys():
                delta = cur_dict[tower_str]["health"] - prev_dict[tower_str]["health"]

                if idx == 1:
                    player_1_reward += delta
                    player_2_reward -= delta
                else:
                    player_1_reward -= delta
                    player_2_reward += delta

        return player_1_reward, player_2_reward


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        screen_w = self.arena.width  * self.arena.tile_size
        screen_h = self.arena.height * self.arena.tile_size

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_w, screen_h))
            else:  
                # rgb_array: use an offscreen surface, no display needed
                self.screen = pygame.Surface((screen_w, screen_h))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        self.arena.render(self.screen)

        if self.render_mode == "human":
            self.arena.render(self.screen)
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])  # TODO: do i even need this!?
        else:  # rgb_array
            rgb_array = pygame.surfarray.array3d(self.screen)
            rgb_array = rgb_array.transpose(1, 0, 2)  # pygame is (w,h,c), numpy expects (h,w,c)
            return rgb_array


    def close(self):
        if self.screen is not None:
            if self.render_mode == "human":
                pygame.display.quit()
            pygame.quit()
            self.screen = None


if __name__ == "__main__":
    # quick test

    # env = ClashRoyaleEnv()  # Works but the latter has some useful checks
    env = gym.make("ClashRoyaleEnv-v0", render_mode="rgb_array")

    env = CRFlattenNormWrapper(env)

    env = gym.wrappers.RecordVideo(
        env,
        video_folder="./videos",
        name_prefix=f"debug",
        episode_trigger=lambda _: True,
    )
    
    state, _ = env.reset()
    done = False

    player_1_spawn_cooldown = 0.0
    player_2_spawn_cooldown = 0.0

    player_1_spawn = {
        "player_1_skip": 0,
        "player_1_card_idx": 0,
        "player_1_card_position": np.array([9, 32-10]),

        "player_2_skip": 1,
        "player_2_card_idx": 0,
        "player_2_card_position": np.array([9, 10]),
    }

    player_2_spawn = {
        "player_1_skip": 1,
        "player_1_card_idx": 0,
        "player_1_card_position": np.array([9, 32-10]),

        "player_2_skip": 0,
        "player_2_card_idx": 0,
        "player_2_card_position": np.array([9, 10]),
    }

    try:
        while not done:
            if player_1_spawn_cooldown > 0.25:
                player_1_spawn_cooldown = 0.0
                action = player_1_spawn
            elif player_2_spawn_cooldown > 0.2:
                player_2_spawn_cooldown = 0.0
                action = player_2_spawn
            else:
                action = env.action_space.sample()
                action["player_1_skip"] = 1
                action["player_2_skip"] = 1

            next_state, reward, termination, truncated, _ = env.step(action)

            player_1_spawn_cooldown += 1/100
            # player_2_spawn_cooldown += 1/100

            print("-----")
            # print(next_state["game_completion_fraction"])
            print("action", action)
            print("next_state", next_state)
            # print("reward", reward)
            # print("termination", termination)
            # print("truncated", truncated)

            done = termination or truncated
    except Exception as e:
        print(e)

    env.close()
