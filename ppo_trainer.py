import os
import random
import numpy as np

from addict import Dict

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import ActorCritic

import gymnasium as gym
import cr_gym_env
from cr_flatten_norm_wrapper import CRFlattenNormWrapper

import wandb


class Trainer:
    def __init__(self, gym_env_name):
        self.gym_env_name = gym_env_name
        self.env = gym.make(self.gym_env_name)
        self.env = CRFlattenNormWrapper(self.env)

        self.arena = self.env.env.env.env.arena
        self.max_num_objects = self.arena.max_num_objects


        self.cfg = Dict()

        self.cfg.seed = 42
        self.set_seed(self.cfg.seed)

        # PPO Specific
        self.cfg.gamma = 0.99
        self.cfg.gae_lambda = 0.95
        self.cfg.clip_eps = 0.2  # PPO clip range
        self.cfg.k_epochs = 10   # gradient steps per rollout
        self.cfg.rollout_steps = 2048  # timesteps to collect before each update
        self.cfg.minibatch_size = 128
        self.cfg.lr = 3e-4 

        self.cfg.max_entropy_coef = 0.005
        self.cfg.min_entropy_coef = 0.005
        self.cfg.entropy_coef = None

        self.cfg.vf_coef = 0.5  # critic loss weight in combined loss
        self.cfg.grad_clip = 0.5
        self.cfg.max_steps = 100_000_000  # total env steps

        # Network Config
        self.cfg.network.entity_encoder_in_ch = self.env.flat_card_space.shape[0]
        self.cfg.network.entity_encoder_mid_ch = 64
        self.cfg.network.entity_encoder_out_ch = 32

        self.cfg.network.trunk_extra_in_ch = 2
        self.cfg.network.trunk_mid_ch = 128
        
        self.cfg.network.num_cards_in_deck = self.env.env.env.env.NUM_CARDS_IN_DECK
        self.cfg.network.max_num_cards = self.max_num_objects
        self.cfg.network.position_space_width = self.arena.width
        self.cfg.network.position_space_height = self.arena.height

        # Replay storing
        self.video_dir = "./videos"
        self.video_every = 250_000
        os.makedirs(self.video_dir, exist_ok=True)

        # WANDB logging
        self.wandb_logging = False
        # self.wandb_logging = True

        if self.wandb_logging:
            wandb.init(
                project="clash_royale-ppo_self_play",
                config=self.cfg.to_dict()
            )


    def train(self):
        self.set_seed(self.cfg.seed)
        state, _ = self.env.reset()

        ep_return = 0.0
        ep_returns = []

        global_step = 0
        next_video = 0

        self.record_episode(step=0, player1_weights=None, player2_weights=None)

        # while global_step < self.cfg.max_steps:
        #     ...


    def get_network_and_optimiser(self, weights=None):
        """
        Self-Play PPO samples from a pool of policies, so the weights arg 
        inits the network with those STRICTLY
        """

        network = ActorCritic(**self.cfg.network.to_dict())
        # network = ActorCritic(
        #     self.cfg.network.entity_encoder_in_ch,
        #     self.cfg.network.entity_encoder_mid_ch, 
        #     self.cfg.network.entity_encoder_out_ch,

        #     self.cfg.network.trunk_extra_in_ch,
        #     self.cfg.network.trunk_mid_ch,

        #     self.cfg.network.num_cards_in_deck,
        #     self.cfg.network.position_space_width,
        #     self.cfg.network.position_space_height,
        # )
        
        if weights:
            network.load_state_dict(t.load(weights, weights_only=True))

        optimiser = optim.Adam(network.parameters(), lr=self.cfg.lr)

        return network, optimiser


    def split_observations(self, obs):
        player_1_num_cards = len(obs["player_1_cards"])
        player_2_num_cards = len(obs["player_2_cards"])

        if player_1_num_cards == 0:
            obs["player_1_cards"] = [np.zeros(self.env.flat_card_space.shape)]
            player_1_num_cards = 1
        if player_2_num_cards == 0:
            obs["player_2_cards"] = [np.zeros(self.env.flat_card_space.shape)]
            player_2_num_cards = 1

        player_1_cards = F.pad(
            t.tensor(np.array(obs["player_1_cards"])),  # (X, 26)
            (0, 0, 0, self.max_num_objects - player_1_num_cards),            # (N, 26)
            "constant", 0
        ).unsqueeze(0)

        player_2_cards = F.pad(
            t.tensor(np.array(obs["player_2_cards"])),
            (0, 0, 0, self.max_num_objects - player_2_num_cards),
            "constant", 0
        ).unsqueeze(0)

        player_1_crown_towers = t.tensor(np.array(obs["player_1_crown_towers"])).unsqueeze(0)
        player_2_crown_towers = t.tensor(np.array(obs["player_2_crown_towers"])).unsqueeze(0)

        obs_1 = {
            "game_completion_fraction": t.tensor(np.array(obs["game_completion_fraction"])).reshape(-1, 1),
            "elixirs":                  t.tensor(np.array(obs["player_1_elixirs"])).reshape(-1, 1),
            "my_cards":                 player_1_cards,
            "opponent_cards":           player_2_cards,
            "my_crown_towers":          player_1_crown_towers,
            "opponent_crown_towers":    player_2_crown_towers,
        }

        # Indices 5 & 6 are x & y coords for their position
        # Flipping them is just a 180 degree rotation about the origin
        # (x, y) -> (width_cell / 2 - x, height_cell / 2 - y)
        width_cell  = self.arena.tile_size * self.arena.width
        height_cell = self.arena.tile_size * self.arena.height
        
        # Doing this to prevent zero padded entries to falsely invert
        player_1_cards = t.tensor(np.array(obs["player_1_cards"]))
        player_2_cards = t.tensor(np.array(obs["player_2_cards"]))

        player_1_cards[..., 5] = width_cell/2  - player_1_cards[..., 5]
        player_1_cards[..., 6] = height_cell/2 - player_1_cards[..., 6]

        player_2_cards[..., 5] = width_cell/2  - player_2_cards[..., 5]
        player_2_cards[..., 6] = height_cell/2 - player_2_cards[..., 6]

        player_1_cards = F.pad(
            player_1_cards,  # (X, 26)
            (0, 0, 0, self.max_num_objects - player_1_num_cards),            # (N, 26)
            "constant", 0
        ).unsqueeze(0)

        player_2_cards = F.pad(
            player_2_cards,
            (0, 0, 0, self.max_num_objects - player_2_num_cards),
            "constant", 0
        ) .unsqueeze(0)

        player_1_crown_towers[..., 5] = width_cell/2  - player_1_crown_towers[..., 5]
        player_1_crown_towers[..., 6] = height_cell/2 - player_1_crown_towers[..., 6]

        player_2_crown_towers[..., 5] = width_cell/2  - player_2_crown_towers[..., 5]
        player_2_crown_towers[..., 6] = height_cell/2 - player_2_crown_towers[..., 6]

        obs_2 = {
            "game_completion_fraction": t.tensor(np.array(obs["game_completion_fraction"])).reshape(-1, 1),
            "elixirs":                  t.tensor(np.array(obs["player_2_elixirs"])).reshape(-1, 1),
            "my_cards":                 player_2_cards,
            "opponent_cards":           player_1_cards,
            "my_crown_towers":          player_2_crown_towers,
            "opponent_crown_towers":    player_1_crown_towers,
        }

        return obs_1, obs_2
    

    def record_episode(self, step, player1_weights=None, player2_weights=None):
        """Run one greedy episode and save video to self.video_dir."""

        # TODO: move to __init__
        rec_env = gym.make(self.gym_env_name, render_mode="rgb_array")
        rec_env = CRFlattenNormWrapper(rec_env)
        rec_env = gym.wrappers.RecordVideo(
            rec_env,
            video_folder=self.video_dir,
            name_prefix=f"step{step:07d}",
            episode_trigger=lambda _: True,
        )
        
        net_1, _ = self.get_network_and_optimiser(player1_weights)
        net_2, _ = self.get_network_and_optimiser(player2_weights)
        state, _ = rec_env.reset()
        
        ep_return = np.zeros(2)
        while True:
            state_1, state_2  = self.split_observations(state)
        
            with t.no_grad():
                action_1, _, _, _ = net_1.get_action_and_value(state_1)
                action_2, _, _, _ = net_2.get_action_and_value(state_2)
        
            action = {
                "player_1_skip":          int(action_1["skip"] > 0.5),
                "player_1_card_idx":      action_1["deck_idx"],
                "player_1_card_position": (
                    action_1["position"] % self.arena.width,
                    action_1["position"] // self.arena.height
                ),

                "player_2_skip":          int(action_2["skip"] > 0.5),
                "player_2_card_idx":      action_2["deck_idx"],
                "player_2_card_position": (
                    action_2["position"] % self.arena.width,
                    action_2["position"] // self.arena.height
                ),
            }

            state, reward, terminated, truncated, _ = rec_env.step(action)
        
            ep_return += np.array(reward)
        
            if terminated or truncated:
                break
        
        rec_env.close()
        
        print(f"\t [video] step {step}:\t return: {ep_return:.1f}")
        
        return ep_return

    
    def set_seed(self, seed: int = 42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        t.manual_seed(seed)
        t.cuda.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False


if __name__ == "__main__":
    trainer = Trainer(gym_env_name="ClashRoyaleEnv-v0")

    trainer.train()