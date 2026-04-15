import os
import random
import numpy as np

from addict import Dict

import torch as t
import torch.nn as nn
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

        self.cfg.network.trunk_extra_in_ch = 2 + 3 * self.cfg.network.entity_encoder_out_ch
        self.cfg.network.trunk_mid_ch = 128
        
        self.cfg.network.num_cards_in_deck = self.env.NUM_CARDS_IN_DECK
        self.cfg.network.position_space_width = self.env.arena.width
        self.cfg.network.position_space_height = self.env.arena.height

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

        self.record_episode(step=0, weights=None)

        while global_step < self.cfg.max_steps:
            ...


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

    
    def record_episode(self, step, weights=None):
        """Run one greedy episode and save video to self.video_dir."""

        rec_env = gym.make(self.gym_env_name, render_mode="rgb_array")
        rec_env = CRFlattenNormWrapper(rec_env)
        rec_env = gym.wrappers.RecordVideo(
            rec_env,
            video_folder=self.video_dir,
            name_prefix=f"step{step:07d}",
            episode_trigger=lambda _: True,
        )
        
        net, _ = self.get_network_and_optimiser(weights)
        state, _ = rec_env.reset()
        
        ep_return = 0.0
        while True:
            with torch.no_grad():
                action, _, _, _ = net.get_action_and_value(state)
        
            state, reward, terminated, truncated, _ = rec_env.step(action)
        
            ep_return += reward
        
            if terminated or truncated:
                break
        
        rec_env.close()
        
        print(f"\t [video] step {step}:\t return: {ep_return:.1f}")
        
        return ep_return


    
    def set_seed(seed: int = 42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    trainer = Trainer(gym_env_name="ClashRoyaleEnv-v0")

    print(trainer.env.flat_card_space.shape[0])