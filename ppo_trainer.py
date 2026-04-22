import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")  # TODO: fix code later


import os
import time
from copy import deepcopy

import random
import numpy as np

from addict import Dict
from rich import print

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import ActorCritic

import gymnasium as gym
import cr_gym_env
from cr_flatten_norm_wrapper import CRFlattenNormWrapper

from rollout_buffer import RolloutBuffer
from checkpoint_management import *

import wandb


class Trainer:
    def __init__(self, gym_env_name):
        t.set_default_dtype(t.float32)

        if t.cuda.is_available():
            device = t.device("cuda")
        else:
            device = t.device("cpu")

        print(f"Using device: {device}")

        t.set_default_device(device)


        self.gym_env_name = gym_env_name

        self.env = gym.make(self.gym_env_name)
        # self.num_envs = 4
        # self.env = gym.vector.make(self.gym_env_name, num_envs=self.num_envs, asynchronous=True)

        self.env = CRFlattenNormWrapper(self.env)

        self.arena = self.env.unwrapped.arena
        self.max_num_objects = self.arena.max_num_objects

        #
        occupancy_grid = self.arena.cell_occupancy
        scale = self.arena.tile_size

        tiled_occupancy_grid = np.where(occupancy_grid == 1, 1, 0)[scale//2::scale, scale//2::scale]
        invalid_position_mask = tiled_occupancy_grid.astype(bool).T
        invalid_position_mask[: self.arena.height//2, :] = True  # other player's half is always invalid

        ### CONFIGS ###
        self.cfg = Dict()

        self.cfg.seed = 42
        self.set_seed(self.cfg.seed)

        # Network Config
        self.cfg.network.entity_encoder_in_ch = self.env.flat_card_space.shape[0]
        self.cfg.network.entity_encoder_mid_ch = 64
        self.cfg.network.entity_encoder_out_ch = 32

        self.cfg.network.trunk_extra_in_ch = 2
        self.cfg.network.trunk_mid_ch = 128
        
        self.cfg.network.num_cards_in_deck = self.env.unwrapped.NUM_CARDS_IN_DECK
        self.cfg.network.max_num_cards = self.max_num_objects
        self.cfg.network.position_space_width = self.arena.width
        self.cfg.network.position_space_height = self.arena.height

        self.cfg.network.invalid_position_mask = t.tensor(invalid_position_mask).flatten()

        # Buffer Related
        self.cfg.buffer.gae_gamma = 0.99
        self.cfg.buffer.gae_lambda = 0.95

        self.cfg.buffer.n_steps = int(self.arena.game_duration * 1/self.env.unwrapped.FIXED_DT * 10)  # Usually 10 to 100 episodes
        if os.environ.get("DEBUG_MODE"):
            self.cfg.buffer.n_steps = 2048

        # Elo Rating
        self.cfg.elo.initial_rating = 1200
        self.cfg.elo.scale = 400
        self.cfg.elo.k_factor = 32

        self.current_elo = self.cfg.elo.initial_rating

        # Player Pool
        # self.checkpoint_manager = AdvancedTemporal_CheckpointManagement(
        #     checkpoint_dir="./checkpoints",
        #     loading_latest_ratio=0.5,
        #     loading_delta_window=0.2,
        #     min_games_before_checkpointing=100,
        #     score_queue_size=100,
        #     avg_score_threshold=0.55,
        # )

        self.checkpoint_manager = AdvancedEloBased_CheckpointManagement(
            checkpoint_dir="./checkpoints",
            elo_cfg=self.cfg.elo,
            loading_latest_ratio=0.5,
            min_games_before_checkpointing=100,
            score_queue_size=100,
            avg_score_threshold=0.55
        )

        # PPO Update
        self.cfg.ppo_clip = 0.2
        self.cfg.grad_clip = 0.5

        # Loss
        self.cfg.lr = 3e-4 * 4  # sqrt n law
        self.cfg.critic_loss_coef = 0.5
        self.entropy_loss_coef = 0.01

        # Misc
        self.cfg.minibatch_size = 2048
        if os.environ.get("DEBUG_MODE"):
            self.cfg.minibatch_size = 128

        self.cfg.k_epochs = 3  # gradient steps per rollout
        self.cfg.max_steps = 1_000_000_000  # total env steps

        # Replay storing
        self.video_dir = "./videos/try_4"
        self.video_every_k_global_steps = 20_000
        if os.environ.get("DEBUG_MODE"):
            self.video_every_k_global_steps = 5_000
        os.makedirs(self.video_dir, exist_ok=True)

        # WANDB logging
        self.wandb_logging = True
        if os.environ.get("DEBUG_MODE"):
            self.wandb_logging = False

        if self.wandb_logging:
            wandb.init(
                project="clash_royale-ppo_self_play",
                config=self.cfg.to_dict()
            )


    def train(self):
        self.set_seed(self.cfg.seed)

        ep_return = np.zeros(2)
        ep_returns = []

        global_step = 0
        next_video = self.video_every_k_global_steps

        net_1, optimiser_1 = self.get_network_and_optimiser()
        net_2 = deepcopy(net_1)
        opponent_elo = self.cfg.elo.initial_rating

        state, _ = self.env.reset()
        state_1, state_2  = self.split_observations(state)

        while global_step < self.cfg.max_steps:
            buffer = RolloutBuffer(**self.cfg.buffer.to_dict())

            for _ in range(self.cfg.buffer.n_steps):
                with t.no_grad():
                    action_1, log_prob_1, entropy_1, value_1 = net_1.get_action_and_value(state_1)
                    action_2, _, _, _ = net_2.get_action_and_value(state_2) 

                action = self.join_actions(action_1, action_2)

                try:
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                except Exception as e:
                    print(e)
                    terminated = True

                done = terminated or truncated

                # buffer.push(state_1, action_1, log_prob_1, reward[0], value_1, done)
                buffer.push(state_1, action_1, log_prob_1, reward[0], value_1, terminated)
                """
                If a game ends simply because time ran out (truncated), GAE will set (1 - next_done) to 0. 
                This incorrectly forces the value of the final state to 0, making the critic think running out 
                of time violently acts as a massive negative return drop.
                Solution: PPO should only mask out values if the agent actually "dies" or the game ends artificially.
                """

                state = next_state
                state_1, state_2  = self.split_observations(state)

                ep_return += np.array(reward)

                if done:
                    # TODO: get this from the env instead
                    score = 0
                    if ep_return[0] > ep_return[1]:
                        score = 1
                    elif ep_return[0] == ep_return[1]:
                        score = 0.5

                    # ELO update
                    E_A = 1 / (1 + 10 ** ((opponent_elo - self.current_elo) / self.cfg.elo.scale))
                    self.current_elo = self.current_elo + self.cfg.elo.k_factor * (score - E_A)

                    if self.wandb_logging:
                        wandb.log(
                            {
                                "elo": self.current_elo,
                                "return": ep_return[0],
                                "score": score
                            }, 
                            step=global_step
                        )

                    opponent_elo = self.checkpoint_manager.load(net_2, self.current_elo)
                    self.checkpoint_manager.update(net_1, score, self.current_elo)

                    ep_returns.append(ep_return)
                    ep_return = np.zeros(2)

                    ep_seed = np.random.randint(0, 2**31)
                    state, _ = self.env.reset(seed=ep_seed)
                    state_1, state_2  = self.split_observations(state)

                global_step += 1

                if global_step >= next_video:
                    self.record_episode(global_step, net_1, net_2)
                    next_video += self.video_every_k_global_steps

            # Bootstrap final value and compute GAE
            with t.no_grad():
                _, _, _, last_value_1 = net_1.get_action_and_value(state_1)
                # buffer.compute_gae(last_value_1, done)
                buffer.compute_gae(last_value_1, terminated)

            # PPO update
            if os.environ.get("PROFILE_MODE"):
                ppo_update_start_time = time.time()
            
            actor_loss, critic_loss, entropy, ratio_mean, advantage_mean = self.ppo_update(buffer, net_1, optimiser_1)
            
            if os.environ.get("PROFILE_MODE"):
                ppo_update_end_time = time.time()
                print(f"\t PPO update time: {ppo_update_end_time - ppo_update_start_time:.3f} seconds")

            buffer.reset()

            avg_100 = np.mean(ep_returns[-100:], axis=0)
            avg_100_player_1 = avg_100[0]

            print(
                f"step {global_step:7d} | avg(last 100 eps): {avg_100_player_1:10.3f} | "
                f"actor_loss: {actor_loss:7.3f} | critic_loss: {critic_loss:7.3f} | entropy: {entropy:.3f}"
            )

            if self.wandb_logging:
                wandb.log({
                    "actor_loss":     actor_loss,
                    "critic_loss":    critic_loss,
                    "entropy":        entropy,
                    "ratio_mean":     ratio_mean,
                    "advantage_mean": advantage_mean
                }, step=global_step)

    
    def ppo_update(self, buffer, net, optimiser):
        # TODO: use an average meter instead 
        num_steps = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_ratio_mean = 0
        total_advantage_mean = 0

        for epoch in range(self.cfg.k_epochs):
            for batch in buffer.get_minibatches(self.cfg.minibatch_size):
                states, actions, old_log_probs, advantages, returns = batch

                # Actor Loss
                _, new_log_probs, entropies, values = net.get_action_and_value(states, actions)
                ratio = (new_log_probs - old_log_probs).exp()

                actor_loss = -t.min(
                    advantages * ratio, 
                    advantages * t.clip(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip)
                ).mean()

                # Critic Loss
                G = returns
                V = values
                # critic_loss = ( (G - V) ** 2 ).mean()
                critic_loss = F.smooth_l1_loss(input=V, target=G)

                # Backprop
                loss = actor_loss + self.cfg.critic_loss_coef * critic_loss - self.entropy_loss_coef * entropies.mean()

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),  self.cfg.grad_clip)
                optimiser.step()

                num_steps += 1
                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy     += entropies.mean().item()

                total_ratio_mean     += ratio.mean().item()
                total_advantage_mean += advantages.mean().item()

        return (
            total_actor_loss     / num_steps,
            total_critic_loss    / num_steps, 
            total_entropy        / num_steps,
            total_ratio_mean     / num_steps,
            total_advantage_mean / num_steps,
        )


    def record_episode(self, step, net_1, net_2):
        """Run one greedy episode and save video to self.video_dir."""

        rec_env = gym.make(self.gym_env_name, render_mode="rgb_array")
        rec_env = CRFlattenNormWrapper(rec_env)
        rec_env = gym.wrappers.RecordVideo(
            rec_env,
            video_folder=self.video_dir,
            name_prefix=f"step{step:07d}",
            episode_trigger=lambda _: True,
            disable_logger=True,
        )
        
        state, _ = rec_env.reset()
        ep_return = np.zeros(2)

        if os.environ.get("PROFILE_MODE"):
            env_step_times = []

        try:
            while True:
                state_1, state_2  = self.split_observations(state)
                
                with t.no_grad():
                    action_1, _, _, _ = net_1.get_action_and_value(state_1)
                    action_2, _, _, _ = net_2.get_action_and_value(state_2)
            
                action = self.join_actions(action_1, action_2)


                if os.environ.get("PROFILE_MODE"):
                    env_step_start_time = time.time()
                
                state, reward, terminated, truncated, _ = rec_env.step(action)
                
                if os.environ.get("PROFILE_MODE"):
                    env_step_end_time = time.time()
                    env_step_times.append(env_step_end_time - env_step_start_time)
            
                ep_return += np.array(reward)

                if terminated or truncated:
                    break

        except Exception as e:
            print(e)
        
        rec_env.close()

        if os.environ.get("PROFILE_MODE"):
            print(f"\t [video] Average env step time: {np.mean(env_step_times):.3f} seconds")
        
        print(f"\t [video] step {step}:\t return: {ep_return}")
        
        if self.wandb_logging:
            video_path = os.path.join(self.video_dir, f"step{step:07d}-episode-0.mp4")
            if os.path.exists(video_path):
                wandb.log({"video": wandb.Video(video_path, format="mp4")}, step=step)

        return ep_return


    def get_network_and_optimiser(self, weights=None):
        """
        Self-Play PPO samples from a pool of policies, so the weights arg 
        inits the network with those
        """

        network = ActorCritic(**self.cfg.network.to_dict())
        
        if weights:
            network.load_state_dict(t.load(weights, weights_only=True))

        optimiser = optim.Adam(network.parameters(), lr=self.cfg.lr)

        return network, optimiser


    def split_observations(self, obs):
        player_1_num_cards = len(obs["player_1_cards"])
        player_2_num_cards = len(obs["player_2_cards"])

        if player_1_num_cards == 0:
            player_1_cards_arr = np.zeros((0, self.env.flat_card_space.shape[0]), dtype=np.float32)
        else:
            player_1_cards_arr = np.array(obs["player_1_cards"], dtype=np.float32)
            
        if player_2_num_cards == 0:
            player_2_cards_arr = np.zeros((0, self.env.flat_card_space.shape[0]), dtype=np.float32)
        else:
            player_2_cards_arr = np.array(obs["player_2_cards"], dtype=np.float32)

        player_1_cards = F.pad(
            t.tensor(player_1_cards_arr),  # (X, card_dim)
            (0, 0, 0, self.max_num_objects - player_1_num_cards),         # (N, card_dim)
            "constant", 0
        ).unsqueeze(0)

        player_2_cards = F.pad(
            t.tensor(player_2_cards_arr),
            (0, 0, 0, self.max_num_objects - player_2_num_cards),
            "constant", 0
        ).unsqueeze(0)

        player_1_crown_towers = t.tensor(np.array(obs["player_1_crown_towers"], dtype=np.float32)).unsqueeze(0)
        player_2_crown_towers = t.tensor(np.array(obs["player_2_crown_towers"], dtype=np.float32)).unsqueeze(0)

        obs_1 = {
            "game_completion_fraction": t.tensor(np.array(obs["game_completion_fraction"], dtype=np.float32)).reshape(-1, 1),
            "elixirs":                  t.tensor(np.array(obs["player_1_elixirs"], dtype=np.float32)).reshape(-1, 1),
            "my_cards":                 player_1_cards,
            "opponent_cards":           player_2_cards,
            "my_crown_towers":          player_1_crown_towers,
            "opponent_crown_towers":    player_2_crown_towers,
        }

        # Flipping them is just a 180 degree rotation about the origin
        # (x, y) -> (width_cell - x, height_cell - y)
        width_cell  = self.arena.tile_size * self.arena.width
        height_cell = self.arena.tile_size * self.arena.height
        position_x_idx, position_y_idx = np.arange(*self.env.flattened_card_space_indices["position"])
        
        # Doing this to prevent zero padded entries to falsely invert
        player_1_cards = t.tensor(player_1_cards_arr)
        player_2_cards = t.tensor(player_2_cards_arr)

        player_1_cards[..., position_x_idx] = width_cell  - player_1_cards[..., position_x_idx]
        player_1_cards[..., position_y_idx] = height_cell - player_1_cards[..., position_y_idx]

        player_2_cards[..., position_x_idx] = width_cell  - player_2_cards[..., position_x_idx]
        player_2_cards[..., position_y_idx] = height_cell - player_2_cards[..., position_y_idx]

        player_1_cards = F.pad(
            player_1_cards,  # (X, card_dim)
            (0, 0, 0, self.max_num_objects - player_1_num_cards),  # (N, card_dim)
            "constant", 0
        ).unsqueeze(0)

        player_2_cards = F.pad(
            player_2_cards,
            (0, 0, 0, self.max_num_objects - player_2_num_cards),
            "constant", 0
        ) .unsqueeze(0)

        player_1_crown_towers[..., position_x_idx] = width_cell  - player_1_crown_towers[..., position_x_idx]
        player_1_crown_towers[..., position_y_idx] = height_cell - player_1_crown_towers[..., position_y_idx]

        player_2_crown_towers[..., position_x_idx] = width_cell  - player_2_crown_towers[..., position_x_idx]
        player_2_crown_towers[..., position_y_idx] = height_cell - player_2_crown_towers[..., position_y_idx]

        obs_2 = {
            "game_completion_fraction": t.tensor(np.array(obs["game_completion_fraction"], dtype=np.float32)).reshape(-1, 1),
            "elixirs":                  t.tensor(np.array(obs["player_2_elixirs"], dtype=np.float32)).reshape(-1, 1),
            "my_cards":                 player_2_cards,
            "opponent_cards":           player_1_cards,
            "my_crown_towers":          player_2_crown_towers,
            "opponent_crown_towers":    player_1_crown_towers,
        }

        return obs_1, obs_2


    def join_actions(self, action_1, action_2):
        return {
            "player_1_skip":     action_1["skip"],
            "player_1_card_idx": action_1["deck_idx"],
            "player_1_card_position": (
                action_1["position"] % self.arena.width,
                action_1["position"] // self.arena.width
            ),

            "player_2_skip":     action_2["skip"],
            "player_2_card_idx": action_2["deck_idx"],
            "player_2_card_position": (
                self.arena.width  - action_2["position"] % self.arena.width,
                self.arena.height - action_2["position"] // self.arena.width
            ),
        }

    
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