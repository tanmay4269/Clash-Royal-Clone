import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")  # TODO: fix code later


import os
import argparse
from copy import deepcopy
import time; os.environ["TZ"] = "Asia/Kolkata"

import random
import numpy as np

from addict import Dict
from rich import print

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import ActorCritic, BotNet

import gymnasium as gym
import cr_gym_env
from cr_flatten_norm_wrapper import CRFlattenNormWrapper

from rollout_buffer import RolloutBuffer
from checkpoint_management import *
from utils import AverageMeter, HeatmapVisualizerWrapper
from diagnostics_logger import DiagnosticsLogger

import wandb



class Trainer:
    def __init__(
        self, 
        gym_env_name, 
        run_name=None, 
        use_lr_tuner=True, 
        overfit_mode=None, 
        wandb_logging=True, 
        debug=False
):
        t.set_default_dtype(t.float32)

        if t.cuda.is_available():
            device = t.device("cuda")
        else:
            device = t.device("cpu")

        print(f"Using device: {device}")

        t.set_default_device(device)

        self.debug = debug or bool(os.environ.get("DEBUG_MODE"))
        if self.debug:
            os.environ["DEBUG_MODE"] = "1"

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
        invalid_position_mask[self.arena.height//2 :, :] = True  # bottom/opponent half is always invalid in player-1 view
        
        ### CONFIGS ###
        self.cfg = Dict()
        
        datetime_str = time.strftime('%m%d-%H%M%S')
        if run_name:
            self.cfg.run_name = f"{run_name}_{datetime_str}"
        else:
            self.cfg.run_name = datetime_str

        if self.debug:
            self.cfg.run_name = f"DEBUG_{self.cfg.run_name}"

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
        if self.debug:
            self.cfg.buffer.n_steps = 2048

        # Elo Rating
        self.cfg.elo.initial_rating = 1200
        self.cfg.elo.scale = 400
        self.cfg.elo.k_factor = 32

        self.current_elo = self.cfg.elo.initial_rating

        # Player Pool
        self.cfg.checkpoint_manager_type = 'elo_based'

        if self.cfg.checkpoint_manager_type == 'advanced_temporal':
            self.checkpoint_manager = AdvancedTemporal_CheckpointManagement(
                checkpoint_dir="./checkpoints",
                loading_latest_ratio=0.5,
                loading_delta_window=0.2,
                min_games_before_checkpointing=100,
                score_queue_size=100,
                avg_score_threshold=0.55,
            )
        elif self.cfg.checkpoint_manager_type == 'elo_based':
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
        self.cfg.critic_loss_coef = 0.5
        self.entropy_loss_coef = 0.01  # TODO: later try linear decay

        # LR Finder
        self.cfg.lr_tuner.enabled = use_lr_tuner
        if self.debug:
            self.cfg.lr_tuner.enabled = False

        self.cfg.lr_tuner.min_lr = 1e-7
        self.cfg.lr_tuner.max_lr = 1e-1
        self.cfg.lr_tuner.num_steps = 200
        self.cfg.lr_tuner.pick_lr_factor = 0.3

        self.cfg.lr = 3e-4  # fallback

        self.lr_tuned = False

        # Misc
        self.cfg.minibatch_size = 2048
        if os.environ.get("DEBUG_MODE"):
            self.cfg.minibatch_size = 128

        self.cfg.k_epochs = 6  # gradient steps per rollout
        self.cfg.max_steps = 10_000_000  # total env steps

        # None, 'single-buffer', 'fixed-opponent', 'vs-random', 'vs-skip', or 'vs-scripted'
        self.overfit_mode = overfit_mode

        # Replay storing
        self.video_dir = f"./videos/{self.cfg.run_name}/"
        if self.debug:
            self.video_dir = "./videos/DEBUG/"
        os.makedirs(self.video_dir, exist_ok=True)

        self.video_every_k_global_steps = 20_000
        if self.debug:
            self.video_every_k_global_steps = 5_000

        # WANDB logging
        self.wandb_logging = wandb_logging
        if self.debug:
            self.wandb_logging = False

        if self.wandb_logging:
            wandb.init(
                project="clash_royale-ppo_self_play",
                name=self.cfg.run_name,
                config=self.cfg.to_dict()
            )


    def train(self):
        self.set_seed(self.cfg.seed)

        ep_return = np.zeros(2)
        ep_returns = []

        global_step = 0
        next_video = 0
        # next_video = self.video_every_k_global_steps

        net_1, optimiser_1 = self.get_network_and_optimiser()
        initial_net = deepcopy(net_1)
        opponent_elo = self.cfg.elo.initial_rating
        
        if self.overfit_mode in ['vs-random', 'vs-skip', 'vs-scripted']:
            bot_type = self.overfit_mode.split('-')[1]
            net_2 = BotNet(
                bot_type,
                self.cfg.network.invalid_position_mask, 
                self.cfg.network.num_cards_in_deck, 
                self.cfg.network.position_space_width,
                self.cfg.network.position_space_height
            )
        else:
            net_2 = deepcopy(net_1)

        state, _ = self.env.reset(seed=self.cfg.seed)
        state_1, state_2  = self.split_observations(state)
        last_done = False
        
        self.logger = DiagnosticsLogger(self.cfg, self.wandb_logging)

        while global_step < self.cfg.max_steps:
            buffer = RolloutBuffer(**self.cfg.buffer.to_dict())

            for _ in range(self.cfg.buffer.n_steps):
                if global_step >= next_video:
                    self.record_episode(global_step, net_1, net_2)
                    next_video += self.video_every_k_global_steps

                with t.no_grad():
                    action_1, log_prob_1, entropy_1, value_1 = net_1.get_action_and_value(state_1)
                    action_2, _, _, _ = net_2.get_action_and_value(state_2) 

                action = self.join_actions(action_1, action_2)

                # Collect metrics for Player 1
                self.logger.on_step(action_1, self.env)

                try:
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                except Exception as e:
                    print(f"Env step failed: {e}")

                    next_state = state  # stay in place safely
                    reward = [0.0, 0.0] # neutral fallback
                    terminated = True
                    truncated = False

                done = terminated or truncated
                last_done = terminated

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

                    self.logger.on_episode_end(
                        self.env, terminated, truncated, 
                        self.current_elo, ep_return[0], score, global_step
                    )

                    ep_returns.append(ep_return)
                    ep_return = np.zeros(2)

                    ep_seed = self.cfg.seed
                    if self.overfit_mode not in ['fixed-opponent', 'vs-random', 'vs-skip', 'vs-scripted']:
                        opponent_elo = self.checkpoint_manager.load(net_2, self.current_elo)
                        self.checkpoint_manager.update(net_1, score, self.current_elo)
                        ep_seed = np.random.randint(0, 2**31)

                    state, _ = self.env.reset(seed=ep_seed)
                    state_1, state_2  = self.split_observations(state)
                    last_done = False  # fresh episode start — bootstrap is valid

                global_step += 1

            # Bootstrap final value and compute GAE
            with t.no_grad():
                _, _, _, last_value_1 = net_1.get_action_and_value(state_1)
                buffer.compute_gae(last_value_1, last_done)

            pre_update_net = deepcopy(net_1)

            # PPO update
            actor_loss, critic_loss, entropy, ratio_mean, advantage_mean, explained_variance = \
                self.ppo_update(buffer, net_1, optimiser_1, global_step)
            
            self.logger.on_ppo_update(global_step, buffer, net_1, initial_net, pre_update_net)

            buffer.reset()

            print(
                f"step {global_step:7d} | avg(last 100 eps): {(np.mean(ep_returns[-100:], axis=0)[0]):10.3f} | "
                f"actor_loss: {actor_loss:7.3f} | critic_loss: {critic_loss:7.3f} | entropy: {entropy:.3f}"
            )

            if self.wandb_logging:
                wandb.log({
                    "actor_loss":         actor_loss,
                    "critic_loss":        critic_loss,
                    "entropy":            entropy,
                    "ratio_mean":         ratio_mean,
                    "advantage_mean":     advantage_mean,
                    "explained_variance": explained_variance
                }, step=global_step)

    
    def ppo_update(self, buffer, net, optimiser, global_step=0):
        if self.cfg.lr_tuner.enabled and not self.lr_tuned:
            self.lr_finder(buffer, net, optimiser)
            self.lr_tuned = True

        # Linear Learning Rate Decay
        frac = max(0.0, 1.0 - (global_step / self.cfg.max_steps))
        current_lr = self.cfg.lr * frac
        for param_group in optimiser.param_groups:
            param_group["lr"] = current_lr

        num_epochs = self.cfg.k_epochs
        if self.overfit_mode == 'single-buffer':
            num_epochs = 1_000_000
            
        actor_loss_meter = AverageMeter()
        critic_loss_meter = AverageMeter()
        entropy_meter = AverageMeter()
        ratio_mean_meter = AverageMeter()
        advantage_mean_meter = AverageMeter()
        explained_variance_meter = AverageMeter()

        if os.environ.get("PROFILE_MODE"):
            ppo_update_start_time = time.time()

        for epoch in range(num_epochs):
            for batch in buffer.get_minibatches(self.cfg.minibatch_size):
                actor_loss, critic_loss, entropy, ratio_mean, advantage_mean, explained_variance = \
                    self.on_batch_update(
                        net,
                        optimiser,
                        batch,
                        optimize=True,
                    )

                actor_loss_meter.add_sample(actor_loss)
                critic_loss_meter.add_sample(critic_loss)
                entropy_meter.add_sample(entropy)
                ratio_mean_meter.add_sample(ratio_mean)
                advantage_mean_meter.add_sample(advantage_mean)
                explained_variance_meter.add_sample(explained_variance)

            if epoch % 10 == 0 and self.overfit_mode == 'single-buffer':
                window = 10 * len(buffer) // self.cfg.minibatch_size
                _log = {
                    "actor_loss":         actor_loss_meter.window_average(window),
                    "critic_loss":        critic_loss_meter.window_average(window),
                    "entropy":            entropy_meter.window_average(window),
                    "ratio_mean":         ratio_mean_meter.window_average(window),
                    "advantage_mean":     advantage_mean_meter.window_average(window),
                    "explained_variance": explained_variance_meter.window_average(window)
                }

                if self.wandb_logging:
                    wandb.log(_log, step=epoch)
                print(f"Epoch {epoch:7d} | " + " | ".join([f"{k}: {v:.3f}" for k, v in _log.items()]))

        if os.environ.get("PROFILE_MODE"):
            ppo_update_end_time = time.time()
            print(f"PPO update time: {ppo_update_end_time - ppo_update_start_time:.3f} seconds")

        return (
            actor_loss_meter.average(),
            critic_loss_meter.average(),
            entropy_meter.average(),
            ratio_mean_meter.average(),
            advantage_mean_meter.average(),
            explained_variance_meter.average()
        )


    def on_batch_update(self, net, optimiser, batch, optimize=False, scheduler=None):
        states, actions, old_log_probs, advantages, returns = batch

        _, new_log_probs, entropies, values = net.get_action_and_value(states, actions)
        ratio = (new_log_probs - old_log_probs).exp()

        actor_loss = -t.min(
            advantages * ratio,
            advantages * t.clip(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip)
        ).mean()

        critic_loss = ((returns - values) ** 2).mean()
        loss = actor_loss + self.cfg.critic_loss_coef * critic_loss - self.entropy_loss_coef * entropies.mean()

        if optimize:
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
            optimiser.step()
            if scheduler is not None:
                scheduler.step()

        explained_variance = 1 - (returns - values).var() / returns.var()

        return (
            actor_loss.item(),
            critic_loss.item(),
            entropies.mean().item(),
            ratio.mean().item(),
            advantages.mean().item(),
            explained_variance.item(),
        )


    def lr_finder(self, buffer, net, optimiser):
        print("[lr_finder] Starting learning rate tuning...")

        initial_state = deepcopy(net.state_dict())
        min_lr = self.cfg.lr_tuner.min_lr
        max_lr = self.cfg.lr_tuner.max_lr
        num_steps = self.cfg.lr_tuner.num_steps

        temp_net = deepcopy(net)
        temp_optimiser = optim.Adam(temp_net.parameters(), lr=min_lr)

        def lr_multiplier(step):
            if num_steps <= 1:
                return 1.0
            return (max_lr / min_lr) ** (step / (num_steps - 1))

        lr_scheduler = optim.lr_scheduler.LambdaLR(temp_optimiser, lr_lambda=lr_multiplier)

        minibatches = list(buffer.get_minibatches(self.cfg.minibatch_size))
        if not minibatches:
            raise RuntimeError("lr_finder received no minibatches")

        lr_history = []
        loss_history = []
        
        beta = 0.98  # smoothing factor
        avg_loss = 0.0
        best_loss = 0.0

        for step_idx in range(num_steps):
            batch = minibatches[step_idx % len(minibatches)]
            current_lr = temp_optimiser.param_groups[0]["lr"]

            actor_loss, critic_loss, entropy, _, _, _ = self.on_batch_update(
                temp_net,
                temp_optimiser,
                batch,
                optimize=True,
                scheduler=lr_scheduler,
            )

            # raw_loss = actor_loss + self.cfg.critic_loss_coef * critic_loss - self.entropy_loss_coef * entropy

            # In PPO, Actor Objective + Entropy behaves terribly in LR Searches
            # because large steps break the trust region ratio, causing loss to explode instantly.
            # Using just the Critic loss (MSE) behaves like standard supervised regression.
            raw_loss = critic_loss
            
            # Exponentially smooth the loss to handle batch variance
            avg_loss = beta * avg_loss + (1 - beta) * raw_loss
            smoothed_loss = avg_loss / (1 - beta**(step_idx + 1)) # bias correction

            # Stop if loss explodes (diverges)
            if step_idx > 0 and smoothed_loss > 4 * best_loss:
                break
                
            if step_idx == 0 or smoothed_loss < best_loss:
                best_loss = smoothed_loss

            lr_history.append(current_lr)
            loss_history.append(smoothed_loss)

        temp_net.load_state_dict(initial_state)

        if not loss_history:
            raise RuntimeError("[lr_finder] failed to produce loss values")

        min_idx = int(np.argmin(loss_history))
        min_loss_lr = lr_history[min_idx]
        suggested_lr = float(np.clip(min_loss_lr * self.cfg.lr_tuner.pick_lr_factor, min_lr, max_lr))

        for param_group in optimiser.param_groups:
            param_group["lr"] = suggested_lr

        self.cfg.lr = suggested_lr

        print(f"[lr_finder] min_loss_lr: {min_loss_lr:.3e} | suggested_lr: {suggested_lr:.3e}")


    def record_episode(self, step, net_1, net_2):
        """Run one greedy episode and save video to self.video_dir."""

        rec_env = gym.make(self.gym_env_name, render_mode="rgb_array")
        rec_env = CRFlattenNormWrapper(rec_env)
        rec_env = HeatmapVisualizerWrapper(rec_env)
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
                    _, skip_logits_1, deck_logits_1, pos_logits_1 = net_1(state_1)
                    _, skip_logits_2, deck_logits_2, pos_logits_2 = net_2(state_2)

                    if net_1.invalid_position_mask is not None:
                        pos_logits_1 = pos_logits_1.masked_fill(net_1.invalid_position_mask, float('-inf'))
                    if net_2.invalid_position_mask is not None:
                        pos_logits_2 = pos_logits_2.masked_fill(net_2.invalid_position_mask, float('-inf'))

                    skip_probs_1 = t.sigmoid(skip_logits_1).squeeze(0).cpu().numpy()
                    deck_probs_1 = F.softmax(deck_logits_1, dim=-1).squeeze(0).cpu().numpy()
                    pos_logits_np_1 = pos_logits_1.squeeze(0).cpu().numpy()
                    pos_probs_1  = F.softmax(pos_logits_1, dim=-1).squeeze(0).cpu().numpy()

                    skip_dist_1 = t.distributions.Bernoulli(logits=skip_logits_1)
                    deck_dist_1 = t.distributions.Categorical(logits=deck_logits_1)
                    pos_dist_1 = t.distributions.Categorical(logits=pos_logits_1)

                    skip_dist_2 = t.distributions.Bernoulli(logits=skip_logits_2)
                    deck_dist_2 = t.distributions.Categorical(logits=deck_logits_2)
                    pos_dist_2 = t.distributions.Categorical(logits=pos_logits_2)

                    action_1 = {
                        "skip": skip_dist_1.sample().detach(),
                        "deck_idx": deck_dist_1.sample().detach(),
                        "position": pos_dist_1.sample().detach(),
                    }
                    action_2 = {
                        "skip": skip_dist_2.sample().detach(),
                        "deck_idx": deck_dist_2.sample().detach(),
                        "position": pos_dist_2.sample().detach(),
                    }
            
                action = self.join_actions(action_1, action_2)
                rec_env.env.update(
                    player_idx=1,
                    skip_prob=skip_probs_1.item(),
                    deck_probs=deck_probs_1,
                    pos_probs=pos_probs_1,
                    pos_logits=pos_logits_np_1,
                    action=action_1,
                    env_action=action,
                )


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
            print(f"[video] Average env step time: {np.mean(env_step_times):.3f} seconds")
        
        print(f"[video] step {step}:\t return: {ep_return}")
        
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

        position_x_idx, position_y_idx = np.arange(*self.env.flattened_card_space_indices["position"])

        def pad_cards(cards_arr, num_cards):
            return F.pad(
                t.tensor(cards_arr),  # (X, card_dim)
                (0, 0, 0, self.max_num_objects - num_cards),  # (N, card_dim)
                "constant",
                0,
            ).unsqueeze(0)

        def rotate_entities_180(entities):
            rotated = entities.clone()
            rotated[..., position_x_idx] *= -1
            rotated[..., position_y_idx] *= -1
            return rotated

        player_1_cards = pad_cards(player_1_cards_arr, player_1_num_cards)
        player_2_cards = pad_cards(player_2_cards_arr, player_2_num_cards)

        player_1_crown_towers = t.tensor(np.array(obs["player_1_crown_towers"], dtype=np.float32)).unsqueeze(0)
        player_2_crown_towers = t.tensor(np.array(obs["player_2_crown_towers"], dtype=np.float32)).unsqueeze(0)

        game_completion_fraction = t.tensor(np.array(obs["game_completion_fraction"], dtype=np.float32)).reshape(-1, 1)

        # Network convention: each policy sees itself in player-1 view:
        # self at the top, opponent at the bottom, with the usual top-left x/y origin.
        obs_1 = {
            "game_completion_fraction": game_completion_fraction,
            "elixirs":                  t.tensor(np.array(obs["player_1_elixirs"], dtype=np.float32)).reshape(-1, 1),
            "my_cards":                 player_1_cards,
            "opponent_cards":           player_2_cards,
            "my_crown_towers":          player_1_crown_towers,
            "opponent_crown_towers":    player_2_crown_towers,
        }

        obs_2 = {
            "game_completion_fraction": game_completion_fraction,
            "elixirs":                  t.tensor(np.array(obs["player_2_elixirs"], dtype=np.float32)).reshape(-1, 1),
            "my_cards":                 rotate_entities_180(player_2_cards),
            "opponent_cards":           rotate_entities_180(player_1_cards),
            "my_crown_towers":          rotate_entities_180(player_2_crown_towers),
            "opponent_crown_towers":    rotate_entities_180(player_1_crown_towers),
        }

        return obs_1, obs_2


    def join_actions(self, action_1, action_2):
        def scalar_int(value):
            if hasattr(value, "detach"):
                return int(value.detach().cpu().item())
            return int(value)

        def policy_position_to_xy(action):
            pos_idx = scalar_int(action["position"])
            return pos_idx % self.arena.width, pos_idx // self.arena.width

        def rotate_xy_180(x, y):
            return self.arena.width - 1 - x, self.arena.height - 1 - y

        return {
            "player_1_skip":          scalar_int(action_1["skip"]),
            "player_1_card_idx":      scalar_int(action_1["deck_idx"]),
            "player_1_card_position": policy_position_to_xy(action_1),

            "player_2_skip":          scalar_int(action_2["skip"]),
            "player_2_card_idx":      scalar_int(action_2["deck_idx"]),
            "player_2_card_position": rotate_xy_180(*policy_position_to_xy(action_2)),
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
    parser = argparse.ArgumentParser(description="Train PPO for Clash Royale")
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=None, 
        help="Prefix for the run name; date-time will be appended."
    )
    parser.add_argument(
        "--use_lr_tuner", 
        action=argparse.BooleanOptionalAction, 
        default=True, 
        help="Enable or disable LR tuner."
    )
    parser.add_argument(
        "--overfit_mode", 
        type=str, 
        default=None, 
        choices=['single-buffer', 'fixed-opponent', 'vs-random', 'vs-skip', 'vs-scripted'], 
        help="Overfit mode to use."
    )
    parser.add_argument(
        "--wandb_logging", 
        action=argparse.BooleanOptionalAction, 
        default=True, 
        help="Enable or disable wandb logging."
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode."
    )
    args = parser.parse_args()

    trainer = Trainer(
        gym_env_name="ClashRoyaleEnv-v0",
        run_name=args.run_name,
        use_lr_tuner=args.use_lr_tuner,
        overfit_mode=args.overfit_mode,
        wandb_logging=args.wandb_logging,
        debug=args.debug
    )

    trainer.train()
