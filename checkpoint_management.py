import os
from collections import deque

import numpy as np
import torch as t


class ConstantWindow_CheckpointManagement:
    def __init__(
        self, 
        checkpoint_dir, 
        window_size, 
        save_every_k_global_steps,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.window_size = window_size
        self.save_every_k_global_steps = save_every_k_global_steps

        self.checkpoint_counter = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)


    def load(self, net):
        checkpoint_min = max(0, self.checkpoint_counter - self.window_size)
        checkpoint_max = self.checkpoint_counter - 1

        if checkpoint_max <= checkpoint_min:
            return 

        checkpoint_sample = np.random.randint(checkpoint_min, checkpoint_max)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint_sample}.pt")
        net.load_state_dict(t.load(checkpoint_path, weights_only=True))
    

    def store(self, net, global_step):
        if global_step % self.save_every_k_global_steps != 0:
            return 

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.checkpoint_counter}.pt")
        t.save(net.state_dict(), checkpoint_path)

        self.checkpoint_counter += 1


class DeltaWindow_CheckpointManagement:
    def __init__(
        self, 
        checkpoint_dir, 
        delta, 
        save_every_k_global_steps, 
    ):
        self.checkpoint_dir = checkpoint_dir
        self.delta = delta
        self.save_every_k_global_steps = save_every_k_global_steps

        self.checkpoint_counter = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)


    def load(self, net):
        checkpoint_min = self.checkpoint_counter * (1 - self.delta)
        checkpoint_max = self.checkpoint_counter - 1

        if checkpoint_max <= checkpoint_min:
            return 

        checkpoint_sample = np.random.randint(checkpoint_min, checkpoint_max)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint_sample}.pt")
        net.load_state_dict(t.load(checkpoint_path, weights_only=True))


    def store(self, net, global_step):
        if global_step % self.save_every_k_global_steps != 0:
            return 

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.checkpoint_counter}.pt")
        t.save(net.state_dict(), checkpoint_path)

        self.checkpoint_counter += 1


class AdvancedTemporal_CheckpointManagement:
    def __init__(
        self, 
        checkpoint_dir, 
        
        # Loading
        loading_latest_ratio=0.5,
        loading_delta_window=0.2,

        # Storage
        min_games_before_checkpointing=100,
        score_queue_size=100,
        avg_score_threshold=0.55,
    ):
        self.checkpoint_dir = checkpoint_dir

        # Loading
        self.loading_latest_ratio = loading_latest_ratio
        self.loading_delta_window = loading_delta_window

        # Storage
        self.min_games_before_checkpointing = min_games_before_checkpointing
        self.avg_score_threshold = avg_score_threshold

        self.checkpoint_counter = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.score_queue = deque(maxlen=score_queue_size)


    def load(self, net):
        if np.random.rand() < self.loading_latest_ratio:
            checkpoint_sample = self.checkpoint_counter - 1
        else:
            checkpoint_min = self.checkpoint_counter * (1 - self.loading_delta_window)
            checkpoint_max = self.checkpoint_counter - 1

            if checkpoint_max <= checkpoint_min:
                return 

            checkpoint_sample = np.random.randint(checkpoint_min, checkpoint_max)

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint_sample}.pt")
        net.load_state_dict(t.load(checkpoint_path, weights_only=True))


    def update(self, net, returns):
        score = 0
        if returns[0] > returns[1]:
            score = 1
        elif returns[0] == returns[1]:
            score = 0.5
        self.score_queue.append(score)

        if len(self.score_queue) < self.min_games_before_checkpointing:
            return

        if np.mean(self.score_queue) < self.avg_score_threshold:
            return

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.checkpoint_counter}.pt")
        t.save(net.state_dict(), checkpoint_path)

        self.checkpoint_counter += 1
        self.score_queue.clear()


class AdvancedEloBased_CheckpointManagement:
    def __init__(
        self, 
        checkpoint_dir, 
        elo_cfg,
        
        # Loading
        loading_latest_ratio=0.5,

        # Storage
        min_games_before_checkpointing=100,
        score_queue_size=100,
        avg_score_threshold=0.55,

    ):
        self.checkpoint_dir = checkpoint_dir
        self.elo_cfg = elo_cfg

        # Loading
        self.loading_latest_ratio = loading_latest_ratio

        # Storage
        self.min_games_before_checkpointing = min_games_before_checkpointing
        self.avg_score_threshold = avg_score_threshold

        self.checkpoint_counter = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.score_queue = deque(maxlen=score_queue_size)


        self.stored_by_idx = {}  # checkpoint_idx -> (elo, path)
        self.stored_by_elo = {}  # elo -> [checkpoint_indices]


    def load(self, net, current_elo):
        current_elo = int(current_elo)
        
        if self.checkpoint_counter == 0:
            return self.elo_cfg.initial_elo

        if np.random.rand() < self.loading_latest_ratio:
            checkpoint_sample = self.checkpoint_counter - 1

            opponent_elo, checkpoint_path = self.stored_by_idx.get(checkpoint_sample, None)

            net.load_state_dict(t.load(checkpoint_path, weights_only=True))
            return opponent_elo

        E_As = []
        for opponent_elo in self.stored_by_elo.keys():
            E_As.append(
                1 / (1 + 10 ** ((opponent_elo - current_elo) / self.elo_cfg.scale))
            )

        probs = 0.5 - abs(0.5 - np.array(E_As))  # peak at 0.5, falls off to 0 on either extremes
        probs = probs / np.sum(probs)
        checkpoint_sample = np.random.choice(len(E_As), p=probs)

        checkpoint_indices = self.stored_by_elo.get(checkpoint_sample, None)
        assert checkpoint_indices is not None

        checkpoint_sample = np.random.choice(len(checkpoint_indices))
        opponent_elo, checkpoint_path = self.stored_by_idx.get(checkpoint_sample, None)
        assert checkpoint_path is not None

        net.load_state_dict(t.load(checkpoint_path, weights_only=True))
        return opponent_elo


    def update(self, net, score, current_elo):
        current_elo = int(current_elo)

        self.score_queue.append(score)

        if len(self.score_queue) < self.min_games_before_checkpointing:
            return

        if np.mean(self.score_queue) < self.avg_score_threshold:
            return

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.checkpoint_counter}_{current_elo}.pt")
        t.save(net.state_dict(), checkpoint_path)

        self.stored_by_idx[self.checkpoint_counter] = (current_elo, checkpoint_path)
        if current_elo not in self.stored_by_elo:
            self.stored_by_elo[current_elo] = []
        self.stored_by_elo[current_elo].append(self.checkpoint_counter)

        self.checkpoint_counter += 1
        self.score_queue.clear()
