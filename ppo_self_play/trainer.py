from addict import Dict
import wandb

import gymnasium as gym


class Trainer:
    def __init__(self, gym_env_name):
        self.gym_env_name = gym_env_name
        self.env = gym.make(self.gym_env_name)
        self.env = FlattenWithSequence(self.env)

        self.cfg = Dict()
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

        
        self.video_dir = "./videos"
        self.video_every = 250_000
        os.makedirs(self.video_dir, exist_ok=True)


        self.wandb_logging = False
        # self.wandb_logging = True

        if self.wandb_logging:
            wandb.init(
                project="clash_royale-ppo_self_play",
                config=self.cfg.to_dict()
            )


if __name__ == "__main__":
    trainer = Trainer(gym_env_name="ClashRoyaleEnv-v0")