import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import wandb

###############
# Environment #
###############

ENV_NAME = "ClashRoyaleEnv-v0"
env = gym.make(ENV_NAME)


#######################
# PPO Hyperparameters #
#######################

GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2   # PPO clip range
K_EPOCHS        = 10    # gradient steps per rollout
ROLLOUT_STEPS   = 2048  # timesteps to collect before each update
MINIBATCH_SIZE  = 128
LR              = 3e-4

MAX_ENTROPY_COEF = 0.005
MIN_ENTROPY_COEF = 0.005
ENTROPY_COEF     = None

VF_COEF   = 0.5          # critic loss weight in combined loss
GRAD_CLIP = 0.5
MAX_STEPS = 100_000_000  # total environment steps

VIDEO_DIR   = "./videos"
VIDEO_EVERY = 250_000   # record a snapshot every N env steps

WANDB_LOGGING = False
# WANDB_LOGGING = True

os.makedirs(VIDEO_DIR, exist_ok=True)

if WANDB_LOGGING:
    wandb.init(
        project="clash_royale-ppo_self_play",
        config=dict(
            gamma=GAMMA, 
            gae_lambda=GAE_LAMBDA, 
            clip_eps=CLIP_EPS,
            k_epochs=K_EPOCHS, 
            rollout_steps=ROLLOUT_STEPS,
            lr=LR, 
            entropy_coef=ENTROPY_COEF, 
            vf_coef=VF_COEF,
        ),
    )


#######################################
# Actor-Critic Network (shared trunk) #
#######################################
"""
TODOs:
**Input:** All entities/troops/projectiles (variable count), elixir (scalar), game clock (scalar), available cards in hand.

**Encoder:** Each entity/troop/projectile → shared embedding network → sum all embeddings. Concat with elixir + game clock → neck network → game state vector.

**Head 1 — Deploy or not:** Binary head off game state vector.

**Head 2 — Which card:** Each card embedding paired with game state vector → shared scorer network → score per card → mask cards you can't afford → softmax → card probabilities.

**Head 3 — Where to deploy:** Game state vector → position head → score per grid cell → mask invalid cells → softmax → position probabilities.

All three heads trained jointly. Invalid actions masked, never sampled.
"""

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.actor_mu = nn.Sequential(
            nn.Linear(128, act_dim)
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """x: (B, obs_dim) -> mu: (B, act_dim), std: (B, act_dim), value: (B,)"""
        x = torch.as_tensor(x, dtype=torch.float32)
        # TODO: Normalisation of x

        x = self.trunk(x)
        value = self.critic(x).squeeze(-1)

        mu = self.actor_mu(x)
        std = self.log_std.clamp(-20, 2).exp().expand_as(mu)

        return mu, std, value

    def get_action_and_value(self, x, action=None):
        """x: (B, obs_dim), action: (B, act_dim) or None
        -> action: (B, act_dim), log_prob: (B,), entropy: (B,), value: (B,)"""
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        mu, std, value = self(x)
        dist = torch.distributions.normal.Normal(mu, std)

        if action is None:
            u = dist.rsample()
            action = torch.tanh(u)
        else:
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)
            u = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))

        log_prob = (dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action.detach().numpy(), log_prob, entropy, value


ac = ActorCritic()
optimiser = optim.Adam(ac.parameters(), lr=LR)


##################
# Rollout Buffer #
##################

class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, act_dim):
        self.n_steps = n_steps
        self.states      = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((n_steps, act_dim), dtype=np.float32)
        self.log_probs   = np.zeros(n_steps,            dtype=np.float32)
        self.rewards     = np.zeros(n_steps,            dtype=np.float32)
        self.values      = np.zeros(n_steps,            dtype=np.float32)
        self.dones       = np.zeros(n_steps,            dtype=np.float32)
        self.advantages  = np.zeros(n_steps,            dtype=np.float32)
        self.returns     = np.zeros(n_steps,            dtype=np.float32)
        self.ptr = 0

    def push(self, state, action, log_prob, reward, value, done):
        """state, action, log_prob, reward, value, done -> None (advances ptr)"""

        self.states[self.ptr]    = state
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.dones[self.ptr]     = done

        self.ptr += 1

    def compute_gae(self, last_value, last_done):
        """last_value: float, last_done: float -> None (fills self.advantages, self.returns)"""
        advantage = 0.0

        for i in range(self.n_steps - 1, -1, -1):
            if i == self.n_steps - 1:
                next_value = last_value
                next_done = last_done
            else:
                next_value = self.values[i + 1]
                next_done = self.dones[i + 1]

            td_error = self.rewards[i] + GAMMA * next_value * (1 - next_done) - self.values[i]
            advantage = td_error + GAMMA * GAE_LAMBDA * (1 - next_done) * advantage

            self.advantages[i] = advantage

        # returns BEFORE normalizing advantages
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_minibatches(self, batch_size):
        """batch_size: int -> yields (states, actions, old_log_probs, advantages, returns) as torch tensors"""
        indices = np.random.permutation(self.n_steps)

        for start in range(0, self.n_steps - batch_size + 1, batch_size):
            batch_idx = indices[start : start + batch_size]

            yield (
                torch.tensor(self.states[batch_idx]),
                torch.tensor(self.actions[batch_idx]),
                torch.tensor(self.log_probs[batch_idx]),
                torch.tensor(self.advantages[batch_idx]),
                torch.tensor(self.returns[batch_idx]),
            )

    def reset(self):
        self.ptr = 0


buffer = RolloutBuffer(ROLLOUT_STEPS, obs_dim, act_dim)


##############
# PPO Update #
##############

def ppo_update(buffer):
    """buffer: RolloutBuffer -> (mean_actor_loss, mean_critic_loss, mean_entropy)"""
    
    num_steps = 0
    total_actor_loss = 0
    total_critic_loss = 0
    total_entropy = 0
    total_ratio_mean = 0
    total_advantage_mean = 0

    for epoch in range(K_EPOCHS):
        for batch in buffer.get_minibatches(MINIBATCH_SIZE):
            states, actions, old_log_probs, advantages, returns = batch

            # Actor Loss
            _, new_log_probs, entropies, values = ac.get_action_and_value(states, actions)
            ratio = (new_log_probs - old_log_probs).exp()

            actor_loss = -torch.min(
                advantages * ratio, 
                advantages * torch.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            ).mean()

            # Critic Loss
            G = returns
            V = values
            # critic_loss = ( (G - V) ** 2 ).mean()
            critic_loss = nn.functional.smooth_l1_loss(input=V, target=G)

            # Backprop
            loss = actor_loss + VF_COEF * critic_loss - ENTROPY_COEF * entropies.mean()

            # print("loss", loss.item())
            # for name, param in ac.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} | grad_mean: {param.grad.mean().item()}")
            #     else:
            #         print(f"{name} | NO GRADIENT")

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(),  GRAD_CLIP)
            optimiser.step()

            num_steps += 1
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropies.mean().item()

            total_ratio_mean += ratio.mean().item()
            total_advantage_mean += advantages.mean().item()

    return (
        total_actor_loss / num_steps,
        total_critic_loss / num_steps, 
        total_entropy / num_steps,
        total_ratio_mean / num_steps,
        total_advantage_mean / num_steps,
    )


def record_episode(step):
    """Run one greedy episode and save video to VIDEO_DIR."""
    rec_env = gym.make(ENV_NAME, render_mode="rgb_array")
    rec_env = gym.wrappers.RecordVideo(
        rec_env,
        video_folder=VIDEO_DIR,
        name_prefix=f"step{step:07d}",
        episode_trigger=lambda _: True,
    )
    state, _ = rec_env.reset()
    ep_return = 0.0
    while True:
        with torch.no_grad():
            action, _, _, _ = ac.get_action_and_value(state)
        state, reward, terminated, truncated, _ = rec_env.step(action)
        ep_return += reward
        if terminated or truncated:
            break
    rec_env.close()
    print(f"  [video] step {step} — return: {ep_return:.1f} — saved to {VIDEO_DIR}/")
    return ep_return


#################
# Training Loop #
#################

"""
TODOs:
- [ ] Self Play
- [ ] Pool of policies
"""

np.random.seed(42)
state, _ = env.reset()
ep_return  = 0.0
ep_returns = []
global_step = 0
next_video  = 0

record_episode(0)

while global_step < MAX_STEPS:
    ENTROPY_COEF = max(MIN_ENTROPY_COEF, MAX_ENTROPY_COEF * (1 - global_step / MAX_STEPS))

    # Rollout collection
    for _ in range(ROLLOUT_STEPS):
        with torch.no_grad():
            action, log_prob, _, value = ac.get_action_and_value(state)

        next_state, reward, termination, truncated, _ = env.step(action)
        done = termination or truncated
        buffer.push(state, action, log_prob, reward, value, done)

        state = next_state

        ep_return += reward
        
        if done:
            ep_returns.append(ep_return)
            ep_return = 0.0

            ep_seed = np.random.randint(0, 2**31)
            state, _ = env.reset(seed=ep_seed)

        global_step += 1

        if global_step >= next_video:
            record_episode(global_step)
            next_video += VIDEO_EVERY

    # Bootstrap final value and compute GAE
    with torch.no_grad():
        _, _, _, last_value = ac.get_action_and_value(state)
        buffer.compute_gae(last_value, done)

    # PPO update #
    actor_loss, critic_loss, entropy, ratio_mean, advantage_mean = ppo_update(buffer)
    buffer.ptr = 0   # reset buffer

    if ep_returns:
        avg_100 = np.mean(ep_returns[-100:])
        print(f"step {global_step:7d} | avg(last 100 eps): {avg_100:7.1f} | "
              f"actor_loss: {actor_loss:7.3f} | critic_loss: {critic_loss:7.3f} | entropy: {entropy:.3f}")

    if WANDB_LOGGING:
        wandb.log({
            "avg_100":        np.mean(ep_returns[-100:]) if ep_returns else 0,
            "actor_loss":     actor_loss,
            "critic_loss":    critic_loss,
            "entropy":        entropy,
            "ratio_mean":     ratio_mean,
            "advantage_mean": advantage_mean
        }, step=global_step)

record_episode(global_step)

if WANDB_LOGGING:
    wandb.finish()

print(f"\nDone. Final avg (last 100): {np.mean(ep_returns[-100:]):.1f}")