import time
import gymnasium as gym
import cr_gym_env
from cr_flatten_norm_wrapper import CRFlattenNormWrapper
from parallel_env import ParallelEnvManager
import numpy as np

def profile_single_env(num_steps=1000):
    env = gym.make("ClashRoyaleEnv-v0")
    env = CRFlattenNormWrapper(env)
    
    env.reset()
    
    start_time = time.time()
    for _ in range(num_steps):
        action = env.action_space.sample()
        # Force a lot of troops to spawn to stress test the collision/rendering
        action["player_1_skip"] = 0
        action["player_2_skip"] = 0
        
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
            
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_steps / total_time
    
    print(f"[Single Env] Total Time for {num_steps} steps: {total_time:.2f}s")
    print(f"[Single Env] Steps per second (FPS): {fps:.2f}")

def profile_parallel_env(num_envs=8, num_steps_per_env=250):
    # Total steps = num_envs * num_steps_per_env = 8 * 250 = 2000
    penv = ParallelEnvManager(
        num_envs=num_envs,
        env_name="ClashRoyaleEnv-v0",
        env_kwargs={},
        wrapper_cls=CRFlattenNormWrapper,
    )
    
    penv.reset([42 + i for i in range(num_envs)])
    
    # We need an action space to sample from
    env = gym.make("ClashRoyaleEnv-v0")
    env = CRFlattenNormWrapper(env)
    action_space = env.action_space
    
    start_time = time.time()
    for _ in range(num_steps_per_env):
        actions = []
        for _ in range(num_envs):
            action = action_space.sample()
            action["player_1_skip"] = 0
            action["player_2_skip"] = 0
            actions.append(action)
            
        results = penv.step(actions)
        
        # Reset if done
        for i, res in enumerate(results):
            if res[2] or res[3]: # terminated or truncated
                penv.reset_single(i, seed=np.random.randint(0, 10000))
                
    end_time = time.time()
    total_time = end_time - start_time
    total_steps = num_envs * num_steps_per_env
    fps = total_steps / total_time
    
    print(f"[Parallel Env ({num_envs} workers)] Total Time for {total_steps} steps: {total_time:.2f}s")
    print(f"[Parallel Env ({num_envs} workers)] Total Steps per second: {fps:.2f}")
    print(f"[Parallel Env ({num_envs} workers)] Steps per second per worker: {fps / num_envs:.2f}")

if __name__ == '__main__':
    print("Profileing Single Environment...")
    profile_single_env(num_steps=500)
    
    print("\nProfileing 8x Parallel Environments...")
    profile_parallel_env(num_envs=8, num_steps_per_env=125) # 1000 total steps
