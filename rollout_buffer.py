import numpy as np
import torch as t


class RolloutBuffer:
    def __init__(self, n_steps, gae_gamma, gae_lambda):
        self.n_steps = n_steps
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda

        self.states      = None  # Lazy
        self.actions     = None
        self.log_probs   = t.zeros(n_steps, dtype=t.float32)
        self.rewards     = t.zeros(n_steps, dtype=t.float32)
        self.values      = t.zeros(n_steps, dtype=t.float32)
        self.dones       = t.zeros(n_steps, dtype=t.float32)
        self.advantages  = t.zeros(n_steps, dtype=t.float32)
        self.returns     = t.zeros(n_steps, dtype=t.float32)

        self.ptr = 0

        self.state_shapes, self.action_shapes = None, None  # cache for unfolding


    def push(
        self,
        state, action, log_prob, reward, value, 
        done
    ):
        state = self.flatten_dict(state, "state_shapes")
        action = self.flatten_dict(action, "action_shapes")

        if self.states is None:
            self.states = t.zeros((self.n_steps, state.shape[0]), dtype=t.float32)

        if self.actions is None:
            self.actions = t.zeros((self.n_steps, action.shape[0]), dtype=t.float32)

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
                next_done = self.dones[i]

            td_error = self.rewards[i] + self.gae_gamma * next_value * (1 - next_done) - self.values[i]
            advantage = td_error + self.gae_gamma * self.gae_lambda * (1 - next_done) * advantage

            self.advantages[i] = advantage

        # returns BEFORE normalizing advantages
        self.returns = self.advantages + self.values


    def get_minibatches(self, batch_size):
        """batch_size: int -> yields (states, actions, old_log_probs, advantages, returns) as torch tensors"""
        indices = np.random.permutation(self.n_steps)

        for start in range(0, self.n_steps - batch_size + 1, batch_size):
            batch_idx = indices[start : start + batch_size]

            # Per-minibatch advantage normalization
            mb_advantages = self.advantages[batch_idx]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            yield (
                self.unflatten_dict(self.states[batch_idx], self.state_shapes),
                self.unflatten_dict(self.actions[batch_idx], self.action_shapes),
                self.log_probs[batch_idx],
                mb_advantages,
                self.returns[batch_idx],
            )


    def flatten_dict(self, data, cache_attr):
        if getattr(self, cache_attr) is None:
            setattr(self, cache_attr, {
                k: v.shape if isinstance(v, t.Tensor) else None
                for k, v in data.items()
            })

        return t.cat([
            v.flatten() if isinstance(v, t.Tensor) else t.tensor(v)
            for v in data.values()
        ])


    def unflatten_dict(self, flattened_array, shapes):
        unflattened_dict = {}
        current_index = 0
        
        for key, shape in shapes.items():
            if shape is None:
                unflattened_dict[key] = flattened_array[current_index]
                current_index += 1
            else:
                size = np.prod(shape)
                
                array_slice = flattened_array[:, current_index : current_index + size]
                unflattened_dict[key] = array_slice.reshape(-1, *shape[1:])
            
                current_index += size
            
        return unflattened_dict


    def __len__(self):
        return self.n_steps

    
    def reset(self):
        self.ptr = 0
