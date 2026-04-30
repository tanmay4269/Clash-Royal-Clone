import wandb
import numpy as np
import torch as t
import plotly.graph_objects as go


class DiagnosticsLogger:
    def __init__(self, cfg, wandb_logging=True):
        self.cfg = cfg
        self.wandb_logging = wandb_logging

        self.reset_episode_stats()
        self.reset_buffer_stats()

    def reset_episode_stats(self):
        self.current_ep_elixir_sum = 0
        self.current_ep_steps = 0
        self.ep_skips = 0
        self.ep_decks = []
        self.ep_pos_x = []
        self.ep_pos_y = []

    def reset_buffer_stats(self):
        self.buffer_games_completed = 0
        self.buffer_games_terminated = 0
        self.buffer_games_truncated = 0
        self.buffer_towers_killed_by_p1 = 0
        self.buffer_towers_killed_by_p2 = 0

    def on_step(self, action_1, env):
        # Collect metrics for Player 1
        action_1_cpu = {k: v.cpu().numpy() for k, v in action_1.items()}
        self.current_ep_steps += 1
        
        # We unwrap twice to ensure we reach the underlying cr_gym_env if wrapped
        arena = env.unwrapped.arena
        self.current_ep_elixir_sum += arena.player_side_1.elixirs
        
        self.ep_skips += int(action_1_cpu["skip"].item())
        if action_1_cpu["skip"].item() == 0:
            self.ep_decks.append(int(action_1_cpu["deck_idx"].item()))
            pos = int(action_1_cpu["position"].item())
            self.ep_pos_x.append(pos % arena.width)
            self.ep_pos_y.append(pos // arena.width)

    def on_episode_end(self, env, terminated, truncated, current_elo, ep_return, score, global_step):
        self.buffer_games_completed += 1
        if terminated:
            self.buffer_games_terminated += 1
        if truncated:
            self.buffer_games_truncated += 1

        towers_killed_by_p1 = 0
        towers_killed_by_p2 = 0
        for t_name in ["king_tower", "princess_tower_1", "princess_tower_2"]:
            if float(env.unwrapped._cur_obs["player_2_crown_towers"][t_name]["health"]) <= 0:
                towers_killed_by_p1 += 1
            if float(env.unwrapped._cur_obs["player_1_crown_towers"][t_name]["health"]) <= 0:
                towers_killed_by_p2 += 1

        self.buffer_towers_killed_by_p1 += towers_killed_by_p1
        self.buffer_towers_killed_by_p2 += towers_killed_by_p2

        if self.wandb_logging:
            log_dict = {
                "elo": current_elo,
                "return": ep_return,
                "score": score,
                "game_diagnostics/ep_duration_frames": self.current_ep_steps,
                "game_diagnostics/avg_elixir_p1": self.current_ep_elixir_sum / max(1, self.current_ep_steps),
                "action_diagnostics/skip_ratio": self.ep_skips / max(1, self.current_ep_steps),
            }

            if self.ep_decks:
                log_dict["action_diagnostics/deck_idx_hist"] = wandb.Histogram(self.ep_decks)
                
                # Plotly stacked bar chart
                counts = np.bincount(self.ep_decks, minlength=self.cfg.network.num_cards_in_deck)
                proportions = counts / max(1, sum(counts))
                fig = go.Figure(data=[
                    go.Bar(name=f"Card {i}", x=["Deck"], y=[proportions[i]])
                    for i in range(self.cfg.network.num_cards_in_deck)
                ])
                fig.update_layout(barmode='stack', title="Deck Usage Proportions")
                log_dict["action_diagnostics/deck_stacked_chart"] = wandb.Html(fig.to_html(auto_play=False))

            if self.ep_pos_x:
                log_dict["action_diagnostics/pos_x_hist"] = wandb.Histogram(self.ep_pos_x)
                log_dict["action_diagnostics/pos_y_hist"] = wandb.Histogram(self.ep_pos_y)

            wandb.log(log_dict, step=global_step)

        self.reset_episode_stats()

    def on_ppo_update(self, global_step, buffer, net_curr, net_init, net_prev):
        if not self.wandb_logging:
            self.reset_buffer_stats()
            return
            
        # Log buffer specific game_diagnostics
        wandb.log({
            "game_diagnostics/buffer_games_completed": self.buffer_games_completed,
            "game_diagnostics/buffer_games_truncated": self.buffer_games_truncated,
            "game_diagnostics/buffer_games_terminated": self.buffer_games_terminated,
            "game_diagnostics/avg_towers_killed_by_p1": self.buffer_towers_killed_by_p1 / max(1, self.buffer_games_completed),
            "game_diagnostics/avg_towers_killed_by_p2": self.buffer_towers_killed_by_p2 / max(1, self.buffer_games_completed),
        }, step=global_step)

        self.reset_buffer_stats()

        # Compute per head diagnostics
        with t.no_grad():
            batch = next(iter(buffer.get_minibatches(self.cfg.minibatch_size)))
            states_tensor = batch[0]
            
            _, skip_log_curr, deck_log_curr, pos_log_curr = net_curr(states_tensor)
            _, skip_log_init, deck_log_init, pos_log_init = net_init(states_tensor)
            _, skip_log_prev, deck_log_prev, pos_log_prev = net_prev(states_tensor)

            # Distributions
            skip_dist_curr = t.distributions.Bernoulli(logits=skip_log_curr)
            deck_dist_curr = t.distributions.Categorical(logits=deck_log_curr)
            pos_dist_curr = t.distributions.Categorical(logits=pos_log_curr)

            skip_dist_init = t.distributions.Bernoulli(logits=skip_log_init)
            deck_dist_init = t.distributions.Categorical(logits=deck_log_init)
            pos_dist_init = t.distributions.Categorical(logits=pos_log_init)

            skip_dist_prev = t.distributions.Bernoulli(logits=skip_log_prev)
            deck_dist_prev = t.distributions.Categorical(logits=deck_log_prev)
            pos_dist_prev = t.distributions.Categorical(logits=pos_log_prev)

            ent_skip = skip_dist_curr.entropy().mean().item()
            ent_deck = deck_dist_curr.entropy().mean().item()
            ent_pos = pos_dist_curr.entropy().mean().item()

            kl_skip_init = t.distributions.kl.kl_divergence(skip_dist_curr, skip_dist_init).mean().item()
            kl_deck_init = t.distributions.kl.kl_divergence(deck_dist_curr, deck_dist_init).mean().item()
            kl_pos_init = t.distributions.kl.kl_divergence(pos_dist_curr, pos_dist_init).mean().item()

            kl_skip_prev = t.distributions.kl.kl_divergence(skip_dist_curr, skip_dist_prev).mean().item()
            kl_deck_prev = t.distributions.kl.kl_divergence(deck_dist_curr, deck_dist_prev).mean().item()
            kl_pos_prev = t.distributions.kl.kl_divergence(pos_dist_curr, pos_dist_prev).mean().item()

        wandb.log({
            "per_head_diagnostics/entropy/skip": ent_skip,
            "per_head_diagnostics/entropy/deck_idx": ent_deck,
            "per_head_diagnostics/entropy/position": ent_pos,
            "per_head_diagnostics/kl_vs_initial/skip": kl_skip_init,
            "per_head_diagnostics/kl_vs_initial/deck_idx": kl_deck_init,
            "per_head_diagnostics/kl_vs_initial/position": kl_pos_init,
            "per_head_diagnostics/kl_vs_pre_update/skip": kl_skip_prev,
            "per_head_diagnostics/kl_vs_pre_update/deck_idx": kl_deck_prev,
            "per_head_diagnostics/kl_vs_pre_update/position": kl_pos_prev,
        }, step=global_step)
