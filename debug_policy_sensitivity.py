import os

os.environ.setdefault("DEBUG_MODE", "1")

import argparse
import numpy as np
import torch as t
import torch.nn.functional as F

from ppo_trainer import Trainer


def pooled_forward(net, obs, masked_cards):
    all_entities = t.cat([
        obs["my_cards"],
        obs["opponent_cards"],
        obs["my_crown_towers"],
        obs["opponent_crown_towers"],
    ], dim=1).to(dtype=t.float32)

    all_embeddings = net.entity_encoder(all_entities)
    n = net.max_num_cards

    my_card_embeddings = all_embeddings[:, 0:n]
    opponent_card_embeddings = all_embeddings[:, n:2 * n]
    my_tower_embeddings = all_embeddings[:, 2 * n:2 * n + 3]
    opponent_tower_embeddings = all_embeddings[:, 2 * n + 3:]

    def pool(embeddings, entities):
        if not masked_cards:
            return embeddings.mean(dim=1)

        mask = (entities.abs().sum(dim=-1, keepdim=True) > 0).to(dtype=embeddings.dtype)
        count = mask.sum(dim=1).clamp(min=1.0)
        return (embeddings * mask).sum(dim=1) / count

    trunk_input = t.cat([
        obs["game_completion_fraction"],
        obs["elixirs"],
        my_tower_embeddings.flatten(start_dim=1),
        opponent_tower_embeddings.flatten(start_dim=1),
        pool(my_card_embeddings, obs["my_cards"]),
        pool(opponent_card_embeddings, obs["opponent_cards"]),
    ], dim=-1).to(dtype=t.float32)

    trunk_out = net.trunk(trunk_input)
    return (
        net.actor_skip_net(trunk_out).squeeze(-1),
        net.actor_deck_idx_net(trunk_out),
        net.actor_position_net(trunk_out),
    )


def max_delta(cur, prev):
    if prev is None:
        return None
    cur = t.nan_to_num(cur, neginf=0.0, posinf=0.0)
    prev = t.nan_to_num(prev, neginf=0.0, posinf=0.0)
    return float((cur - prev).abs().max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trainer = Trainer("ClashRoyaleEnv-v0")
    net, _ = trainer.get_network_and_optimiser()
    state, _ = trainer.env.reset(seed=args.seed)

    prev_old = None
    prev_masked = None
    old_peaks = []
    masked_peaks = []

    print("step cards skip old_peak masked_peak old_dpos masked_dpos deck_probs")
    for step in range(args.steps):
        state_1, state_2 = trainer.split_observations(state)

        with t.no_grad():
            old_skip, old_deck, old_pos = pooled_forward(net, state_1, masked_cards=False)
            masked_skip, masked_deck, masked_pos = pooled_forward(net, state_1, masked_cards=True)

            if net.invalid_position_mask is not None:
                old_pos = old_pos.masked_fill(net.invalid_position_mask, float("-inf"))
                masked_pos = masked_pos.masked_fill(net.invalid_position_mask, float("-inf"))

            action_1, _, _, _ = net.get_action_and_value(state_1)
            action_2, _, _, _ = net.get_action_and_value(state_2)

        old_peak = int(t.argmax(old_pos, dim=-1).item())
        masked_peak = int(t.argmax(masked_pos, dim=-1).item())
        old_peaks.append(old_peak)
        masked_peaks.append(masked_peak)

        n_my = int((state_1["my_cards"].abs().sum(dim=-1) != 0).sum().item())
        n_opp = int((state_1["opponent_cards"].abs().sum(dim=-1) != 0).sum().item())
        deck_probs = F.softmax(masked_deck, dim=-1).squeeze(0).detach().cpu().numpy()

        print(
            f"{step:04d}",
            f"{n_my}/{n_opp}",
            f"{float(t.sigmoid(masked_skip).item()):.4f}",
            old_peak,
            masked_peak,
            "None" if prev_old is None else f"{max_delta(old_pos, prev_old):.5f}",
            "None" if prev_masked is None else f"{max_delta(masked_pos, prev_masked):.5f}",
            np.round(deck_probs, 3).tolist(),
        )

        prev_old = old_pos.detach()
        prev_masked = masked_pos.detach()

        state, _, terminated, truncated, _ = trainer.env.step(trainer.join_actions(action_1, action_2))
        if terminated or truncated:
            break

    print("unique_old_peaks", len(set(old_peaks)), sorted(set(old_peaks)))
    print("unique_masked_peaks", len(set(masked_peaks)), sorted(set(masked_peaks)))


if __name__ == "__main__":
    main()
