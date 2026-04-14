from gymnasium import spaces
from gymnasium.spaces.utils import flatten, flatten_space
import gymnasium as gym
import numpy as np


class CRFlattenNormWrapper(gym.ObservationWrapper):
    SEQUENCE_KEYS = ("player_1_cards", "player_2_cards")
    TOWER_SUBKEYS = ("king_tower", "princess_tower_1", "princess_tower_2")

    def __init__(self, env):
        super().__init__(env)
        base = env.observation_space

        self._card_space  = base["player_1_cards"].feature_space
        self._tower_space = base["player_1_crown_towers"]["king_tower"]

        flat_card_space  = flatten_space(self._card_space)
        flat_tower_space = flatten_space(self._tower_space)

        # Precompute normalization constants from flattened space bounds
        self._card_mid,  self._card_half  = self._bounds(flat_card_space)
        self._tower_mid, self._tower_half = self._bounds(flat_tower_space)

        self.observation_space = spaces.Dict({
            "game_completion_fraction": base["game_completion_fraction"],
            "player_1_elixirs":         base["player_1_elixirs"],
            "player_1_cards":           spaces.Sequence(flat_card_space),
            "player_1_crown_towers":    spaces.Tuple((flat_tower_space,) * 3),
            "player_2_elixirs":         base["player_2_elixirs"],
            "player_2_cards":           spaces.Sequence(flat_card_space),
            "player_2_crown_towers":    spaces.Tuple((flat_tower_space,) * 3),
        })
        
        # * DEBUG *
        self._tower_labels = self._get_feature_labels(self._tower_space)
        self._card_labels  = self._get_feature_labels(self._card_space)

    @staticmethod
    def _bounds(flat_box_space):
        mid       = (flat_box_space.low + flat_box_space.high) / 2.0
        half_span = (flat_box_space.high - flat_box_space.low) / 2.0
        half_span = np.where(half_span == 0, 1.0, half_span)  # avoid div/0
        return mid.astype(np.float32), half_span.astype(np.float32)

    @staticmethod
    def _norm(x, mid, half_span):
        return np.clip((x - mid) / half_span, -1.0, 1.0).astype(np.float32)

    def _flat_norm_card(self, card):
        return self._norm(flatten(self._card_space, card), self._card_mid, self._card_half)

    def _flat_norm_tower(self, tower):
        return self._norm(flatten(self._tower_space, tower), self._tower_mid, self._tower_half)

    def observation(self, obs):
        return {
            "game_completion_fraction": obs["game_completion_fraction"] * 2.0 - 1.0,
            "player_1_elixirs":         (obs["player_1_elixirs"] / 10) * 2.0 - 1.0,
            "player_1_cards":           [self._flat_norm_card(c) for c in obs["player_1_cards"]],
            "player_1_crown_towers":    tuple(self._flat_norm_tower(obs["player_1_crown_towers"][k]) for k in self.TOWER_SUBKEYS),
            "player_2_elixirs":         (obs["player_2_elixirs"] / 10) * 2.0 - 1.0,
            "player_2_cards":           [self._flat_norm_card(c) for c in obs["player_2_cards"]],
            "player_2_crown_towers":    tuple(self._flat_norm_tower(obs["player_2_crown_towers"][k]) for k in self.TOWER_SUBKEYS),
        }

    def _get_feature_labels(self, space):
        """Returns list of string labels, one per flat feature index."""
        labels = []
        for key, subspace in space.spaces.items():
            flat_sub = flatten_space(subspace)
            n = flat_sub.shape[0] if flat_sub.shape else 1
            if n == 1:
                labels.append(key)
            else:
                labels.extend(f"{key}[{i}]" for i in range(n))
        return labels