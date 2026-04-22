import torch as t
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(
        self, 
        
        entity_encoder_in_ch, 
        entity_encoder_mid_ch, 
        entity_encoder_out_ch,

        trunk_extra_in_ch,
        trunk_mid_ch,

        num_cards_in_deck,
        max_num_cards,
        position_space_width,
        position_space_height,

        invalid_position_mask=None,
    ):
        super().__init__()

        self.max_num_cards = max_num_cards
        self.position_space_width  = position_space_width
        self.position_space_height = position_space_height
        
        self.invalid_position_mask = invalid_position_mask


        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_encoder_in_ch, entity_encoder_mid_ch),
            nn.LayerNorm(entity_encoder_mid_ch),
            nn.ReLU(),

            nn.Linear(entity_encoder_mid_ch, entity_encoder_mid_ch),
            nn.LayerNorm(entity_encoder_mid_ch),
            nn.ReLU(),

            nn.Linear(entity_encoder_mid_ch, entity_encoder_out_ch),
        )

        self.trunk = nn.Sequential(
            nn.Linear(trunk_extra_in_ch + (3 + 3 + 1 + 1) * entity_encoder_out_ch, trunk_mid_ch),
            nn.LayerNorm(trunk_mid_ch),
            nn.ReLU(),

            nn.Linear(trunk_mid_ch, trunk_mid_ch),
            nn.LayerNorm(trunk_mid_ch),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            nn.Linear(trunk_mid_ch, 1)
        )

        self.actor_skip_net = nn.Sequential(
            nn.Linear(trunk_mid_ch, 1)
        )

        self.actor_deck_idx_net = nn.Sequential(
            nn.Linear(trunk_mid_ch, num_cards_in_deck)
        )

        self.actor_position_net = nn.Sequential(
            nn.Linear(trunk_mid_ch, position_space_width * position_space_height)
        )


    def forward(self, obs):
        """
        obs, which is just one player's, is expected to be a dict with:
        - game_completion_fraction: (B, 1)
        - elixirs: (B, 1)
        - my_cards: (B, N, card_dim)
            - where N is the upper cap on number of entities at once on the arena
            - zero padding is used
        - opponent_cards: (B, N, card_dim)
        - my_crown_towers: (B, 3, card_dim)
        - opponent_crown_towers: (B, 3, card_dim)
        """
        
        all_entities = t.cat([
            obs["my_cards"], 
            obs["opponent_cards"], 
            obs["my_crown_towers"], 
            obs["opponent_crown_towers"], 
        ], dim=1).to(dtype=t.float32)

        all_embeddings = self.entity_encoder(all_entities)

        my_card_embeddings       = all_embeddings[:, 0 : self.max_num_cards]
        opponent_card_embeddings = all_embeddings[:, self.max_num_cards : 2 * self.max_num_cards]

        my_crown_tower_embeddings       = all_embeddings[:, 2 * self.max_num_cards : 2 * self.max_num_cards + 3]
        opponent_crown_tower_embeddings = all_embeddings[:, 2 * self.max_num_cards + 3 :]

        trunk_input = t.cat([
            obs["game_completion_fraction"],
            obs["elixirs"],
            my_crown_tower_embeddings.flatten(start_dim=1),        # (B, 3 * entity_encoder_out_ch)
            opponent_crown_tower_embeddings.flatten(start_dim=1),  # (B, 3 * entity_encoder_out_ch)
            my_card_embeddings.mean(dim=1),        # (B, entity_encoder_out_ch)
            opponent_card_embeddings.mean(dim=1),  # (B, entity_encoder_out_ch)
        ], dim=-1).to(dtype=t.float32)  # (B, trunk_extra_in_ch + entity_encoder_out_ch)

        trunk_out = self.trunk(trunk_input)

        value = self.critic(trunk_out).squeeze(-1)  # (B,)

        skip_logits = self.actor_skip_net(trunk_out).squeeze(-1)  # (B,)
        deck_logits = self.actor_deck_idx_net(trunk_out)
        pos_logits = self.actor_position_net(trunk_out)
        
        return value, skip_logits, deck_logits, pos_logits


    def get_action_and_value(
        self, 
        obs, 
        action=None,
        invalid_deck_mask=None,
        invalid_position_mask=None,
    ):
        """
        obs: same as that taken by self.forward
        invalid_deck_mask: based on elixir or something more realistic like CR's random sampling in the deck
        invalid_position_mask: just your half of the arena is deployable into
        """
        value, skip_logits, deck_logits, pos_logits = self(obs)

        if invalid_deck_mask is not None:
            deck_logits = deck_logits.masked_fill(invalid_deck_mask, float('-inf'))
        if invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(invalid_position_mask, float('-inf'))
        if self.invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(self.invalid_position_mask, float('-inf'))
        
        skip_dist = t.distributions.Bernoulli(logits=skip_logits)
        deck_dist = t.distributions.Categorical(logits=deck_logits)
        pos_dist  = t.distributions.Categorical(logits=pos_logits)

        if action is None:
            action_skip = skip_dist.sample()
            action_deck = deck_dist.sample()
            action_pos  = pos_dist.sample()
        else:
            action_skip = action["skip"]
            action_deck = action["deck_idx"]
            action_pos  = action["position"]

        # Log Probs
        skip_log_prob = skip_dist.log_prob(action_skip)
        deck_log_prob = deck_dist.log_prob(action_deck)
        pos_log_prob  = pos_dist.log_prob(action_pos)

        log_prob = skip_log_prob + deck_log_prob + pos_log_prob

        # Entropy
        skip_entropy = skip_dist.entropy()
        deck_entropy = deck_dist.entropy()
        pos_entropy  = pos_dist.entropy()

        entropy = skip_entropy + deck_entropy + pos_entropy

        action = {
            "skip": action_skip.detach(), 
            "deck_idx": action_deck.detach(), 
            "position": action_pos.detach()
        }

        return action, log_prob, entropy, value
