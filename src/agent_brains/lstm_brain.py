import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from typing import Dict
from agent_brains.brain import logits_to_actions
from world.structs import Action
from flax.linen.initializers import constant, orthogonal


class LSTMBrain(nn.RNNCellBase):
    config: Dict

    def setup(self):
        self.agent_encoder = nn.Dense(
            self.config["HSIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.bot_encoder = nn.Dense(
            self.config["HSIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )

        self.entity_seq = nn.SelfAttention(num_heads=4, deterministic=True)
        self.entity_dec = nn.MultiHeadDotProductAttention(num_heads=4, deterministic=True)

        self.terrain_encoder = nn.Dense(
            self.config["HSIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.terrain_dec = nn.Dense(
            self.config["HSIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )

        self.action_process = nn.Dense(
            self.config["HSIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.action_logits = nn.Dense(6 + self.config["MESSAGE_SIZE"] * 2 + 5, bias_init=constant(0.0))

        self.lstm_cell = nn.LSTMCell(self.config["HSIZE"])

    def __call__(self, carry, inputs):
        grid_lc, agent_attrs_na, bot_attrs_na, self_attrs = inputs.grid, inputs.agents, inputs.bots, inputs.self

        agent_attr_embeddings_nh = self.agent_encoder(agent_attrs_na)
        agent_attr_embeddings_nh = nn.tanh(agent_attr_embeddings_nh)

        bot_attr_embeddings_nh = self.bot_encoder(bot_attrs_na)
        bot_attr_embeddings_nh = nn.tanh(bot_attr_embeddings_nh)

        self_attrs_h = self.agent_encoder(self_attrs)
        self_attrs_h = nn.tanh(self_attrs_h)  # Q: Should this be here?

        entity_attr_embeddings_nh = jnp.concatenate([agent_attr_embeddings_nh, bot_attr_embeddings_nh], axis=0)

        entities_nh = self.entity_seq(entity_attr_embeddings_nh)

        grid_lh = self.terrain_encoder(grid_lc)
        grid_h = self.terrain_dec(grid_lh.reshape((-1,)))

        entities_h = self.entity_dec(self_attrs_h[None, ...], entities_nh).squeeze(0)

        h = jnp.concatenate([entities_h, grid_h], axis=0)
        carry, h = self.lstm_cell(carry, h)

        action_logits_a = self.action_process(h[None, ...])
        action_logits_a = jnp.tanh(action_logits_a)
        action_logits_a = self.action_logits(action_logits_a).squeeze(0)
        action_logits_a = jnp.tanh(action_logits_a)

        actions = logits_to_actions(action_logits_a, self.config)

        return actions, carry

    @classmethod
    def initialize_carry(cls, rng, input_shape):
        return nn.LSTMCell(64).initialize_carry(rng, input_shape)

    def calc_message_influence(self, params):
        import pdb

        pdb.set_trace()
