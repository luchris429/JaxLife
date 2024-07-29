import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from typing import Dict
from agent_brains.brain import logits_to_actions
from world.structs import Action
from flax.linen.initializers import constant, orthogonal


class ProfilingBrain(nn.RNNCellBase):
    config: Dict

    def setup(self):
        if self.config.get("PROFILING_ACTION_ONLY"):
            self.action_logits = self.param(
                "action_logits",
                nn.initializers.zeros,
                (6 + self.config["MESSAGE_SIZE"] * 2 + 5,),
            )
        elif self.config.get("PROFILING_ACTION_2"):
            self.action_process = nn.Dense(
                self.config["HSIZE"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
            self.action_logits = nn.Dense(6 + self.config["MESSAGE_SIZE"] * 2 + 5, bias_init=constant(0.0))
            self.dense_cell = nn.Dense(
                self.config["HSIZE"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
        elif self.config.get("PROFILING_CONST"):
            self.action_process = nn.Dense(
                self.config["HSIZE"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
            self.action_logits = nn.Dense(6 + self.config["MESSAGE_SIZE"] * 2 + 5, bias_init=constant(0.0))
            self.dense_cell = nn.Dense(
                self.config["HSIZE"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
        else:
            self.agent_encoder = nn.Dense(
                self.config["HSIZE"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
            self.agent_seq = nn.SelfAttention(num_heads=4, deterministic=True)
            self.agent_dec = nn.MultiHeadDotProductAttention(num_heads=4, deterministic=True)

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

            # self.lstm_cell = nn.LSTMCell(self.config["HSIZE"])
            self.dense_cell = nn.Dense(
                self.config["HSIZE"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )

    def __call__(self, carry, inputs):
        if self.config.get("PROFILING_ACTION_ONLY"):
            print("PROFILING ACTION ONLY")
            action_logits_a = jnp.tanh(self.action_logits)
        elif self.config.get("PROFILING_ACTION_2"):
            print("PROFILING ACTION ONLY 2")
            everything_flattened = jnp.ones((1,))

            # h = self.action_logits(everything_flattened[None,...]).squeeze(0) <= THIS IS FAST
            # action_logits_a = jnp.tanh(h)

            # h = self.dense_cell(everything_flattened[None,...]) # <= THIS IS SLOW AND THEYRE JUST 1x1 MATMULS...
            # action_logits_a = self.action_process(h)
            # action_logits_a = jnp.tanh(action_logits_a)
            # action_logits_a = self.action_logits(action_logits_a).squeeze(0)
            # action_logits_a = jnp.tanh(action_logits_a)

            # action_logits_a = everything_flattened[None,...] # <= THIS IS FAST
            # action_logits_a = jnp.tanh(action_logits_a)
            # action_logits_a = self.action_logits(action_logits_a).squeeze(0)
            # action_logits_a = jnp.tanh(action_logits_a)

            h = everything_flattened[None, ...]  # <= THIS IS SLOW
            action_logits_a = self.action_process(h)
            action_logits_a = jnp.tanh(action_logits_a)
            action_logits_a = self.action_logits(action_logits_a).squeeze(0)
            action_logits_a = jnp.tanh(action_logits_a)

        else:
            # grid_lc, agent_attrs_na, self_attrs = inputs["grid"], inputs["agents"], inputs["self"]
            grid_lc, agent_attrs_na, self_attrs = (
                inputs.grid,
                inputs.agents,
                inputs.self,
            )
            if self.config.get("PROFILING_TERRAIN_ONLY"):
                print("ONLY INPUT TERRAIN")
                everything_flattened = jnp.concatenate([grid_lc.reshape((-1,)), self_attrs.reshape((-1,))], axis=0)
            elif self.config.get("PROFILING_CONST"):
                print("INPUT CONST")
                everything_flattened = jnp.ones((1,))
            else:
                print("INPUT EVERYTHING")
                everything_flattened = jnp.concatenate(
                    [
                        grid_lc.reshape((-1,)),
                        agent_attrs_na.reshape((-1,)),
                        self_attrs.reshape((-1,)),
                    ],
                    axis=0,
                )
            h = self.dense_cell(everything_flattened[None, ...])

            action_logits_a = self.action_process(h)
            action_logits_a = jnp.tanh(action_logits_a)
            action_logits_a = self.action_logits(action_logits_a).squeeze(0)
            action_logits_a = jnp.tanh(action_logits_a)

        actions = logits_to_actions(action_logits_a, self.config)

        return actions, None

    @classmethod
    def initialize_carry(cls, rng, input_shape):
        return None
