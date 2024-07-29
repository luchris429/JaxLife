import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict
from world.structs import Action


class ZeroBrain(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, carry, inputs):
        zero_action = Action(
            x_move=0.0,
            y_move=0.0,
            push=0.0,
            eat=0.0,
            hit=0.0,
            read_terrain_bit=0.0,
            reproduce=0.0,
            message_other=0.0,
            self_message=jnp.zeros(self.config["MESSAGE_SIZE"]),
            other_message=jnp.zeros(self.config["MESSAGE_SIZE"]),
            terrain_energy_gain=0.0,
            terrain_move_cost=0.0,
            terrain_push_cost=0.0,
            terrain_message_cost=0.0,
            terrain_reproduce_cost=0.0,
        )
        return zero_action, jnp.zeros((self.config["HSIZE"],))

    def initialize_carry(self, rng, input_shape):
        return jnp.zeros(input_shape)
