import jax
import jax.numpy as jnp
from world.structs import TerrainState


class Random:
    def __init__(self, config):
        self.config = config

    def initialize(self, rng):
        (
            rng_energy_amt,
            rng_move_cost,
            rng_push_cost,
            rng_energy_gain,
            rng_message_cost,
            rng_reproduce_cost,
            rng_bits,
        ) = jax.random.split(rng, 7)
        terrain_states = TerrainState(
            energy_amt=jax.random.uniform(rng_energy_amt, shape=(self.config["MAP_SIZE"], self.config["MAP_SIZE"]))
            * 8.0,
            move_cost=jax.random.uniform(rng_move_cost, shape=(self.config["MAP_SIZE"], self.config["MAP_SIZE"])),
            push_cost=jax.random.uniform(rng_push_cost, shape=(self.config["MAP_SIZE"], self.config["MAP_SIZE"])),
            energy_gain=jax.random.uniform(rng_energy_gain, shape=(self.config["MAP_SIZE"], self.config["MAP_SIZE"])),
            message_cost=jax.random.uniform(rng_message_cost, shape=(self.config["MAP_SIZE"], self.config["MAP_SIZE"])),
            reproduce_cost=jax.random.uniform(
                rng_reproduce_cost, shape=(self.config["MAP_SIZE"], self.config["MAP_SIZE"])
            ),
            bits=jax.random.uniform(rng_bits, shape=(self.config["MAP_SIZE"], self.config["MAP_SIZE"])) > 0.5,
        )
        return terrain_states

    def update(self, world_state, rng):
        new_init = self.initialize(rng)
        new_terrain = world_state.terrain.replace(
            energy_amt=(1 - self.config["TERRAIN_ALPHA"]) * world_state.terrain.energy_amt
            + world_state.terrain.energy_gain,
            move_cost=self.config["TERRAIN_ALPHA"] * new_init.move_cost
            + (1 - self.config["TERRAIN_ALPHA"]) * world_state.terrain.move_cost,
            push_cost=self.config["TERRAIN_ALPHA"] * new_init.push_cost
            + (1 - self.config["TERRAIN_ALPHA"]) * world_state.terrain.push_cost,
            energy_gain=self.config["TERRAIN_ALPHA"] * new_init.energy_gain
            + (1 - self.config["TERRAIN_ALPHA"]) * world_state.terrain.energy_gain,
            message_cost=self.config["TERRAIN_ALPHA"] * new_init.message_cost
            + (1 - self.config["TERRAIN_ALPHA"]) * world_state.terrain.message_cost,
            reproduce_cost=self.config["TERRAIN_ALPHA"] * new_init.reproduce_cost
            + (1 - self.config["TERRAIN_ALPHA"]) * world_state.terrain.reproduce_cost,
        )
        return new_terrain
