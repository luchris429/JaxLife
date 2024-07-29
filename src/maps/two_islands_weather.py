import jax
import jax.numpy as jnp
from world.structs import TerrainState


class TwoIslandsWeather:
    def __init__(self, config):
        self.config = config

    def initialize(self, rng, t=0):
        water = TerrainState(
            energy_amt=0.0,
            move_cost=8.0 * (jnp.sin(t / self.config["TERRAIN_PERIOD"]) + 1) / 2.0 + 0.5,
            push_cost=4.0 * (jnp.sin(t / self.config["TERRAIN_PERIOD"]) + 1) / 2.0 + 0.5,
            energy_gain=1.0 - (jnp.sin(t / self.config["TERRAIN_PERIOD"]) + 1) / 2.0,
            message_cost=0.0,
            reproduce_cost=8.0 * (jnp.sin(t / self.config["TERRAIN_PERIOD"]) + 1) / 2.0 + 1.0,
        )

        grass = TerrainState(
            energy_amt=self.config["MAX_TERRAIN_ENERGY"],
            move_cost=0.5,
            push_cost=0.5,
            energy_gain=1.0,
            message_cost=0.2,
            reproduce_cost=1.0,
        )

        terrain_state = TerrainState(
            energy_amt=jnp.ones((self.config["MAP_SIZE"], self.config["MAP_SIZE"])) * water.energy_amt,
            move_cost=jnp.ones((self.config["MAP_SIZE"], self.config["MAP_SIZE"])) * water.move_cost,
            push_cost=jnp.ones((self.config["MAP_SIZE"], self.config["MAP_SIZE"])) * water.push_cost,
            energy_gain=jnp.ones((self.config["MAP_SIZE"], self.config["MAP_SIZE"])) * water.energy_gain,
            message_cost=jnp.ones((self.config["MAP_SIZE"], self.config["MAP_SIZE"])) * water.message_cost,
            reproduce_cost=jnp.ones((self.config["MAP_SIZE"], self.config["MAP_SIZE"])) * water.reproduce_cost,
        )

        frac = self.config["MAP_SIZE"] // 4
        terrain_state = TerrainState(  # FIRST ISLAND
            energy_amt=terrain_state.energy_amt.at[frac : frac * 2, :].set(grass.energy_amt),
            move_cost=terrain_state.move_cost.at[frac : frac * 2, :].set(grass.move_cost),
            push_cost=terrain_state.push_cost.at[frac : frac * 2, :].set(grass.push_cost),
            energy_gain=terrain_state.energy_gain.at[frac : frac * 2, :].set(grass.energy_gain),
            message_cost=terrain_state.message_cost.at[frac : frac * 2, :].set(grass.message_cost),
            reproduce_cost=terrain_state.reproduce_cost.at[frac : frac * 2, :].set(grass.reproduce_cost),
        )

        terrain_state = TerrainState(  # SECOND ISLAND
            energy_amt=terrain_state.energy_amt.at[frac * 3 : frac * 4, :].set(grass.energy_amt),
            move_cost=terrain_state.move_cost.at[frac * 3 : frac * 4, :].set(grass.move_cost),
            push_cost=terrain_state.push_cost.at[frac * 3 : frac * 4, :].set(grass.push_cost),
            energy_gain=terrain_state.energy_gain.at[frac * 3 : frac * 4, :].set(grass.energy_gain),
            message_cost=terrain_state.message_cost.at[frac * 3 : frac * 4, :].set(grass.message_cost),
            reproduce_cost=terrain_state.reproduce_cost.at[frac * 3 : frac * 4, :].set(grass.reproduce_cost),
        )

        return terrain_state

    def update(self, world_state, rng):
        new_init = self.initialize(rng, world_state.t)
        new_terrain = world_state.terrain.replace(
            energy_amt=(1 - self.config["TERRAIN_ALPHA"]) * world_state.terrain.energy_amt
            + world_state.terrain.energy_gain,
            move_cost=new_init.move_cost,
            push_cost=new_init.push_cost,
            energy_gain=new_init.energy_gain,
            message_cost=new_init.message_cost,
            reproduce_cost=new_init.reproduce_cost,
        )
        return new_terrain