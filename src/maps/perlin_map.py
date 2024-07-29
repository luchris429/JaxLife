import math

import jax
import jax.numpy as jnp

from maps.noise import generate_fractal_noise_2d
from world.structs import TerrainState
from maps.noise import interpolant


class Perlin:
    def __init__(self, config):
        self.config = config
        self.noise_res = (config["PERLIN_RES"], config["PERLIN_RES"])
        self.noise_angles_shape = (self.noise_res[0] + 1, self.noise_res[1] + 1)

    def _regenerate_base_state_from_angles(self, rng, noise_angles):
        map_shape = (self.config["MAP_SIZE"], self.config["MAP_SIZE"])

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, 7)

        override_angles = jnp.stack(
            [
                noise_angles.energy_amt,
                noise_angles.move_cost,
                noise_angles.push_cost,
                noise_angles.energy_gain,
                noise_angles.message_cost,
                noise_angles.reproduce_cost,
                noise_angles.max_energy,
            ],
            axis=0,
        )
        perlin_maps = jax.vmap(
            generate_fractal_noise_2d,
            in_axes=(0, None, None, None, None, None, None, 0),
        )(rngs, map_shape, self.noise_res, 1, 0.5, 2, interpolant, override_angles)

        return TerrainState(
            energy_amt=perlin_maps[0] * 0.0,
            move_cost=perlin_maps[1] * self.config["MAX_TERRAIN_MOVE_COST"],
            push_cost=perlin_maps[2] * self.config["MAX_TERRAIN_PUSH_COST"],
            energy_gain=perlin_maps[3] * self.config["MAX_TERRAIN_ENERGY_GAIN"],
            message_cost=perlin_maps[4] * self.config["MAX_TERRAIN_MESSAGE_COST"],
            reproduce_cost=perlin_maps[5] * self.config["MAX_TERRAIN_REPRODUCE_COST"],
            max_energy=perlin_maps[6] * self.config["MAX_TERRAIN_ENERGY"],
            # bits=jax.random.uniform(
            #     rngs[7], shape=(self.config["MAP_SIZE"], self.config["MAP_SIZE"])
            # ),
            bits=jnp.zeros((self.config["MAP_SIZE"], self.config["MAP_SIZE"]), dtype=jnp.float32),
        )

    def initialize(self, rng):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, 7)
        noise_angles = TerrainState(
            energy_amt=jax.random.uniform(rngs[0], self.noise_angles_shape) * 2 * math.pi,
            move_cost=jax.random.uniform(rngs[1], self.noise_angles_shape) * 2 * math.pi,
            push_cost=jax.random.uniform(rngs[2], self.noise_angles_shape) * 2 * math.pi,
            energy_gain=jax.random.uniform(rngs[3], self.noise_angles_shape) * 2 * math.pi,
            message_cost=jax.random.uniform(rngs[4], self.noise_angles_shape) * 2 * math.pi,
            reproduce_cost=jax.random.uniform(rngs[5], self.noise_angles_shape) * 2 * math.pi,
            max_energy=jax.random.uniform(rngs[6], self.noise_angles_shape) * 2 * math.pi,
            bits=jnp.zeros(self.noise_angles_shape, dtype=jnp.float32),
        )

        rng, _rng = jax.random.split(rng)
        base_state = self._regenerate_base_state_from_angles(_rng, noise_angles)
        base_state = base_state.replace(energy_amt=base_state.max_energy)

        return base_state, base_state, noise_angles

    def update(self, world_state, rng):
        # Perturb noise angles
        map_shape = (self.config["MAP_SIZE"], self.config["MAP_SIZE"])

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, 7)
        noise_angle_add = TerrainState(
            bits=jnp.zeros(self.noise_angles_shape, dtype=jnp.float32),
            energy_amt=jax.random.uniform(rngs[0], self.noise_angles_shape) * self.config["WEATHER_CHANGE_SPEED"],
            move_cost=jax.random.uniform(rngs[1], self.noise_angles_shape) * self.config["WEATHER_CHANGE_SPEED"],
            push_cost=jax.random.uniform(rngs[2], self.noise_angles_shape) * self.config["WEATHER_CHANGE_SPEED"],
            energy_gain=jax.random.uniform(rngs[3], self.noise_angles_shape) * self.config["WEATHER_CHANGE_SPEED"],
            message_cost=jax.random.uniform(rngs[4], self.noise_angles_shape) * self.config["WEATHER_CHANGE_SPEED"],
            reproduce_cost=jax.random.uniform(rngs[5], self.noise_angles_shape) * self.config["WEATHER_CHANGE_SPEED"],
            max_energy=jax.random.uniform(rngs[6], self.noise_angles_shape) * self.config["WEATHER_CHANGE_SPEED"],
        )
        new_noise_angles = jax.tree_map(lambda x, y: x + y, world_state.noise_angles, noise_angle_add)

        # Regenerate base state
        rng, _rng = jax.random.split(rng)
        new_base_terrain = self._regenerate_base_state_from_angles(_rng, new_noise_angles)

        # Regress to base state and grow energy
        regress_speed = (
            world_state.base_terrain.max_energy / self.config["MAX_TERRAIN_ENERGY"] * self.config["TERRAIN_ALPHA"]
        )

        new_terrain = world_state.terrain.replace(
            energy_amt=world_state.terrain.energy_amt
            + (
                world_state.terrain.energy_gain
                - self.config["MAX_TERRAIN_ENERGY_GAIN"] / self.config["TERRAIN_GAIN_SCALING"]
            ),
            move_cost=regress_speed * world_state.base_terrain.move_cost
            + (1 - regress_speed) * world_state.terrain.move_cost,
            push_cost=regress_speed * world_state.base_terrain.push_cost
            + (1 - regress_speed) * world_state.terrain.push_cost,
            energy_gain=regress_speed * world_state.base_terrain.energy_gain
            + (1 - regress_speed) * world_state.terrain.energy_gain,
            message_cost=regress_speed * world_state.base_terrain.message_cost
            + (1 - regress_speed) * world_state.terrain.message_cost,
            reproduce_cost=regress_speed * world_state.base_terrain.reproduce_cost
            + (1 - regress_speed) * world_state.terrain.reproduce_cost,
        )

        return new_terrain, new_base_terrain, new_noise_angles
