import jax
import numpy as np
import jax.numpy as jnp
from world.structs import TerrainState
from maps.random_init import Random
from maps.four_islands import FourIslands
from maps.two_islands import TwoIslands
from maps.four_islands_weather import FourIslandsWeather
from maps.two_islands_weather import TwoIslandsWeather
from maps.perlin_map import Perlin


class Terrain:
    def __init__(self, config):
        self.config = config
        if self.config["TERRAIN_INIT"] == "random":
            self.terrain = Random(self.config)
        elif self.config["TERRAIN_INIT"] == "four_islands":
            self.terrain = FourIslands(self.config)
        elif self.config["TERRAIN_INIT"] == "two_islands":
            self.terrain = TwoIslands(self.config)
        elif self.config["TERRAIN_INIT"] == "four_islands_weather":
            self.terrain = FourIslandsWeather(self.config)
        elif self.config["TERRAIN_INIT"] == "two_islands_weather":
            self.terrain = TwoIslandsWeather(self.config)
        elif self.config["TERRAIN_INIT"] == "perlin":
            self.terrain = Perlin(self.config)
        else:
            raise Exception(f"Unknown terrain init {self.config['TERRAIN_INIT']}")

    def initialize(self, rng):
        return self.terrain.initialize(rng)

    def update(self, world_state, rng):
        return self.terrain.update(world_state, rng)
