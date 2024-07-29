import jax
import jax.numpy as jnp
import flax.linen as nn
import mediapy as media
import tqdm


class Visualizer:
    def __init__(self, config):
        self.config = config

    def calc_dists(self, agent_coord, pixel_coord):
        return jnp.linalg.norm(agent_coord - pixel_coord)

    def render_pixel(self, world_state, coords):
        """
        This also seems horrendously inefficient...
        """
        agent_coords = jnp.stack([world_state.agent_states.pos_x, world_state.agent_states.pos_y], axis=-1)
        agent_dists = jax.vmap(self.calc_dists, in_axes=(0, None))(agent_coords, coords)
        agent_dists = agent_dists + (1.0 - world_state.agent_states.alive) * 1000.0
        agent_dists = agent_dists.min()
        bot_coords = jnp.stack([world_state.bot_states.pos_x, world_state.bot_states.pos_y], axis=-1)
        bot_dists = jax.vmap(self.calc_dists, in_axes=(0, None))(bot_coords, coords)
        bot_dists = bot_dists + (1.0 - world_state.bot_states.alive) * 1000.0
        iii = bot_dists.argmin()
        bot_bit = world_state.bot_states.memory[iii, -1]
        bot_dists = bot_dists.min()

        terrain_val_energy_amt = (
            world_state.terrain.energy_amt[
                (coords[0] // self.config["CELL_SIZE"]).astype(jnp.int32),
                (coords[1] // self.config["CELL_SIZE"]).astype(jnp.int32),
            ].astype(float)
            / world_state.terrain.max_energy[
                (coords[0] // self.config["CELL_SIZE"]).astype(jnp.int32),
                (coords[1] // self.config["CELL_SIZE"]).astype(jnp.int32),
            ]
        )

        terrain_val_max_energy = (
            world_state.terrain.max_energy[
                (coords[0] // self.config["CELL_SIZE"]).astype(jnp.int32),
                (coords[1] // self.config["CELL_SIZE"]).astype(jnp.int32),
            ].astype(float)
            / self.config["MAX_TERRAIN_ENERGY"]
        )

        terrain_val_move_cost = (
            +world_state.terrain.move_cost[
                (coords[0] // self.config["CELL_SIZE"]).astype(jnp.int32),
                (coords[1] // self.config["CELL_SIZE"]).astype(jnp.int32),
            ].astype(float)
            / self.config["MAX_TERRAIN_MOVE_COST"]
        )

        terrain_scalar = (terrain_val_max_energy - terrain_val_move_cost + 1.0) / 2.0
        terrain_scalar = jnp.clip(terrain_scalar, 0.0, 1.0)

        terrain_scalar_int = terrain_scalar
        # terrain_scalar_int -= (terrain_scalar_int < 0.5) * 0.14
        # terrain_scalar_int += (terrain_scalar_int > 0.5) * 0.14
        # terrain_scalar_int = jnp.clip(terrain_scalar_int, 0., 1.)
        terrain_scalar_int = jnp.floor(terrain_scalar_int * 2).astype(int)

        water_colour = jnp.array([35, 137, 218]) / 255.0
        dark_water_colour = jnp.array([15, 94, 156]) / 255.0
        sand_colour = jnp.array([246, 215, 176]) / 255.0
        grass_colour = jnp.array([19, 109, 21]) / 255.0
        green_grass_colour = jnp.array([65, 152, 10]) / 255.0
        mud_colour = jnp.array([112, 84, 62]) / 255.0
        dark_sand_colour = jnp.array([225, 191, 146]) / 255.0

        tg = (terrain_scalar - 0.5) * 2
        mixed_grass_colour = tg * green_grass_colour + (1 - tg) * grass_colour

        tw = terrain_scalar * 2
        mixed_water_colour = tw * water_colour + (1 - tw) * dark_water_colour

        # terrain_val = jax.lax.switch(terrain_scalar, [
        #     lambda: water_colour,
        #     lambda: sand_colour,
        #     lambda: grass_colour,
        #     lambda: stone_colour
        # ])

        terrain_val = jax.lax.switch(
            terrain_scalar_int,
            [
                lambda: mixed_water_colour,
                # lambda: terrain_val_energy_amt * sand_colour + (1 - terrain_val_energy_amt) * dark_sand_colour,
                lambda: terrain_val_energy_amt * mixed_grass_colour + (1 - terrain_val_energy_amt) * mud_colour,
            ],
        )

        # if self.config.get("TERRAIN_RENDER_BITS", False):
        #     terrain_val = (
        #         world_state.terrain.bits[
        #             (coords[0] // self.config["CELL_SIZE"]).astype(jnp.int32),
        #             (coords[1] // self.config["CELL_SIZE"]).astype(jnp.int32),
        #         ]
        #         * jnp.array([1.0, 1.0, 1.0])
        #     )

        agent_val = (
            nn.relu(self.config["AGENT_RADIUS"] - agent_dists)
            * jnp.array([1.0, 0.0, 0.0])
            / self.config["AGENT_RADIUS"]
        )
        bot_colour = jnp.array([0.7, 0.7, 0.7])
        if self.config.get("BOT_RENDER_BITS", False):
            bot_colour = jnp.array([0.7, 0, 0]) * bot_bit + (1 - bot_bit) * jnp.array([0.0, 0.7, 0])
        bot_val = nn.relu(self.config["BOT_RADIUS"] - bot_dists) * bot_colour / self.config["BOT_RADIUS"]
        entity_val = jnp.where(agent_dists < bot_dists, agent_val, bot_val)
        pixel_val = jnp.where(
            jnp.logical_or(
                agent_dists < self.config["AGENT_RADIUS"],
                bot_dists < self.config["BOT_RADIUS"],
            ),
            entity_val,
            terrain_val,
        )
        return pixel_val

    def render_frame(self, world_state):
        w, h = self.config["IMG_SIZE_W"], self.config["IMG_SIZE_H"]
        min_x, min_y = 0, 0
        max_x, max_y = (
            self.config["MAP_SIZE"] * self.config["CELL_SIZE"],
            self.config["MAP_SIZE"] * self.config["CELL_SIZE"],
        )
        coords = jnp.mgrid[min_x : max_x : complex(0, w), min_y : max_y : complex(0, h)].reshape(2, -1).T
        pixel_vals = jax.vmap(self.render_pixel, in_axes=(None, 0))(world_state, coords)
        img = pixel_vals.reshape((w, h, 3))
        return img

    def render_frame_idx(self, world_states, i):
        return self.render_frame(jax.tree_map(lambda x: x[i], world_states))

    def render(self, world_states, fname="out.mp4", show=True):
        first_frame = jax.jit(self.render_frame)(jax.tree_map(lambda x: x[0], world_states))
        with media.VideoWriter(fname, shape=first_frame.shape[:2], fps=self.config["FPS"], crf=18) as video:
            video.add_image(first_frame)
            for i in tqdm.trange(1, len(world_states.t)):
                frame = self.render_frame(jax.tree_map(lambda x: x[i], world_states))
                # frame = jax.jit(self.render_frame_idx)(world_states, i)
                video.add_image(frame)
        if show:
            media.show_video(media.read_video(fname))
