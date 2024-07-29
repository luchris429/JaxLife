import jax
import jax.numpy as jnp
import numpy as np
from utils import tree_where
from world.agent import Agent
from world.bots import Bot
from world.terrain import Terrain
from world.structs import WorldState, AgentState, TerrainState


class World:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.agent = Agent(config)
        self.bot = Bot(config)
        self.terrain = Terrain(config)

    def initialize(self, rng):
        rng, rng_agents, rng_terrain = jax.random.split(rng, 3)
        rng_agents = jax.random.split(rng_agents, self.config["NUM_AGENTS"])
        agent_states = jax.vmap(self.agent.init_randoms)(rng_agents)
        new_alive = jnp.arange(self.config["NUM_AGENTS"]) < self.config["MIN_AGENTS"]
        new_energy = jnp.where(new_alive, self.config["INITIAL_ENERGY"], 0.0)
        agent_states = agent_states.replace(alive=new_alive, energy=new_energy)

        rng, rng_bots, rng_terrain = jax.random.split(rng, 3)
        rng_bots = jax.random.split(rng_bots, self.config["NUM_BOTS"])
        bot_states = jax.vmap(self.bot.init_randoms)(rng_bots)

        rng, _rng = jax.random.split(rng)
        terrain, base_terrain, noise_angles = self.terrain.initialize(_rng)

        return WorldState(
            t=0,
            agent_states=agent_states,
            bot_states=bot_states,
            terrain=terrain,
            base_terrain=base_terrain,
            noise_angles=noise_angles,
            rng=rng,
        )

    def step(self, world_state, unused):
        metrics = {}
        metrics["main/kardashev_score"] = 0.0
        world_state = self.agent_update(world_state, metrics)
        for _ in range(self.config["BOT_STEP_RATIO"]):
            world_state = self.bot_update(world_state, metrics)
        world_state = self.terrain_update(world_state, metrics)
        world_state = world_state.replace(t=world_state.t + 1)
        metrics["num_agents"] = jnp.sum(world_state.agent_states.alive)
        metrics["agent_proportion_measure"] = (0.7 - metrics["num_agents"] / self.config["NUM_AGENTS"]) ** 2
        metrics["mean_energy"] = (
            world_state.agent_states.energy * world_state.agent_states.alive
        ).sum() / world_state.agent_states.alive.sum()
        metrics["mean_terrain_energy"] = jnp.mean(world_state.terrain.energy_amt)
        metrics["mean_terrain_move_cost"] = jnp.mean(world_state.terrain.move_cost)
        metrics["mean_terrain_push_cost"] = jnp.mean(world_state.terrain.push_cost)
        metrics["mean_terrain_energy_gain"] = jnp.mean(world_state.terrain.energy_gain)
        metrics["mean_terrain_message_cost"] = jnp.mean(world_state.terrain.message_cost)
        metrics["mean_terrain_reproduce_cost"] = jnp.mean(world_state.terrain.reproduce_cost)
        metrics["average_age"] = (
            world_state.agent_states.age * world_state.agent_states.alive
        ).sum() / world_state.agent_states.alive.sum()
        metrics["main/kardashev_score_norm"] = metrics["main/kardashev_score"] / metrics["num_agents"]
        metrics["main/total_energy_gain"] = jnp.mean(world_state.terrain.energy_gain)
        if self.config["RETURN_WORLD_STATE"]:
            metrics["world_state"] = WorldState.clone_no_params(world_state)
        return world_state, metrics

    def _clip_state(self, world_state, update_entities=True):
        terrain_state = world_state.terrain

        new_terrain_states = terrain_state.replace(
            energy_amt=terrain_state.energy_amt.clip(min=0.0, max=terrain_state.max_energy),
            move_cost=terrain_state.move_cost.clip(min=0.0),
            push_cost=terrain_state.push_cost.clip(min=0.0),
            energy_gain=terrain_state.energy_gain.clip(min=0.0, max=2.0),
            message_cost=terrain_state.message_cost.clip(min=0.0),
            reproduce_cost=terrain_state.reproduce_cost.clip(min=0.0),
        )

        if update_entities:
            agent_states = world_state.agent_states
            agent_states = agent_states.replace(
                alive=jnp.logical_and(agent_states.alive, agent_states.energy > 0),
                energy=agent_states.energy.clip(min=0.0),
                # age=agent_states.age.clip(min=0.0),
                pos_x=agent_states.pos_x % (self.config["MAP_SIZE"] * self.config["CELL_SIZE"]),
                pos_y=agent_states.pos_y % (self.config["MAP_SIZE"] * self.config["CELL_SIZE"]),
                self_message=jnp.tanh(agent_states.self_message),
                other_message=jnp.tanh(agent_states.other_message),
                genome_params=world_state.agent_states.genome_params,
            )

            bot_states = world_state.bot_states
            bot_states = bot_states.replace(
                alive=jnp.logical_or(bot_states.alive, jnp.ones_like(bot_states.alive)),
                energy=bot_states.energy.clip(min=0.0),
                pos_x=bot_states.pos_x % (self.config["MAP_SIZE"] * self.config["CELL_SIZE"]),
                pos_y=bot_states.pos_y % (self.config["MAP_SIZE"] * self.config["CELL_SIZE"]),
                program=jnp.clip(bot_states.program, -1, 1),
                memory=jnp.clip(bot_states.memory, -1, 1),
            )

            world_state = world_state.replace(agent_states=agent_states, bot_states=bot_states)

        return world_state.replace(terrain=new_terrain_states)

    def terrain_update(self, world_state, metrics):
        rng, _rng = jax.random.split(world_state.rng)
        new_terrain_states, new_base_states, new_noise_angles = self.terrain.update(world_state, _rng)

        world_state = world_state.replace(
            terrain=new_terrain_states,
            base_terrain=new_base_states,
            noise_angles=new_noise_angles,
            rng=rng,
        )

        return self._clip_state(world_state, update_entities=False)

    def bot_update(self, world_state: WorldState, metrics):
        rng = world_state.rng
        world_state_no_params = WorldState.clone_no_params(world_state)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.config["NUM_BOTS"])
        world_state_diff, all_actions, energy_cost = jax.vmap(self.bot.act, in_axes=(0, 0, 0, None))(
            rngs,
            world_state.bot_states,
            jnp.arange(self.config["NUM_BOTS"]),
            world_state_no_params,
        )
        world_state_no_params = jax.tree_map(lambda x, y: x.sum(0) + y, world_state_diff, world_state_no_params)

        # Turn alive back into bool
        world_state_no_params = world_state_no_params.replace(
            bot_states=world_state_no_params.bot_states.replace(
                alive=world_state_no_params.bot_states.alive.astype(bool)
            )
        )
        metrics["main/kardashev_bot"] = jnp.sum(energy_cost)
        metrics["main/kardashev_score"] += jnp.sum(energy_cost)
        metrics["bot_actions/push"] = jnp.mean((all_actions.push))
        metrics["bot_actions/hit"] = jnp.mean((all_actions.hit))
        metrics["bot_actions/eat"] = jnp.mean((all_actions.eat))
        metrics["bot_actions/reproduce"] = jnp.mean((all_actions.reproduce))
        metrics["bot_actions/terrain_energy_gain"] = jnp.mean(all_actions.terrain_energy_gain)
        metrics["bot_actions/terrain_move_cost"] = jnp.mean(all_actions.terrain_move_cost)
        metrics["bot_actions/terrain_push_cost"] = jnp.mean(all_actions.terrain_push_cost)
        metrics["bot_actions/terrain_message_cost"] = jnp.mean(all_actions.terrain_message_cost)
        metrics["bot_actions/terrain_reproduce_cost"] = jnp.mean(all_actions.terrain_reproduce_cost)

        # Make sure all values valid
        agent_states = world_state_no_params.agent_states
        agent_states = agent_states.replace(
            hidden_state_params=world_state.agent_states.hidden_state_params,
            genome_params=world_state.agent_states.genome_params,
        )
        # metrics["num_died"] = jnp.sum(
        #     jnp.logical_and(agent_states.energy <= 0, agent_states.alive)
        # )

        new_world_state = world_state.replace(
            terrain=world_state_no_params.terrain,
            agent_states=agent_states,
            bot_states=world_state_no_params.bot_states,
            rng=rng,
        )

        return self._clip_state(new_world_state, update_entities=True)

    def calc_saliency(self, world_state: WorldState):
        world_state_no_params = WorldState.clone_no_params(world_state)
        saliency = jax.vmap(self.agent.calc_saliency, in_axes=(0, 0, None))(
            world_state.agent_states,
            jnp.arange(self.config["NUM_AGENTS"]),
            world_state_no_params,
        )
        saliency = jax.tree_map(
            lambda x: (x * world_state.agent_states.alive).sum() / jnp.sum(world_state.agent_states.alive), saliency
        )
        return saliency

    def agent_update(self, world_state: WorldState, metrics):
        rng = world_state.rng
        world_state_no_params = WorldState.clone_no_params(world_state)
        if not self.config["IGNORE_AGENT_UPDATE"]:
            (
                world_state_diff,
                next_hidden_states,
                reproduce_acts,
                all_actions,
                energy_cost,
            ) = jax.vmap(self.agent.act, in_axes=(0, 0, None))(
                world_state.agent_states,
                jnp.arange(self.config["NUM_AGENTS"]),
                world_state_no_params,
            )
            world_state_no_params = jax.tree_map(lambda x, y: x.sum(0) + y, world_state_diff, world_state_no_params)

            metrics["main/kardashev_agent"] = jnp.sum(energy_cost)
            metrics["main/kardashev_score"] += jnp.sum(energy_cost)
            metrics["actions/push"] = jnp.mean(jnp.abs(all_actions.push))
            metrics["actions/hit"] = jnp.mean(jnp.abs(all_actions.hit))
            metrics["actions/eat"] = jnp.mean(jnp.abs(all_actions.eat))
            metrics["actions/reproduce"] = jnp.mean(jnp.abs(all_actions.reproduce))

            metrics["actions/terrain_energy_gain"] = jnp.mean(all_actions.terrain_energy_gain)
            metrics["actions/terrain_move_cost"] = jnp.mean(all_actions.terrain_move_cost)
            metrics["actions/terrain_push_cost"] = jnp.mean(all_actions.terrain_push_cost)
            metrics["actions/terrain_message_cost"] = jnp.mean(all_actions.terrain_message_cost)
            metrics["actions/terrain_reproduce_cost"] = jnp.mean(all_actions.terrain_reproduce_cost)
        else:
            reproduce_acts = jnp.zeros(self.config["NUM_AGENTS"])
            next_hidden_states = world_state.agent_states.hidden_state_params
        # Make sure all values valid
        agent_states = world_state_no_params.agent_states  # .alive.at[world_state.agent_states.energy <= 0].set(False)
        metrics["num_died"] = jnp.sum(jnp.logical_and(agent_states.energy <= 0, agent_states.alive))

        world_state_no_params = self._clip_state(
            world_state_no_params.replace(agent_states=agent_states),
            update_entities=True,
        )
        agent_states = world_state_no_params.agent_states.replace(
            hidden_state_params=next_hidden_states,
            genome_params=world_state.agent_states.genome_params,
        )

        # Sort by energy DESC
        idx_sorted = jnp.argsort(-agent_states.energy, axis=-1)
        agent_states = jax.tree_map(lambda x: x[idx_sorted], agent_states)
        print("*" * 10)
        print(f"reproduce_acts.shape: {reproduce_acts.shape}")
        reproduce_acts = reproduce_acts[idx_sorted]

        # Make in new agents
        parent_idxs = jnp.where(
            reproduce_acts,
            size=self.config["NUM_AGENTS"],
            fill_value=-1,
        )[0]
        rng, rng_reproduce = jax.random.split(rng)
        # new_agent_states = self.agent.reproduce(world_state, parent_idxs, rng_reproduce)
        new_agent_states = self.agent.reproduce(rng_reproduce, parent_idxs, world_state)
        num_new_childs = jnp.sum(parent_idxs != -1)
        num_agents = jnp.sum(agent_states.alive) + num_new_childs
        if self.config["ONLY_RESET_WHEN_NO_AGENTS"]:
            num_random = self.config["MIN_AGENTS"] * (num_agents == 0.0)
        else:
            num_random = (self.config["MIN_AGENTS"] - num_agents) * (num_agents < self.config["MIN_AGENTS"])
        num_new = num_random + num_new_childs
        metrics["num_new"] = num_new
        metrics["num_new_random"] = num_random
        metrics["num_new_childs"] = num_new_childs

        # Sort by energy ASC
        agent_states = jax.tree_map(lambda x: x[::-1], agent_states)

        agent_states = tree_where(
            jnp.logical_and(~agent_states.alive, jnp.arange(self.config["NUM_AGENTS"]) < num_new),
            new_agent_states,
            agent_states,
        )

        new_world_state = world_state.replace(
            terrain=world_state_no_params.terrain,
            agent_states=agent_states,
            bot_states=world_state_no_params.bot_states,
            rng=rng,
        )

        return new_world_state
