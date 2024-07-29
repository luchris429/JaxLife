import jax
import jax.numpy as jnp

from robot_brains.complex_program_brain import RobotComplexProgramBrain
from robot_brains.dot_product_brain import RobotDotProductBrain
from robot_brains.random_brain import RobotRandomBrain
from utils import tree_where
from world.structs import (
    AgentState,
    TerrainState,
    AgentObservation,
    BotState,
    BotObservation,
)
from agent_brains.lstm_brain import LSTMBrain
from agent_brains.zero_brain import ZeroBrain
from agent_brains.random_brain import RandomBrain
from agent_brains.debug_brain import DebugBrain
from agent_brains.profiling_brain import ProfilingBrain


class Bot:
    def __init__(self, config):
        self.config = config
        if self.config["BOT_BRAIN"] == "random_brain":
            self.brain = RobotRandomBrain(config)
        elif self.config["BOT_BRAIN"] == "dot_product_brain":
            self.brain = RobotDotProductBrain(config)
        elif self.config["BOT_BRAIN"] == "complex_program_brain":
            self.brain = RobotComplexProgramBrain(config)
        else:
            raise Exception(f"Unknown brain: {self.config['BOT_BRAIN']}")
        self.dummy_input = BotObservation.dummy_obs(self.config)

    def init_randoms(self, rng):
        rng_x, rng_y = jax.random.split(rng)
        return BotState(
            alive=True,
            energy=self.config["BOT_INITIAL_ENERGY"],
            memory=jnp.zeros(self.config["BOT_MEMORY_SIZE"]),
            program=jnp.zeros(self.config["BOT_PROGRAM_SIZE"]),
            pos_x=jax.random.uniform(
                rng_x,
                (),
                minval=0,
                maxval=self.config["MAP_SIZE"] * self.config["CELL_SIZE"],
            ),
            pos_y=jax.random.uniform(
                rng_y,
                (),
                minval=0,
                maxval=self.config["MAP_SIZE"] * self.config["CELL_SIZE"],
            ),
        )

    def act(self, rng, own_state, own_idx, world_state):
        obs, infos = self.collect_obs(own_state, own_idx, world_state)

        rng, _rng = jax.random.split(rng)
        action = self.brain(_rng, obs)

        grid_x, grid_y = (own_state.pos_x // self.config["CELL_SIZE"]).astype(jnp.int32), (
            own_state.pos_y // self.config["CELL_SIZE"]
        ).astype(jnp.int32)
        terrain = jax.tree_map(lambda x: x[grid_x, grid_y], world_state.terrain)
        world_state_diff_zero = jax.tree_map(lambda x: jnp.zeros_like(x), world_state)

        agent_states_diff = jax.tree_map(lambda x: jnp.zeros_like(x), world_state.agent_states)

        bot_states_diff = jax.tree_map(lambda x: jnp.zeros_like(x), world_state.bot_states)

        # DIFF SELF
        diff_bots_pos_x = bot_states_diff.pos_x.at[own_idx].add(self.config["MOVE_SPEED"] * action.x_move)
        diff_bots_pos_y = bot_states_diff.pos_y.at[own_idx].add(self.config["MOVE_SPEED"] * action.y_move)
        diff_bots_memory = bot_states_diff.memory.at[own_idx].add(
            action.self_message
        )  # NOTE: ADDING TO OWN MESSAGE -- not replacing

        # DIFF OTHER BOTS
        norms_bot = jnp.linalg.norm(infos["bot_diffs"][infos["bot_idxs"]], axis=-1, keepdims=True) + 0.1
        dirs_bot = infos["bot_diffs"][infos["bot_idxs"]] / norms_bot

        diff_bots_pos_x = diff_bots_pos_x.at[infos["bot_idxs"]].add(
            self.config["PUSH_SPEED"] * action.push * dirs_bot[:, 0]
        )
        diff_bots_pos_y = diff_bots_pos_y.at[infos["bot_idxs"]].add(
            self.config["PUSH_SPEED"] * action.push * dirs_bot[:, 1]
        )
        diff_bots_program = bot_states_diff.program.at[infos["bot_idxs"]].add(
            (-world_state.bot_states.program[infos["bot_idxs"]] + action.other_message) * action.message_other
        )

        # Hitting another agent makes them lose energy, and you gain energy; or if the hit action is negative, you give energy to another agent.
        hit = action.hit
        hit_benefit = hit * self.config["HIT_STRENGTH"] * jnp.exp(-norms_bot * self.config["HIT_DISTANCE_DECAY"])
        diff_bots_energy = bot_states_diff.energy.at[infos["bot_idxs"]].add(-hit_benefit[:, 0])
        diff_bots_energy = diff_bots_energy.at[own_idx].add(hit_benefit.sum() * self.config["HIT_STEAL_FRACTION"])

        # DIFF AGENTS
        norms_agent = jnp.linalg.norm(infos["agent_diffs"][infos["agent_idxs"]], axis=-1, keepdims=True) + 0.1
        dirs_agent = infos["agent_diffs"][infos["agent_idxs"]] / norms_agent

        diff_agents_pos_x = agent_states_diff.pos_x.at[infos["agent_idxs"]].add(
            self.config["PUSH_SPEED"] * action.push * dirs_agent[:, 0]
        )
        diff_agents_pos_y = agent_states_diff.pos_y.at[infos["agent_idxs"]].add(
            self.config["PUSH_SPEED"] * action.push * dirs_agent[:, 1]
        )
        diff_agents_other_message = agent_states_diff.other_message.at[infos["agent_idxs"]].add(
            (-world_state.agent_states.other_message[infos["agent_idxs"]] + action.other_message) * action.message_other
        )
        hit_benefit = hit * self.config["HIT_STRENGTH"] * jnp.exp(-norms_agent * self.config["HIT_DISTANCE_DECAY"])
        diff_agents_energy = agent_states_diff.energy.at[infos["agent_idxs"]].add(-hit_benefit[:, 0])
        diff_bots_energy = diff_bots_energy.at[own_idx].add(hit_benefit.sum() * self.config["HIT_STEAL_FRACTION"])

        terrain_diff = jax.tree_map(lambda x: jnp.zeros_like(x), world_state.terrain)

        energy_gain_diff = terrain_diff.energy_gain.at[grid_x, grid_y].add(
            self.config["ACT_TERRAIN_ENERGY_GAIN"] * action.terrain_energy_gain
        )
        move_cost_diff = terrain_diff.move_cost.at[grid_x, grid_y].add(
            self.config["ACT_TERRAIN_MOVE_COST"] * action.terrain_move_cost
        )
        push_cost_diff = terrain_diff.push_cost.at[grid_x, grid_y].add(
            self.config["ACT_TERRAIN_PUSH_COST"] * action.terrain_push_cost
        )
        message_cost_diff = terrain_diff.message_cost.at[grid_x, grid_y].add(
            self.config["ACT_TERRAIN_MESSSAGE_COST"] * action.terrain_message_cost
        )
        reproduce_cost = terrain_diff.reproduce_cost.at[grid_x, grid_y].add(
            self.config["ACT_TERRAIN_REPRODUCE_COST"] * action.terrain_reproduce_cost
        )
        energy_amt_diff = terrain_diff.energy_amt.at[grid_x, grid_y].add(
            -self.config["EAT_RATE"] * terrain.energy_amt * action.eat
        )

        # New pos
        new_pos_x = own_state.pos_x + self.config["MOVE_SPEED"] * action.x_move
        new_pos_y = own_state.pos_y + self.config["MOVE_SPEED"] * action.y_move

        new_grid_x, new_grid_y = (new_pos_x // self.config["CELL_SIZE"]).astype(jnp.int32), (
            new_pos_y // self.config["CELL_SIZE"]
        ).astype(jnp.int32)

        should_write = action.read_terrain_bit < -0.99
        bits_diff = terrain_diff.bits.at[grid_x, grid_y].add(
            should_write
            * (-world_state.terrain.bits[grid_x, grid_y] + own_state.memory[-1])  # write memory bit to the terrain
        )

        should_read = action.read_terrain_bit > +0.99

        diff_bots_memory = diff_bots_memory.at[own_idx, -1].add(
            should_read
            * (-world_state.bot_states.memory[own_idx, -1] + world_state.terrain.bits[new_grid_x, new_grid_y])
        )

        hit = action.hit
        energy_cost = (
            self.config["MOVE_COST"] * terrain.move_cost * jnp.abs(action.x_move)
            + self.config["MOVE_COST"] * terrain.move_cost * jnp.abs(action.y_move)
            + self.config["PUSH_COST"] * terrain.push_cost * jnp.abs(action.push)
            + self.config["MESSAGE_COST"] * terrain.message_cost * action.message_other
            # + -self.config["EAT_RATE"] * terrain.energy_amt * action.eat
            + self.config["HIT_COST"] * hit
            + self.config["TERRAIN_MOD_COST"]
            * (
                action.terrain_energy_gain
                + action.terrain_move_cost
                + action.terrain_push_cost
                + action.terrain_message_cost
                + action.terrain_reproduce_cost
            )
            + self.config["LIFE_COST"]
        )
        eat_gain = self.config["EAT_RATE"] * terrain.energy_amt * action.eat

        diff_bots_energy = diff_bots_energy.at[own_idx].add(-energy_cost + eat_gain)

        agent_states_diff = agent_states_diff.replace(
            pos_x=diff_agents_pos_x,
            pos_y=diff_agents_pos_y,
            other_message=diff_agents_other_message,
            energy=diff_agents_energy,
        )

        bot_states_diff = bot_states_diff.replace(
            pos_x=diff_bots_pos_x,
            pos_y=diff_bots_pos_y,
            memory=diff_bots_memory,
            program=diff_bots_program,
            energy=diff_bots_energy,
        )

        terrain_diff = terrain_diff.replace(
            energy_amt=energy_amt_diff,
            energy_gain=energy_gain_diff,
            move_cost=move_cost_diff,
            push_cost=push_cost_diff,
            message_cost=message_cost_diff,
            reproduce_cost=reproduce_cost,
            bits=bits_diff,
        )

        world_state_diff_alive = world_state_diff_zero.replace(
            agent_states=agent_states_diff,
            terrain=terrain_diff,
            bot_states=bot_states_diff,
        )
        world_state_diff = tree_where(own_state.alive, world_state_diff_alive, world_state_diff_zero)

        return world_state_diff, action, energy_cost

    def collect_obs(self, own_state, own_idx, world_state):
        agent_diffs_x = jnp.abs(world_state.agent_states.pos_x - own_state.pos_x)
        agent_diffs_y = jnp.abs(world_state.agent_states.pos_y - own_state.pos_y)
        agent_diffs = jnp.stack([agent_diffs_x, agent_diffs_y], axis=-1)
        agent_diffs = jnp.minimum(
            agent_diffs,
            self.config["MAP_SIZE"] * self.config["CELL_SIZE"] - agent_diffs,
        )

        agent_dists = jnp.linalg.norm(agent_diffs, axis=-1) + (1.0 - world_state.agent_states.alive) * 1000000.0
        agent_dists_idx_sorted = jnp.argsort(agent_dists, axis=-1)
        agent_dists_idx_sorted = agent_dists_idx_sorted[: self.config["BOT_NUM_VIEW_AGENTS"]]
        visible_agents = jax.tree_map(lambda x: x[agent_dists_idx_sorted], world_state.agent_states)
        visible_agent_diffs = agent_diffs[agent_dists_idx_sorted]
        visible_agents = visible_agents.replace(pos_x=visible_agent_diffs[:, 0], pos_y=visible_agent_diffs[:, 1])

        bot_diffs_x = jnp.abs(world_state.bot_states.pos_x - own_state.pos_x)
        bot_diffs_y = jnp.abs(world_state.bot_states.pos_y - own_state.pos_y)
        bot_diffs = jnp.stack([bot_diffs_x, bot_diffs_y], axis=-1)
        bot_diffs = jnp.minimum(bot_diffs, self.config["MAP_SIZE"] * self.config["CELL_SIZE"] - bot_diffs)

        bot_dists = jnp.linalg.norm(bot_diffs, axis=-1) + (1.0 - world_state.bot_states.alive) * 1.0

        # Set my own index's distance to a large number so I don't see myself
        bot_dists = bot_dists.at[own_idx].set(1000 * self.config["MAP_SIZE"] ** 1)

        bot_dists_idx_sorted = jnp.argsort(bot_dists, axis=-1)
        bot_dists_idx_sorted = bot_dists_idx_sorted[: self.config["BOT_NUM_VIEW_BOTS"]]
        visible_bots = jax.tree_map(lambda x: x[bot_dists_idx_sorted], world_state.bot_states)
        visible_bot_diffs = bot_diffs[bot_dists_idx_sorted]
        visible_bots = visible_bots.replace(pos_x=visible_bot_diffs[:, 0], pos_y=visible_bot_diffs[:, 1])

        max_dist = 2 * self.config["MAP_SIZE"] * self.config["CELL_SIZE"]
        # jax.debug.print("MAX {}|{}", world_state.agent_states.pos_y.max(), world_state.bot_states.pos_y.max())
        all_dists = jnp.concatenate(
            [
                agent_dists[agent_dists_idx_sorted] * max_dist + world_state.agent_states.pos_y[agent_dists_idx_sorted],
                bot_dists[bot_dists_idx_sorted] * max_dist + world_state.bot_states.pos_y[bot_dists_idx_sorted],
            ],
            axis=0,
        )
        all_dists_idx_sorted = jnp.argsort(all_dists, axis=-1)
        all_dists_idx_sorted = all_dists_idx_sorted[:2]
        all_messages = jnp.concatenate([visible_agents.self_message, visible_bots.memory], axis=0)
        other_messages = all_messages[all_dists_idx_sorted]
        idxs_actually = jnp.concatenate(
            [
                jnp.arange(self.config["NUM_AGENTS"])[agent_dists_idx_sorted],
                jnp.arange(self.config["NUM_BOTS"])[bot_dists_idx_sorted],
            ]
        )[all_dists_idx_sorted]
        # jax.debug.print("Collect OBS {}|{}=>{}||{}++{}=> D={}|{}++{}", own_idx, all_dists_idx_sorted, len(agent_dists), other_messages[:, -1],  world_state.agent_states.pos_y[agent_dists_idx_sorted[:2]], all_dists[all_dists_idx_sorted], world_state.agent_states.alive.sum(), idxs_actually)
        # jax.debug.breakpoint()

        return BotObservation(
            other_messages=other_messages,
            self_message=own_state.memory,
            program=own_state.program,
        ), {
            "agent_idxs": agent_dists_idx_sorted,
            "bot_idxs": bot_dists_idx_sorted,
            "agent_diffs": agent_diffs,
            "bot_diffs": bot_diffs,
        }
