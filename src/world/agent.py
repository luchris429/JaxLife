import jax
import jax.numpy as jnp
from utils import tree_where
from world.structs import AgentState, TerrainState, AgentObservation, BotState, Action
from agent_brains.lstm_brain import LSTMBrain
from agent_brains.zero_brain import ZeroBrain
from agent_brains.random_brain import RandomBrain
from agent_brains.debug_brain import DebugBrain
from agent_brains.profiling_brain import ProfilingBrain


class Agent:
    def __init__(self, config):
        self.config = config
        if self.config["BRAIN"] == "lstm_brain":
            self.network = LSTMBrain(config)
        elif self.config["BRAIN"] == "zero_brain":
            self.network = ZeroBrain(config)
        elif self.config["BRAIN"] == "random_brain":
            self.network = RandomBrain(config)
        elif self.config["BRAIN"] == "debug_brain":
            self.network = DebugBrain(config)
        elif self.config["BRAIN"] == "profiling_brain":
            self.network = ProfilingBrain(config)
        else:
            raise Exception(f"Unknown brain: {self.config['BRAIN']}")
        self.dummy_input = AgentObservation.dummy_obs(self.config)

    def init_randoms(self, rng):
        rng_id, rng_x, rng_y, rng_genome, rng_hidden = jax.random.split(rng, 5)
        hidden_state = self.network.initialize_carry(rng_hidden, (self.config["HSIZE"],))
        genome_params = self.network.init(rng_genome, hidden_state, self.dummy_input)
        return AgentState(
            alive=True,
            energy=self.config["INITIAL_ENERGY"],
            age=0.0,
            id=jax.random.uniform(rng_id, (self.config["ID_SIZE"],), minval=-1.0, maxval=1.0),
            # id=jnp.zeros(self.config["ID_SIZE"]),
            self_message=jnp.zeros(self.config["MESSAGE_SIZE"]),
            other_message=jnp.zeros(self.config["MESSAGE_SIZE"]),
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
            last_action=jnp.zeros(self.config["ACTION_REP_SIZE"]),
            genome_params=genome_params,
            hidden_state_params=hidden_state,
        )

    def mutate(self, rng, parent_state):
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        mutated_genome_params = jax.tree_map(
            lambda x: x + jax.random.normal(rng, x.shape) * self.config["MUTATION_STD"],
            parent_state.genome_params,
        )
        hidden_state = self.network.initialize_carry(_rng, (self.config["HSIZE"],))
        mutated_id = parent_state.id + jax.random.uniform(_rng2, (self.config["ID_SIZE"],), minval=-1.0, maxval=1.0)
        mutated_id = jnp.clip(mutated_id, -1.0, 1.0)
        return AgentState(
            alive=True,
            energy=self.config["INITIAL_ENERGY"],
            age=0.0,
            id=mutated_id,
            self_message=jnp.zeros_like(parent_state.self_message),
            other_message=jnp.zeros_like(parent_state.other_message),
            pos_x=parent_state.pos_x,
            pos_y=parent_state.pos_y,
            last_action=jnp.zeros_like(parent_state.last_action),
            genome_params=mutated_genome_params,
            hidden_state_params=hidden_state,
        )

    def reproduce(self, rng, parent_idxs, world_state):
        rng, rng_child, rng_random = jax.random.split(rng, 3)
        num_parents = parent_idxs.shape[0]
        rng_childs = jax.random.split(rng_child, num_parents)
        rng_randoms = jax.random.split(rng_random, num_parents)
        # parent_states = world_state.agent_states[parent_idxs]
        parent_states = jax.tree_map(lambda x: x[parent_idxs], world_state.agent_states)  # [parent_idxs]
        child_states = jax.vmap(self.mutate)(rng_childs, parent_states)
        random_states = jax.vmap(self.init_randoms)(rng_randoms)
        child_states = tree_where(parent_idxs != -1, child_states, random_states)
        return child_states

    def calc_saliency(self, own_state, own_idx, world_state):
        obs, infos = self.collect_obs(own_state, own_idx, world_state)

        def obs_loss(in_obs):
            action, hidden_state_params = self.network.apply(
                own_state.genome_params, own_state.hidden_state_params, in_obs
            )
            return Action.to_repr(action).mean()

        saliency = jax.grad(obs_loss)(obs)
        saliency = jax.tree_map(lambda x: jnp.abs(x), saliency)
        grid_saliency, agent_saliency, bot_saliency, self_saliency = (
            saliency.grid,
            saliency.agents,
            saliency.bots,
            saliency.self,
        )
        saliency_metrics = {}
        saliency_metrics["grid_saliency"] = grid_saliency.mean()
        saliency_metrics["agent_saliency"] = agent_saliency.mean()
        saliency_metrics["bot_saliency"] = bot_saliency.mean()
        saliency_metrics["self_saliency"] = self_saliency.mean()
        agent_comm_saliency = agent_saliency[:, -self.config["MESSAGE_SIZE"] :]
        saliency_metrics["agent_comm_saliency"] = agent_comm_saliency.mean()
        agent_comm_saliency = agent_saliency[:, -self.config["MESSAGE_SIZE"] * 2 :]
        saliency_metrics["agent_comm_2_saliency"] = agent_comm_saliency.mean()
        self_comm_saliency = self_saliency[-self.config["MESSAGE_SIZE"] :]
        saliency_metrics["self_comm_saliency"] = self_comm_saliency.mean()
        return saliency_metrics

    def act(self, own_state, own_idx, world_state):
        obs, infos = self.collect_obs(own_state, own_idx, world_state)
        action, hidden_state_params = self.network.apply(own_state.genome_params, own_state.hidden_state_params, obs)

        grid_x, grid_y = (own_state.pos_x // self.config["CELL_SIZE"]).astype(jnp.int32), (
            own_state.pos_y // self.config["CELL_SIZE"]
        ).astype(jnp.int32)
        terrain = jax.tree_map(lambda x: x[grid_x, grid_y], world_state.terrain)
        world_state_diff_zero = jax.tree_map(lambda x: jnp.zeros_like(x), world_state)
        # energy_cost = self.config["LIFE_COST"] * own_state.age / self.config["AGE_COST"]
        # energy_cost = own_state.age / self.config["AGE_COST"] + self.config["LIFE_COST"]

        agent_states_diff = jax.tree_map(lambda x: jnp.zeros_like(x), world_state.agent_states)
        diff_last_action = agent_states_diff.last_action.at[own_idx].add(
            -world_state.agent_states.last_action[own_idx] + action.to_repr()
        )

        bot_states_diff = jax.tree_map(lambda x: jnp.zeros_like(x), world_state.bot_states)
        # DIFF SELF
        diff_age = agent_states_diff.age.at[own_idx].add(self.config["AGE_SPEED"])
        diff_pos_x = agent_states_diff.pos_x.at[own_idx].add(self.config["MOVE_SPEED"] * action.x_move)
        diff_pos_y = agent_states_diff.pos_y.at[own_idx].add(self.config["MOVE_SPEED"] * action.y_move)
        diff_self_message = agent_states_diff.self_message.at[own_idx].add(
            action.self_message
        )  # NOTE: ADDING TO OWN MESSAGE -- not replacing

        # DIFF OTHER AGENTS
        norms_agent = jnp.linalg.norm(infos["agent_diffs"][infos["agent_idxs"]], axis=-1, keepdims=True) + 0.1
        dirs_agent = infos["agent_diffs"][infos["agent_idxs"]] / norms_agent

        diff_pos_x = diff_pos_x.at[infos["agent_idxs"]].add(self.config["PUSH_SPEED"] * action.push * dirs_agent[:, 0])
        diff_pos_y = diff_pos_y.at[infos["agent_idxs"]].add(self.config["PUSH_SPEED"] * action.push * dirs_agent[:, 1])
        diff_other_message = agent_states_diff.other_message.at[infos["agent_idxs"]].add(
            (-world_state.agent_states.other_message[infos["agent_idxs"]] + action.other_message) * action.message_other
        )
        hit = action.hit
        hit_benefit = hit * self.config["HIT_STRENGTH"] * jnp.exp(-norms_agent * self.config["HIT_DISTANCE_DECAY"])
        diff_agents_energy = agent_states_diff.energy.at[infos["agent_idxs"]].add(-hit_benefit[:, 0])
        diff_agents_energy = diff_agents_energy.at[own_idx].add(hit_benefit.sum() * self.config["HIT_STEAL_FRACTION"])

        # DIFF BOTS
        norms_bot = jnp.linalg.norm(infos["bot_diffs"][infos["bot_idxs"]], axis=-1, keepdims=True) + 0.1
        dirs_bot = infos["bot_diffs"][infos["bot_idxs"]] / norms_bot

        diff_bots_pos_x = bot_states_diff.pos_x.at[infos["bot_idxs"]].add(
            self.config["PUSH_SPEED"] * action.push * dirs_bot[:, 0]
        )
        diff_bots_pos_y = bot_states_diff.pos_y.at[infos["bot_idxs"]].add(
            self.config["PUSH_SPEED"] * action.push * dirs_bot[:, 1]
        )
        diff_bots_program = bot_states_diff.program.at[infos["bot_idxs"]].add(
            (-world_state.bot_states.program[infos["bot_idxs"]] + action.other_message) * action.message_other
        )

        # Hitting another agent makes them lose energy, and you gain energy; or if the hit action is negative, you give energy to another agent.
        hit = action.hit
        hit_benefit = hit * self.config["HIT_STRENGTH"] * jnp.exp(-norms_bot * self.config["HIT_DISTANCE_DECAY"])
        diff_bots_energy = bot_states_diff.energy.at[infos["bot_idxs"]].add(-hit_benefit[:, 0])
        diff_agents_energy = diff_agents_energy.at[own_idx].add(hit_benefit.sum() * self.config["HIT_STEAL_FRACTION"])

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

        agent_frac = jnp.sum(world_state.agent_states.alive) / self.config["NUM_AGENTS"]
        reproduce = (
            action.reproduce
            * (
                own_state.energy
                > self.config["MAX_REPRODUCE_COST"]
                * terrain.reproduce_cost
                * self.config["REPRODUCE_FACTOR"]
                * agent_frac
            )
            * own_state.alive
        )
        life_cost = own_state.age / self.config["AGE_COST"] + self.config["LIFE_COST"]
        energy_cost = (
            self.config["MOVE_COST"] * terrain.move_cost * jnp.abs(action.x_move)
            + self.config["MOVE_COST"] * terrain.move_cost * jnp.abs(action.y_move)
            + self.config["PUSH_COST"] * terrain.push_cost * jnp.abs(action.push)
            + self.config["MESSAGE_COST"] * terrain.message_cost * action.message_other
            + self.config["MAX_REPRODUCE_COST"] * agent_frac * terrain.reproduce_cost * reproduce
            + self.config["HIT_COST"] * hit
            + self.config["TERRAIN_MOD_COST"]
            * (
                action.terrain_energy_gain
                + action.terrain_move_cost
                + action.terrain_push_cost
                + action.terrain_message_cost
                + action.terrain_reproduce_cost
            )
            + life_cost
        )
        eat_gain = self.config["EAT_RATE"] * terrain.energy_amt * action.eat

        diff_agents_energy = diff_agents_energy.at[own_idx].add(-energy_cost + eat_gain)

        agent_states_diff = agent_states_diff.replace(
            age=diff_age,
            pos_x=diff_pos_x,
            pos_y=diff_pos_y,
            self_message=diff_self_message,
            other_message=diff_other_message,
            energy=diff_agents_energy,
            last_action=diff_last_action,
        )

        bot_states_diff = bot_states_diff.replace(
            pos_x=diff_bots_pos_x,
            pos_y=diff_bots_pos_y,
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
        )

        world_state_diff_alive = world_state_diff_zero.replace(
            agent_states=agent_states_diff,
            terrain=terrain_diff,
            bot_states=bot_states_diff,
        )
        world_state_diff = tree_where(own_state.alive, world_state_diff_alive, world_state_diff_zero)
        energy_cost = jnp.where(
            own_state.alive,
            energy_cost,
            jnp.zeros_like(energy_cost),
        )
        return world_state_diff, hidden_state_params, reproduce, action, energy_cost

    def collect_obs(self, own_state, own_idx, world_state):
        grid = world_state.terrain

        view_range = self.config["TERRAIN_VIEW_RANGE"]  # CHECK IF THIS IS RIGHT...
        xrange = jnp.arange(-view_range, view_range + 1) + (own_state.pos_x // self.config["CELL_SIZE"]).astype(
            jnp.int32
        )
        yrange = jnp.arange(-view_range, view_range + 1) + (own_state.pos_y // self.config["CELL_SIZE"]).astype(
            jnp.int32
        )
        x_grid, y_grid = jnp.meshgrid(xrange, yrange)
        grid_obs = jax.tree_map(lambda x: x[x_grid, y_grid], grid)
        grid_obs = jax.vmap(TerrainState.to_obs)(grid_obs)
        if self.config["FLATTEN_GRID_OBS"]:
            grid_obs = jnp.reshape(grid_obs, (-1,))

        agent_diffs_x = jnp.abs(world_state.agent_states.pos_x - own_state.pos_x)
        agent_diffs_y = jnp.abs(world_state.agent_states.pos_y - own_state.pos_y)
        agent_diffs = jnp.stack([agent_diffs_x, agent_diffs_y], axis=-1)
        agent_diffs = jnp.minimum(
            agent_diffs,
            self.config["MAP_SIZE"] * self.config["CELL_SIZE"] - agent_diffs,
        )

        agent_dists = (
            jnp.linalg.norm(agent_diffs, axis=-1)
            + (1.0 - world_state.agent_states.alive) * self.config["MAP_SIZE"] * 10
        )
        # Set my own index's distance to a large number so I don't see myself
        agent_dists = agent_dists.at[own_idx].set(1000 * self.config["MAP_SIZE"] ** 2)

        agent_dists_idx_sorted = jnp.argsort(agent_dists, axis=-1)
        agent_dists_idx_sorted = agent_dists_idx_sorted[: self.config["AGENT_NUM_VIEW_AGENTS"]]
        visible_agents = jax.tree_map(lambda x: x[agent_dists_idx_sorted], world_state.agent_states)
        visible_agent_diffs = agent_diffs[agent_dists_idx_sorted]
        visible_agents = visible_agents.replace(pos_x=visible_agent_diffs[:, 0], pos_y=visible_agent_diffs[:, 1])
        agent_obs = jax.vmap(AgentState.to_obs)(visible_agents)

        bot_diffs_x = jnp.abs(world_state.bot_states.pos_x - own_state.pos_x)
        bot_diffs_y = jnp.abs(world_state.bot_states.pos_y - own_state.pos_y)
        bot_diffs = jnp.stack([bot_diffs_x, bot_diffs_y], axis=-1)
        bot_diffs = jnp.minimum(bot_diffs, self.config["MAP_SIZE"] * self.config["CELL_SIZE"] - bot_diffs)

        bot_dists = jnp.linalg.norm(bot_diffs, axis=-1) + (1.0 - world_state.bot_states.alive) * 1.0
        bot_dists_idx_sorted = jnp.argsort(bot_dists, axis=-1)
        bot_dists_idx_sorted = bot_dists_idx_sorted[: self.config["AGENT_NUM_VIEW_BOTS"]]
        visible_bots = jax.tree_map(lambda x: x[bot_dists_idx_sorted], world_state.bot_states)
        visible_bot_diffs = bot_diffs[bot_dists_idx_sorted]
        visible_bots = visible_bots.replace(pos_x=visible_bot_diffs[:, 0], pos_y=visible_bot_diffs[:, 1])
        bot_obs = jax.vmap(BotState.to_obs)(visible_bots)

        return AgentObservation(
            grid=grid_obs,
            agents=agent_obs,
            bots=bot_obs,
            self=AgentState.to_obs(own_state),
        ), {
            "agent_idxs": agent_dists_idx_sorted,
            "agent_diffs": agent_diffs,
            "bot_idxs": bot_dists_idx_sorted,
            "bot_diffs": bot_diffs,
        }
