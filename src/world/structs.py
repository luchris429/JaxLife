import jax
import jax.numpy as jnp
from flax import struct
from typing import Any


@struct.dataclass
class AgentObservation:
    grid: jnp.ndarray
    agents: jnp.ndarray
    bots: jnp.ndarray
    self: jnp.ndarray

    @classmethod
    def dummy_obs(cls, config):
        terrain_obs_size = 7
        agent_obs_size = 4 + config["ID_SIZE"] + 2 * config["MESSAGE_SIZE"] + config["ACTION_REP_SIZE"]
        grid_size = config["TERRAIN_VIEW_RANGE"] * 2 + 1
        return cls(
            grid=jnp.zeros((grid_size * grid_size * terrain_obs_size)),
            agents=jnp.zeros((config["AGENT_NUM_VIEW_AGENTS"], agent_obs_size)),
            bots=jnp.zeros(
                (config["AGENT_NUM_VIEW_BOTS"], 3 + 2 * config["MESSAGE_SIZE"])
            ),
            self=jnp.zeros((agent_obs_size)),
        )


@struct.dataclass
class BotObservation:
    other_messages: jnp.ndarray
    self_message: jnp.ndarray
    program: jnp.ndarray

    @classmethod
    def dummy_obs(cls, config):
        return cls(
            other_messages=jnp.zeros((config["BOT_NUM_VIEW_AGENTS"], config["MESSAGE_SIZE"])),
            self_message=jnp.zeros(config["MESSAGE_SIZE"]),
            program=jnp.zeros(config["MESSAGE_SIZE"]),
        )


@struct.dataclass
class Action:
    x_move: float
    y_move: float
    push: float  # Negative means pull
    eat: float
    hit: float
    reproduce: float
    message_other: float
    self_message: jnp.ndarray
    other_message: jnp.ndarray
    terrain_energy_gain: float
    terrain_move_cost: float
    terrain_push_cost: float
    terrain_message_cost: float
    terrain_reproduce_cost: float
    read_terrain_bit: float  # +1 means read; -1 means write.

    # @classmethod
    def to_repr(self):
        return jnp.array(
            [
                self.x_move,
                self.y_move,
                self.push,
                self.eat,
                self.hit,
                self.reproduce,
                self.message_other,
                self.terrain_energy_gain,
                self.terrain_move_cost,
                self.terrain_push_cost,
                # self.terrain_message_cost,
                self.terrain_reproduce_cost,
            ]
        )


@struct.dataclass
class TerrainState:
    # x: int
    # y: int
    energy_amt: float
    move_cost: float
    push_cost: float
    energy_gain: float
    message_cost: float
    reproduce_cost: float
    max_energy: float

    bits: float

    @classmethod
    def to_obs(cls, self):
        return jnp.concatenate(
            [
                self.energy_amt,
                self.move_cost,
                self.push_cost,
                self.energy_gain,
                self.message_cost,
                self.reproduce_cost,
                self.max_energy,
            ]
        )


@struct.dataclass
class AgentState:
    alive: bool
    energy: float
    age: float
    id: jnp.ndarray
    self_message: jnp.ndarray
    other_message: jnp.ndarray
    pos_x: float
    pos_y: float
    last_action: jnp.ndarray
    genome_params: Any
    hidden_state_params: Any

    @classmethod
    def to_obs(cls, self):
        return jnp.concatenate(
            [
                jnp.log(self.energy + 1e-4)[None,],
                jnp.log(self.age + 1e-4)[None,],
                self.pos_x[None,],
                self.pos_y[None,],
                self.last_action,
                self.id,
                self.self_message,
                self.other_message,
            ]
        )

    @classmethod
    def clone_no_params(cls, self):
        return self.replace(genome_params=None, hidden_state_params=None)

    @classmethod
    def create(cls, p_state):
        return cls(
            alive=True,
            energy=p_state.energy,
            age=0.0,
            id=p_state.id,
            self_message=p_state.self_message,
            other_message=jnp.zeros_like(p_state.self_message),
            pos_x=p_state.pos_x,
            pos_y=p_state.pos_y,
            last_action=-1,
            genome_params=None,
            hidden_state_params=None,
        )


@struct.dataclass
class BotState:
    alive: bool
    energy: float
    memory: jnp.ndarray
    program: jnp.ndarray
    pos_x: float
    pos_y: float

    @classmethod
    def to_obs(cls, self):
        return jnp.concatenate(
            [
                jnp.log(jax.nn.relu(self.energy) + 1e-4)[None,],
                self.pos_x[None,],
                self.pos_y[None,],
                self.memory,
                self.program,
            ]
        )


@struct.dataclass
class WorldState:
    t: int
    agent_states: AgentState
    bot_states: BotState
    terrain: TerrainState
    base_terrain: TerrainState
    noise_angles: TerrainState
    rng: jnp.ndarray

    @classmethod
    def clone_no_params(cls, self):
        """
        This stuff is to help deal with memory.
        TBH I'm not sure if JAX would automatically figure this out on its own w/ eliding.
        """
        return self.replace(
            agent_states=AgentState.clone_no_params(self.agent_states),
        )
