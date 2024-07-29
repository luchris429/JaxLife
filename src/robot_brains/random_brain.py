import jax

from world.structs import Action


class RobotRandomBrain:
    def __init__(self, config):
        self.config = config

    def __call__(self, rng, inputs):
        (
            rng,
            rng_xmove,
            rng_ymove,
            rng_push,
            rng_eat,
            rng_reproduce,
            rng_message_other,
            rng_terrain,
        ) = jax.random.split(rng, 8)
        rng, rng_self_message, rng_other_message = jax.random.split(rng, 3)
        (
            rng,
            rng_energy_gain,
            rng_move_cost,
            rng_push_cost,
            rng_message_cost,
            rng_reproduce_cost,
        ) = jax.random.split(rng, 6)
        rand_action = Action(
            x_move=jax.random.uniform(rng_xmove, minval=-1.0, maxval=1.0),
            y_move=jax.random.uniform(rng_ymove, minval=-1.0, maxval=1.0),
            push=jax.random.uniform(rng_push, minval=-1.0, maxval=1.0),
            hit=jax.random.uniform(rng_push, minval=-1.0, maxval=1.0),
            read_terrain_bit=0.0,
            eat=jax.random.uniform(rng_eat, minval=0.0, maxval=1.0),
            reproduce=jax.random.uniform(rng_reproduce, minval=0.0, maxval=1.0),
            message_other=jax.random.uniform(rng_message_other, shape=(), minval=0.0, maxval=1.0),
            self_message=jax.random.uniform(
                rng_self_message,
                shape=(self.config["MESSAGE_SIZE"],),
                minval=0.0,
                maxval=1.0,
            ),
            other_message=jax.random.uniform(
                rng_other_message,
                shape=(self.config["MESSAGE_SIZE"],),
                minval=0.0,
                maxval=1.0,
            ),
            terrain_energy_gain=jax.random.uniform(rng_energy_gain, minval=0.0, maxval=1.0),
            terrain_move_cost=jax.random.uniform(rng_move_cost, minval=0.0, maxval=1.0),
            terrain_push_cost=jax.random.uniform(rng_push_cost, minval=0.0, maxval=1.0),
            terrain_message_cost=jax.random.uniform(rng_message_cost, minval=0.0, maxval=1.0),
            terrain_reproduce_cost=jax.random.uniform(rng_reproduce_cost, minval=0.0, maxval=1.0),
        )

        return rand_action
