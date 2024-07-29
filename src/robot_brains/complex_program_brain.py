import jax
import jax.numpy as jnp
from world.structs import Action
import flax.linen as nn


class RobotComplexProgramBrain:
    def __init__(self, config):
        self.config = config

    def __call__(self, rng, inputs):
        # The program consists of several things

        NUM_OPS = 7
        LOOKUP_TABLE_SIZE = 8

        operation_to_use = jnp.argmax(inputs.program[:NUM_OPS])
        lookup_table = inputs.program[NUM_OPS : NUM_OPS + LOOKUP_TABLE_SIZE]

        mem = inputs.self_message
        m1 = inputs.other_messages[0]
        m2 = inputs.other_messages[1]
        dummy_bits = m1[-1] > 0, mem[-1] > 0, m2[-1] > 0

        tot = dummy_bits[0] * 4 + dummy_bits[1] * 2 + dummy_bits[2]

        # Operations
        # Bits or floats?
        COPY_ACTION_NAME = 0
        branches = [
            lambda mem, m1, m2: m1,  # copy action, index == 0. THIS SHOULD ALWAYS be the copy action.
            lambda mem, m1, m2: mem,
            lambda mem, m1, m2: mem * m1,
            lambda mem, m1, m2: mem * m1 + m2,
            lambda mem, m1, m2: (
                (
                    jnp.logical_xor(
                        jnp.round(mem[-1]).astype(jnp.int32),
                        jnp.round(m1[-1]).astype(jnp.int32),
                    )
                    * 2.0
                    - 1
                )
                * mem
                * m1
                + (1 - m1) * mem
            )
            .at[-1]
            .set(mem[-1]),
            lambda mem, m1, m2: 1 - (m1 * m2),  # kinda like nand
            lambda mem, m1, m2: mem.at[-1].set(
                lookup_table[jnp.round(tot).astype(jnp.int32)]
            ),  # Not sure how the dummy stuff should work.
        ]
        assert len(branches) == NUM_OPS, "Branches should be the same length as the number of operations."
        action_logits_a = jax.lax.switch(
            operation_to_use,
            branches,
            mem,
            m1,
            m2,
        )

        # jax.debug.print("Op: {}|Mem: {}|{}|: {}+{}+{}|||{} => {}", operation_to_use, action_logits_a[-1], tot, m1[-1], mem[-1], m2[-1], lookup_table, lookup_table[jnp.round(tot).astype(jnp.int32)])
        # If the copy action is selected, we should always write the self message
        should_write_self_message = jnp.logical_or(action_logits_a[13] > 0, operation_to_use == COPY_ACTION_NAME)
        # Now use prod to calculate the actions
        action = Action(
            x_move=action_logits_a[0],
            y_move=action_logits_a[1],
            push=action_logits_a[2],
            eat=action_logits_a[3] > 0,
            reproduce=action_logits_a[4] > 0,
            hit=action_logits_a[5],
            message_other=nn.relu(action_logits_a[6]),
            terrain_energy_gain=action_logits_a[7],
            terrain_move_cost=action_logits_a[8],
            terrain_push_cost=action_logits_a[9],
            terrain_message_cost=action_logits_a[10],
            terrain_reproduce_cost=action_logits_a[11],
            read_terrain_bit=action_logits_a[12],
            # If we write self message, we write the action logits
            self_message=(-inputs.self_message + action_logits_a)
            * should_write_self_message,  # self message is a delta, so we set it to zero to not write.
            # We write our own memory to the other logits
            other_message=inputs.self_message,
        )

        self_message = action.self_message
        action = jax.tree_map(lambda x: jnp.clip(x, -1, 1), action)  # clip actions.
        action = action.replace(self_message=self_message)  # Since self message is a delta we don't clip here

        return action
