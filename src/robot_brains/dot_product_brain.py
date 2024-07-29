import jax

from world.structs import Action
import flax.linen as nn


class RobotDotProductBrain:
    def __init__(self, config):
        self.config = config

    def __call__(self, rng, inputs):
        action_logits_a = inputs.other_messages[0] * inputs.program
        i = 6

        should_write_self_message = action_logits_a[13] > 0
        # Now use prod to calculate the actions
        action = Action(
            x_move=action_logits_a[0],
            y_move=action_logits_a[1],
            push=action_logits_a[2],
            eat=action_logits_a[3] > 0,
            reproduce=action_logits_a[4] > 0,
            hit=action_logits_a[5],
            message_other=nn.relu(action_logits_a[i]),
            terrain_energy_gain=action_logits_a[i + 1],
            terrain_move_cost=action_logits_a[i + 2],
            terrain_push_cost=action_logits_a[i + 3],
            terrain_message_cost=action_logits_a[i + 4],
            terrain_reproduce_cost=action_logits_a[i + 5],
            read_terrain_bit=action_logits_a[i + 6],
            # If we write self message, we write the action logits
            self_message=action_logits_a
            * should_write_self_message,  # self message is a delta, so we set it to zero to not write.
            # We write our own memory to the other logits
            other_message=inputs.self_message,
        )

        return action
