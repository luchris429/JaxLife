import flax.linen as nn
from world.structs import Action


def logits_to_actions(action_logits_a, config):
    i = 6
    j = i + 1
    l = config["MESSAGE_SIZE"]
    actions = Action(
        x_move=action_logits_a[0],
        y_move=action_logits_a[1],
        push=action_logits_a[2],
        eat=action_logits_a[3] > 0,
        reproduce=action_logits_a[4] > 0,
        hit=action_logits_a[5],
        read_terrain_bit=0.0,
        message_other=nn.relu(action_logits_a[i]),
        self_message=action_logits_a[j : j + l],
        other_message=action_logits_a[j + l : j + l * 2],
        terrain_energy_gain=action_logits_a[j + l * 2],
        terrain_move_cost=action_logits_a[j + l * 2 + 1],
        terrain_push_cost=action_logits_a[j + l * 2 + 2],
        terrain_message_cost=action_logits_a[j + l * 2 + 3],
        terrain_reproduce_cost=action_logits_a[j + l * 2 + 4],
    )
    return actions
