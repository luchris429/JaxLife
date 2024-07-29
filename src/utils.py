import jax
import jax.numpy as jnp


def tree_where(mask, tree_x, tree_y):
    """
    Applies jnp.where(mask, x, y) on tree_x and tree_y, reshaping mask as required.
    """

    def apply_mask(x, y, mask):
        # If the shape of x has more dimensions than the mask, reshape the mask
        if len(x.shape) > len(mask.shape):
            new_shape = list(mask.shape) + [1] * (len(x.shape) - len(mask.shape))
            mask = mask.reshape(new_shape)
        return jnp.where(mask, x, y)

    return jax.tree_map(lambda x, y: apply_mask(x, y, mask), tree_x, tree_y)
