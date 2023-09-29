from typing import Tuple, Callable, Optional

import chex
import jax.random
import jax.numpy as jnp

from cnf.core import FlowMatchingCNF


def flow_matching_loss_fn(
        cnf: FlowMatchingCNF,
        params: chex.ArrayTree,
        x_data: chex.Array,
        key: chex.PRNGKey,
        features: Optional[chex.Array] = None
) -> Tuple[chex.Array, dict]:
    key1, key2 = jax.random.split(key)
    batch_size = x_data.shape[0]
    x0 = cnf.sample_base(key1, batch_size)
    t = jax.random.uniform(key2, shape=(batch_size,))
    x_t, u_t_conditional = jax.vmap(cnf.get_x_t_and_conditional_u_t)(x0, x_data, t)
    v_t = cnf.apply(params, x_data, t, features)

    loss = jnp.mean((v_t - u_t_conditional)**2)
    info = {}
    info.update(loss=loss)
    return loss, info
