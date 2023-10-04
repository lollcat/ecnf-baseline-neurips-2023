import chex
import jax.random
import jax.numpy as jnp

from ecnf.nets.egnn import EGNN
from ecnf.utils.test import assert_function_is_equivariant


if __name__ == '__main__':
    n_nodes = 5
    dim = 3
    key = jax.random.PRNGKey(0)

    egnn = EGNN(
        name='dogfish',
        n_blocks=2,
        mlp_units=(16, 16),
        n_invariant_feat_hidden=32,
    )

    dummy_pos = jnp.ones((n_nodes, dim))
    dummy_feat = jnp.ones((n_nodes, 2))
    dummy_time_embed = jnp.ones(11)

    params = egnn.init(key, dummy_pos, dummy_feat, dummy_time_embed)

    def eq_fn(pos: chex.Array) -> chex.Array:
        return egnn.apply(params, pos, dummy_feat, dummy_time_embed)


    assert_function_is_equivariant(equivariant_fn=eq_fn, n_nodes=n_nodes, dim=dim)
