from typing import Union

import chex
import jax.numpy as jnp
import jax

from ecnf.utils.graph import get_senders_and_receivers_fully_connected
from ecnf.utils.numerical import safe_norm

def energy(x: chex.Array, epsilon: float = 1.0, tau: float = 1.0, r: Union[float, chex.Array] = 1.0,
           harmonic_potential_coef: float = 0.5) -> chex.Array:
    chex.assert_rank(x, 2)
    n_nodes, dim = x.shape
    if isinstance(r, float):
        r = jnp.ones(n_nodes) * r
    senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)
    vectors = x[senders] - x[receivers]
    d = safe_norm(vectors, axis=-1)
    term_inside_sum = (r[receivers] / d)**12 - 2*(r[receivers] / d)**6
    energy = epsilon / (2 * tau) * jnp.sum(term_inside_sum)

    # For harmonic potential see https://github.com/vgsatorras/en_flows/blob/main/deprecated/eqnode/test_systems.py#L94.
    # This oscillator is mentioned but not explicity specified in the paper where it was introduced:
    # http://proceedings.mlr.press/v119/kohler20a/kohler20a.pdf.
    centre_of_mass = jnp.mean(x, axis=0)
    harmonic_potential = harmonic_potential_coef*jnp.sum((x - centre_of_mass)**2)
    return energy + harmonic_potential


def log_prob_fn(x: chex.Array):
    if len(x.shape) == 2:
        return - energy(x)
    elif len(x.shape) == 3:
        return - jax.vmap(energy)(x)
    else:
        raise Exception