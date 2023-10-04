import jax.numpy as jnp
import jax
import chex
from functools import partial

from ecnf.utils.graph import get_senders_and_receivers_fully_connected
from ecnf.utils.numerical import safe_norm

def energy(x, a = 0.0, b = -4., c = 0.9, d0 = 4.0, tau = 1.0):
    """Compute energy. Default hyper-parameters from https://arxiv.org/pdf/2006.02425.pdf.
    If we want to add conditioning info we could condition on the parameters a,b,c,d,tau. """
    n_nodes, dim = x.shape
    senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)
    vectors = x[senders] - x[receivers]
    differences = safe_norm(vectors, axis=-1)
    diff_minus_d0 = differences - d0
    energy = jnp.sum(a * diff_minus_d0 + b * diff_minus_d0 ** 2 + c * diff_minus_d0 ** 4, axis=0) / tau / 2
    chex.assert_shape(energy, ())
    return energy


def log_prob_fn(x: chex.Array, temperature=1.0):
    if len(x.shape) == 2:
        return - energy(x, tau=temperature)
    elif len(x.shape) == 3:
        return - jax.vmap(partial(energy, tau=temperature))(x)
    else:
        raise Exception
