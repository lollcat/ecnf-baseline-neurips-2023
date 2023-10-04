import jax.numpy as jnp
import distrax
from functools import partial

import jax.random

from ecnf.cnf.core import FlowMatchingCNF, optimal_transport_conditional_vf
from ecnf.cnf.sample_and_log_prob import sample_and_log_prob_cnf, get_log_prob


if __name__ == '__main__':
    dim = 3
    batch_size = 11

    sigma_min = 1e-3
    base_scale = 100
    base = distrax.MultivariateNormalDiag(loc=jnp.zeros(dim), scale_diag=jnp.ones(dim) * base_scale)

    get_cond_vector_field = partial(optimal_transport_conditional_vf, sigma_min=sigma_min)

    cnf = FlowMatchingCNF(
        init=lambda key, x, t: jnp.zeros(()),
        apply=lambda params, x, t, feat: x*2 + x,
        sample_base=base._sample_n,
        get_x_t_and_conditional_u_t=get_cond_vector_field,
        log_prob_base=base.log_prob,
        sample_and_log_prob_base=base.sample_and_log_prob
    )



    x1 = jnp.ones(dim) + 0.1
    x0 = jnp.zeros(dim) + 0.1

    t = 0.2
    x_t, u_t = get_cond_vector_field(x0, x1, t)

    # Check sample and log prob is correct.
    key = jax.random.PRNGKey(0)
    params = cnf.init(key, x0[None], jnp.zeros(()))

    xT, log_q = sample_and_log_prob_cnf(cnf=cnf, params=params, key=key, features=None, approx=False)
    log_q_ = get_log_prob(cnf=cnf, params=params,  x=xT, key=key, features=None, approx=False)

