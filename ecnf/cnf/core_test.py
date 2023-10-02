import jax.numpy as jnp
import distrax
from functools import partial

from ecnf.cnf.core import FlowMatchingCNF, optimal_transport_conditional_vf


if __name__ == '__main__':
    dim = 3
    batch_size = 11

    sigma_min = 1e-3
    base_scale = 100
    base = distrax.MultivariateNormalDiag(loc=jnp.zeros(dim), scale_diag=jnp.ones(dim) * base_scale)

    get_cond_vector_field = partial(optimal_transport_conditional_vf, sigma_min=sigma_min)

    cnf = FlowMatchingCNF(
        init=lambda key, x: jnp.zeros(()),
        apply=lambda params, x, t: -x,
        sample_base=base._sample_n,
        get_x_t_and_conditional_u_t=get_cond_vector_field,
        log_prob_base=base.log_prob,
        sample_and_log_prob_base=base.sample_and_log_prob
    )



    x1 = jnp.ones(dim) + 0.1
    x0 = jnp.zeros(dim) + 0.1

    t = 0.2
    x_t, u_t = get_cond_vector_field(x0, x1, t)
