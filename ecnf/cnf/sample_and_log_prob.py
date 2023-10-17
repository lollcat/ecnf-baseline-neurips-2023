from typing import Optional, Tuple

import chex
import jax.numpy as jnp
import jax
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController

from ecnf.cnf.core import FlowMatchingCNF


def sample_cnf(cnf: FlowMatchingCNF, params: chex.ArrayTree,
               key: chex.PRNGKey, features: Optional[chex.Array] = None,
               use_fixed_step_size: bool = False,
               rtol: float = 1e-5,
               atol: float = 1e-5,
               step_size: float = 0.05,
               ) -> chex.Array:

    def f(t: chex.Array, y: chex.Array, args: None) -> chex.Array:
        feat = features[None] if features is not None else None

        x = y
        chex.assert_rank(t, 0)
        chex.assert_rank(x, 1)
        vector_field = cnf.apply(params, x[None], t[None], feat)
        return jnp.squeeze(vector_field, axis=0)

    term = ODETerm(f)
    solver = Dopri5()
    x0 = jnp.squeeze(cnf.sample_base(key, 1), axis=0)

    if use_fixed_step_size:
        solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=step_size, y0=x0)
    else:
        solution = diffeqsolve(term, solver, t0=0, t1=1, y0=x0,
                           stepsize_controller=PIDController(rtol=rtol, atol=atol, dtmin=1e-5),
                               dt0=None)
    return jnp.squeeze(solution.ys, axis=0)


def get_log_prob(
        cnf: FlowMatchingCNF,
        params: chex.ArrayTree,
        x: chex.Array,
        key: chex.PRNGKey,
        features: Optional[chex.Array] = None,
        approx: bool = False,
        use_fixed_step_size: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        step_size: float = 0.05,
) -> chex.Array:
    features = features[None] if features is not None else None

    eps = jax.random.normal(key, x.shape)

    if not approx:
        def joint_vector_field(t: chex.Array, y: chex.Array, args: None) -> Tuple[chex.Array, chex.Array]:
            """For inverting flow, keeping track of volume change."""
            x, log_p_so_far = y
            chex.assert_rank(t, 0)
            chex.assert_rank(x, 1)
            vector_field_x_fn = lambda x: jnp.squeeze(cnf.apply(params, x[None], t[None], features), axis=0)
            vector_field, vjp_fn = jax.vjp(vector_field_x_fn, x)
            (dfdy,) = jax.vmap(vjp_fn)(jnp.eye(vector_field.shape[0]))
            div = jnp.trace(dfdy)
            return vector_field, div
    else:
        def joint_vector_field(t: chex.Array, y: chex.Array, args: None) -> Tuple[chex.Array, chex.Array]:
            """For inverting flow, keeping track of volume change."""
            x, log_p_so_far = y
            chex.assert_rank(t, 0)
            chex.assert_rank(x, 1)
            vector_field_x_fn = lambda x: jnp.squeeze(cnf.apply(params, x[None], t[None], features), axis=0)
            vector_field, vjp_fn = jax.vjp(vector_field_x_fn, x)
            (eps_dfdy,) = vjp_fn(eps)
            approx_div = jnp.sum(eps_dfdy * eps)
            return vector_field, approx_div


    term = ODETerm(joint_vector_field)
    solver = Dopri5()

    if use_fixed_step_size:
        solution = diffeqsolve(term, solver, t0=1., t1=0., dt0=-step_size, y0=(x, jnp.zeros(())))
    else:
        solution = diffeqsolve(term, solver, t0=1., t1=0., y0=(x, jnp.zeros(())),
                               stepsize_controller=PIDController(rtol=rtol, atol=atol, dtmin=1e-5),
                               dt0=None)
    x0 = jnp.squeeze(solution.ys[0], axis=0)
    delta_log_likelihood = jnp.squeeze(solution.ys[1], axis=0)
    log_p = cnf.log_prob_base(x0) + delta_log_likelihood
    return log_p



def sample_and_log_prob_cnf(
        cnf: FlowMatchingCNF,
        params: chex.ArrayTree,
        key: chex.PRNGKey,
        features: Optional[chex.Array] = None,
        approx: bool = False,
        use_fixed_step_size: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        step_size: float = 0.05,
) -> Tuple[chex.Array, chex.Array]:

    features = features[None] if features is not None else None

    if not approx:
        def joint_vector_field(t: chex.Array, y: chex.Array, args: None) -> Tuple[chex.Array, chex.Array]:
            """For inverting flow, keeping track of volume change."""
            x, log_p_so_far = y
            chex.assert_rank(t, 0)
            chex.assert_rank(x, 1)
            vector_field_x_fn = lambda x: jnp.squeeze(cnf.apply(params, x[None], t[None], features), axis=0)
            vector_field, vjp_fn = jax.vjp(vector_field_x_fn, x)
            (dfdy,) = jax.vmap(vjp_fn)(jnp.eye(vector_field.shape[0]))
            div = jnp.trace(dfdy)
            return vector_field, div
    else:
        def joint_vector_field(t: chex.Array, y: chex.Array, args: None) -> Tuple[chex.Array, chex.Array]:
            """For inverting flow, keeping track of volume change."""
            x, log_p_so_far = y
            chex.assert_rank(t, 0)
            chex.assert_rank(x, 1)
            vector_field_x_fn = lambda x: jnp.squeeze(cnf.apply(params, x[None], t[None], features), axis=0)
            vector_field, vjp_fn = jax.vjp(vector_field_x_fn, x)
            eps = jax.random.normal(key, x.shape)
            (eps_dfdy,) = vjp_fn(eps)
            approx_div = jnp.sum(eps_dfdy * eps)
            return vector_field, approx_div

    term = ODETerm(joint_vector_field)
    solver = Dopri5()
    x0, log_prob_base = cnf.sample_and_log_prob_base(seed=key, sample_shape=())

    if use_fixed_step_size:
        solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=step_size, y0=x0)
    else:
        solution = diffeqsolve(term, solver, t0=0., t1=1., y0=(x0, jnp.zeros(())),
                               stepsize_controller=PIDController(rtol=rtol, atol=atol, dtmin=1e-5),
                               dt0=None)
    x1 = jnp.squeeze(solution.ys[0], axis=0)
    delta_log_likelihood = jnp.squeeze(solution.ys[1], axis=0)
    log_p = cnf.log_prob_base(x0) - delta_log_likelihood

    return x1, log_p
