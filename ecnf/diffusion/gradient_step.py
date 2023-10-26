from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax
from functools import partial

from ecnf.diffusion.core import DiffusionModel
from ecnf.diffusion.loss import score_matching_loss_fn


class TrainingState(NamedTuple):
    params: chex.ArrayTree
    opt_state: optax.OptState
    key: chex.PRNGKey
    ema_params: Optional[chex.ArrayTree] = None


@partial(jax.jit, static_argnums=(0, 1))
def score_matching_update_fn(
        diffusion_model: DiffusionModel,
        opt_update: optax.TransformUpdateFn,
        state: TrainingState,
        x_data: chex.Array,
        features: Optional[chex.Array] = None,
        ema_beta: float = 0.999
):

    key, subkey = jax.random.split(state.key)
    grads, info = jax.grad(score_matching_loss_fn, has_aux=True, argnums=1)(
        diffusion_model,
        state.params,
        x_data,
        subkey,
        features
    )

    updates, new_opt_state = opt_update(grads, state.opt_state, params=state.params)
    new_params = optax.apply_updates(state.params, updates)
    info.update(
        grad_norm=optax.global_norm(grads),
        update_norm=optax.global_norm(updates),
    )

    if not isinstance(state.ema_params, jnp.ndarray):
        ema_fn = lambda theta_bar, theta_t: theta_bar * ema_beta + (1 - ema_beta)*theta_t
        ema_params = jax.tree_map(ema_fn, state.ema_params, new_params)
    else:
        ema_params = state.ema_params

    return TrainingState(params=new_params, opt_state=new_opt_state, key=key,
                         ema_params=ema_params), info
