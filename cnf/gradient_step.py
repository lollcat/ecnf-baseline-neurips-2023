from typing import NamedTuple, Optional

import chex
import jax
import optax
from functools import partial

from cnf.core import FlowMatchingCNF
from cnf.loss import flow_matching_loss_fn


class TrainingState(NamedTuple):
    params: chex.ArrayTree
    opt_state: optax.OptState
    key: chex.PRNGKey


@partial(jax.jit, static_argnums=(0, 1))
def flow_matching_update_fn(
        cnf: FlowMatchingCNF,
        opt_update: optax.TransformUpdateFn,
        state: TrainingState,
        x_data: chex.Array,
        features: Optional[chex.Array] = None):

    key, subkey = jax.random.split(state.key)
    grads, info = jax.grad(flow_matching_loss_fn, has_aux=True, argnums=1)(
        cnf,
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

    return TrainingState(params=new_params, opt_state=new_opt_state, key=key), info
