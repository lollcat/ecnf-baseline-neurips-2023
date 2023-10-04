from typing import Callable, Tuple, Optional, Union, Any

import chex
import jax
import jax.numpy as jnp

from ecnf.utils.numerical import get_leading_axis_tree


def calculate_forward_ess(log_w: chex.Array, mask: chex.Array) -> dict:
    """Calculate forward ess.
    log_w = p(x)/q(x), where x ~ p(x).
    This can be passed as the `further_fn` to `eacf.train.base.eval_fn`."""
    chex.assert_equal_shape((log_w, mask))
    log_w = jnp.where(mask, log_w, jnp.zeros_like(log_w))  # make sure log_w finite
    log_z_inv = jax.nn.logsumexp(-log_w, b=mask) - jnp.log(jnp.sum(mask))
    log_z_expectation_p_over_q = jax.nn.logsumexp(log_w, b=mask) - jnp.log(jnp.sum(mask))
    log_forward_ess = - log_z_inv - log_z_expectation_p_over_q
    forward_ess = jnp.exp(log_forward_ess)
    info = {}
    info.update(forward_ess=forward_ess)
    return info


def setup_padded_reshaped_data(data: chex.ArrayTree,
                               interval_length: int,
                               reshape_axis=0) -> Tuple[chex.ArrayTree, chex.Array]:
    test_set_size = jax.tree_util.tree_flatten(data)[0][0].shape[0]
    chex.assert_tree_shape_prefix(data, (test_set_size, ))

    padding_amount = (interval_length - test_set_size % interval_length) % interval_length
    test_data_padded_size = test_set_size + padding_amount
    test_data_padded = jax.tree_map(
        lambda x: jnp.concatenate([x, jnp.zeros((padding_amount, *x.shape[1:]), dtype=x.dtype)], axis=0), data
    )
    mask = jnp.zeros(test_data_padded_size, dtype=int).at[jnp.arange(test_set_size)].set(1)


    if reshape_axis == 0:  # Used for pmap.
        test_data_reshaped, mask = jax.tree_map(
            lambda x: jnp.reshape(x, (interval_length, test_data_padded_size // interval_length, *x.shape[1:])),
            (test_data_padded, mask)
        )
    else:
        assert reshape_axis == 1  # for minibatching
        test_data_reshaped, mask = jax.tree_map(
            lambda x: jnp.reshape(x, (test_data_padded_size // interval_length, interval_length, *x.shape[1:])),
            (test_data_padded, mask)
        )
    return test_data_reshaped, mask


# Evaluation
Data = chex.ArrayTree
Mask = chex.Array
FurtherData = Any


def eval_fn(
    x: Data,
    key: chex.PRNGKey,
    eval_on_test_batch_fn: Optional[
            Callable[[Data, chex.PRNGKey, Mask], Union[Tuple[FurtherData, dict], dict]]
        ] = None,
    eval_batch_free_fn: Optional[Callable[[chex.PRNGKey], dict]] = None,
    batch_size: Optional[int] = None,
    mask: Optional[Mask] = None,
) -> Tuple[dict, Optional[FurtherData], Optional[Mask]]:
    info = {}
    key1, key2 = jax.random.split(key)

    n_data_points = get_leading_axis_tree(x)[0]
    if mask is None:
        mask = jnp.ones(n_data_points, dtype=int)

    if eval_on_test_batch_fn is not None:

        def scan_fn(carry, xs):
            # Scan over data in the test set. Vmapping all at once causes memory issues I think?
            x_batch, mask, key = xs
            info = eval_on_test_batch_fn(
                x_batch,
                key=key,
                mask=mask
            )
            return None, info

        (x_batched, mask_batched), mask_batched_new = \
            setup_padded_reshaped_data((x, mask), interval_length=batch_size, reshape_axis=1)
        mask_batched = mask_batched * mask_batched_new

        _, batched_info = jax.lax.scan(
            scan_fn,
            None,
            (x_batched, mask_batched, jax.random.split(key1, get_leading_axis_tree(x_batched, 1)[0])),
        )
        if isinstance(batched_info, dict):
            # Aggregate test set info across batches.
            per_batch_weighting = jnp.sum(mask_batched, axis=-1) / jnp.sum(jnp.sum(mask_batched, axis=-1))
            info.update(jax.tree_map(lambda x: jnp.sum(per_batch_weighting*x), batched_info))
            flat_mask, further_info = None, None
        else:
            per_batch_weighting = jnp.sum(mask_batched, axis=-1) / jnp.sum(jnp.sum(mask_batched, axis=-1))
            info.update(jax.tree_map(lambda x: jnp.sum(per_batch_weighting * x), batched_info[1]))
            flat_mask, further_info = jax.tree_map(lambda x: x.reshape(x.shape[0]*x.shape[1],
                                                                        *x.shape[2:]), (mask_batched, batched_info[0]))


    if eval_batch_free_fn is not None:
        non_batched_info = eval_batch_free_fn(
            key=key2,
        )
        info.update(non_batched_info)

    return info, further_info, flat_mask
