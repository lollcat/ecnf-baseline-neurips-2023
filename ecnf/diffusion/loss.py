# Adapted from https://colab.research.google.com/drive/1SeXMpILhkJPjXUaesvzEhc3Ke6Zl_zxJ?usp=sharing#scrollTo=zOsoqPdXHuL5.

from typing import Optional, Tuple

import chex
import jax.numpy as jnp
import jax

from ecnf.diffusion.core import DiffusionModel

def score_matching_loss_fn(
        model: DiffusionModel,
        params: chex.Array,
        x: chex.Array,
        key: chex.PRNGKey,
        features: Optional[chex.Array] = None,
) -> Tuple[chex.Array, dict]:
    key, subkey = jax.random.split(key)
    random_t = jax.random.uniform(subkey, (x.shape[0],), minval=model.eps, maxval=1.)
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, x.shape)
    if model.zero_mean:
        z = z - jnp.mean(z, axis=1, keepdims=True)
    std = model.marginal_prob_std_fn(random_t)
    perturbed_x = x + z * std[:, None]
    score = model.apply(params, perturbed_x, random_t, features)
    loss = jnp.mean(jnp.sum((score * std[:, None] + z) ** 2, axis=1))
    info = {}
    info.update(loss=loss)
    return loss, info
