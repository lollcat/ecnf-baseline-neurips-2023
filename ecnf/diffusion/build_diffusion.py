"""Build's CNF for application to Cartesian coordinates of molecules."""
from typing import Callable, Sequence

from functools import partial

import jax.numpy as jnp
from jax.tree_util import tree_unflatten
import jax
import distrax
import chex
from flax import linen as nn

from ecnf.cnf.build_cnf import get_timestep_embedding
from ecnf.diffusion.core import DiffusionModel, marginal_prob_std
from ecnf.nets.egnn import EGNN



def build_diffusion(
        n_frames: int,
        dim: int,
        n_blocks_egnn: int,
        mlp_units: Sequence[int],
        n_invariant_feat_hidden: int,
        time_embedding_dim: int,
        base_scale: float = 25.
):
    sigma = base_scale

    class FlatEgnn(nn.Module):

        @nn.compact
        def __call__(self,
                     positions: chex.Array,
                     time: chex.Array,
                     node_features: chex.Array
                     ) -> chex.Array:
            chex.assert_rank(positions, 2)
            chex.assert_rank(node_features, 2)
            chex.assert_rank(time, 1)

            positions = jnp.reshape(positions, (positions.shape[0], n_frames, dim))
            node_features = jnp.reshape(node_features, (node_features.shape[0], n_frames, -1))
            time_embedding = get_timestep_embedding(time, time_embedding_dim)

            net = EGNN(
                n_blocks=n_blocks_egnn,
                mlp_units=mlp_units,
                n_invariant_feat_hidden=n_invariant_feat_hidden,
            )

            vectors = net(positions, node_features, time_embedding)
            flat_vectors = jnp.reshape(vectors, (vectors.shape[0], n_frames*dim))

            sigma_t = marginal_prob_std(time, sigma)
            flat_vectors = flat_vectors / sigma_t
            return flat_vectors



    net = FlatEgnn()
    cnf = DiffusionModel(
        init=net.init,
        apply=net.apply,
        sigma=sigma,
        zero_mean=True
    )

    return cnf
