"""Build's CNF for application to Cartesian coordinates of molecules."""
from typing import Callable, Sequence

from functools import partial

import jax.numpy as jnp
from jax.tree_util import tree_unflatten
import jax
import distrax
import chex
from flax import linen as nn

from ecnf.cnf.core import FlowMatchingCNF, optimal_transport_conditional_vf
from ecnf.nets.egnn import EGNN
from ecnf.cnf.zero_com_base import FlatZeroCoMGaussian


def get_timestep_embedding(timesteps: chex.Array, embedding_dim: int):
    """Build sinusoidal embeddings (from Fairseq)."""
    # https://colab.research.google.com/github/google-research/vdm/blob/main/colab/SimpleDiffusionColab.ipynb#scrollTo=O5rq6xovwhgP

    assert timesteps.ndim == 1
    timesteps = timesteps * 1000

    half_dim = embedding_dim // 2
    emb = jnp.log(10_000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def build_cnf(
        n_frames: int,
        dim: int,
        sigma_min: float,
        base_scale: float,
        n_blocks_egnn: int,
        mlp_units: Sequence[int],
        n_invariant_feat_hidden: int,
        time_embedding_dim: int,
):

    base = distrax.Transformed(distribution=FlatZeroCoMGaussian(dim=dim, n_nodes=n_frames),
                               bijector=distrax.Block(
                                   distrax.ScalarAffine(
                                   shift=jnp.zeros(dim*n_frames), scale=jnp.ones(dim*n_frames)*base_scale),
                                   ndims=1
                               ))
    get_cond_vector_field = partial(optimal_transport_conditional_vf, sigma_min=sigma_min)

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
            return flat_vectors



    net = FlatEgnn()
    cnf = FlowMatchingCNF(init=net.init, apply=net.apply, get_x_t_and_conditional_u_t=get_cond_vector_field,
                          sample_base=base._sample_n, sample_and_log_prob_base=base.sample_and_log_prob,
                          log_prob_base=base.log_prob)

    return cnf
